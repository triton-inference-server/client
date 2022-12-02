// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "data_loader.h"
#include "model_parser.h"

namespace triton { namespace perfanalyzer {

// Holds information about the shared memory locations
struct SharedMemoryData {
  SharedMemoryData(
      size_t byte_size,
      std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> data)
      : byte_size_(byte_size), data_(std::move(data))
  {
  }

  SharedMemoryData() {}

  // Byte size
  size_t byte_size_;

  // Unique pointer holding the shared memory data
  std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> data_;
};

// Holds the status of the inflight sequence
struct SequenceStat {
  SequenceStat(uint64_t seq_id)
      : seq_id_(seq_id), data_stream_id_(0), remaining_queries_(0)
  {
  }
  // If paused, no more requests should be sent other than a single request to
  // finish the active sequence
  bool paused_ = false;
  // The unique correlation id allocated to the sequence
  uint64_t seq_id_;
  // The data stream id providing data for the sequence
  uint64_t data_stream_id_;
  // The number of queries remaining to complete the sequence
  size_t remaining_queries_;
  // A lock to protect sequence data
  std::mutex mtx_;
};

// Holds the running status of the thread.
struct ThreadStat {
  ThreadStat() {}

  // The status of the worker thread
  cb::Error status_;
  // The status of the callback thread for async requests
  cb::Error cb_status_;
  // The statistics of the InferContext
  std::vector<cb::InferStat> contexts_stat_;
  // The concurrency level that the worker should produce
  size_t concurrency_;
  // A vector of request timestamps <start_time, end_time>
  // Request latency will be end_time - start_time
  TimestampVector request_timestamps_;
  // A lock to protect thread data
  std::mutex mu_;
};

/// Wraps the information required to send an inference to the
/// server
struct InferContext {
  explicit InferContext() : inflight_request_cnt_(0) {}
  InferContext(InferContext&&) = delete;
  InferContext(const InferContext&) = delete;
  ~InferContext()
  {
    for (const auto input : inputs_) {
      delete input;
    }
    for (const auto output : outputs_) {
      delete output;
    }
  }
  // The backend to communicate with the server
  std::unique_ptr<cb::ClientBackend> infer_backend_;
  // The vector of pointers to InferInput objects for all possible inputs,
  // potentially including optional inputs with no provided data.
  std::vector<cb::InferInput*> inputs_;
  // The vector of pointers to InferInput objects to be
  // used for inference request.
  std::vector<cb::InferInput*> valid_inputs_;
  // The vector of pointers to InferRequestedOutput objects
  // to be used with the inference request.
  std::vector<const cb::InferRequestedOutput*> outputs_;
  // If not empty, the expected output data in the same order as 'outputs_'
  std::vector<std::vector<std::pair<const uint8_t*, size_t>>> expected_outputs_;
  // The InferOptions object holding the details of the
  // inference.
  std::unique_ptr<cb::InferOptions> options_;
  // The total number of inference in-flight.
  std::atomic<size_t> inflight_request_cnt_;
};

/// The properties of an asynchronous request required in
/// the callback to effectively interpret the response.
struct AsyncRequestProperties {
  AsyncRequestProperties() : sequence_end_(false), delayed_(true) {}
  // The id of in the inference context which was used to
  // send this request.
  uint32_t ctx_id_;
  // The timestamp of when the request was started.
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  // Whether or not the request is at the end of a sequence.
  bool sequence_end_;
  // Whether or not the request is delayed as per schedule.
  bool delayed_;
};


/// Abstract base class for worker threads
///
class Worker {
 protected:
  Worker(
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<DataLoader> data_loader,
      const std::shared_ptr<cb::ClientBackendFactory> factory,
      std::vector<std::shared_ptr<SequenceStat>>& sequence_stat,
      std::unordered_map<std::string, SharedMemoryData>& shared_memory_regions,
      const cb::BackendKind backend_kind,
      const SharedMemoryType shared_memory_type, const bool on_sequence_model,
      const bool async, const bool streaming, const int32_t batch_size,
      const bool using_json_data, const size_t sequence_length,
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      std::atomic<uint64_t>& curr_seq_id,
      std::uniform_int_distribution<uint64_t>& distribution,
      std::condition_variable& wake_signal, std::mutex& wake_mutex,
      bool& execute)
      : parser_(parser), data_loader_(data_loader), factory_(factory),
        sequence_stat_(sequence_stat),
        shared_memory_regions_(shared_memory_regions),
        backend_kind_(backend_kind), shared_memory_type_(shared_memory_type),
        on_sequence_model_(on_sequence_model), async_(async),
        streaming_(streaming), batch_size_(batch_size),
        using_json_data_(using_json_data), sequence_length_(sequence_length),
        start_sequence_id_(start_sequence_id),
        sequence_id_range_(sequence_id_range), curr_seq_id_(curr_seq_id),
        distribution_(distribution), wake_signal_(wake_signal),
        wake_mutex_(wake_mutex), execute_(execute)
  {
  }

  ~Worker() = default;
  Worker(Worker&) = delete;

  /// Helper function to prepare the InferContext for sending
  /// inference request. \param ctx The target InferContext object.
  /// \return cb::Error object indicating success or failure.
  cb::Error PrepareInfer(InferContext* ctx);

  /// Helper function to prepare the InferContext for sending
  /// inference request in shared memory. \param ctx The target
  /// InferContext object. \return cb::Error object indicating
  /// success or failure.
  cb::Error PrepareSharedMemoryInfer(InferContext* ctx);

  /// Updates the input data to use for inference request
  /// \param inputs The vector of pointers to InferInput objects for all
  /// possible inputs, potentially including optional inputs with no provided
  /// data
  /// \param valid_inputs The vector of pointers to InferInput objects to be
  /// used for inference request.
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error UpdateInputs(
      const std::vector<cb::InferInput*>& inputs,
      std::vector<cb::InferInput*>& valid_inputs, int stream_index,
      int step_index);

  /// Updates the expected output data to use for inference request. Empty
  /// vector will be returned if there is no expected output associated to the
  /// step.
  /// \param outputs The vector of outputs to get the expected data
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \param data The vector of pointer and size of the expected outputs
  /// \return cb::Error object indicating success or failure.
  cb::Error UpdateValidationOutputs(
      const std::vector<const cb::InferRequestedOutput*>& outputs,
      int stream_index, int step_index,
      std::vector<std::vector<std::pair<const uint8_t*, size_t>>>& data);

  cb::Error ValidateOutputs(
      const InferContext& ctx, const cb::InferResult* result_ptr);

  void SetInferSequenceOptions(
      const uint32_t seq_stat_index,
      std::unique_ptr<cb::InferOptions>& options);
  void InitNewSequence(int seq_stat_index);
  uint64_t GetNextSeqId(int seq_stat_index);

  /// Generate random sequence length based on 'offset_ratio' and
  /// 'sequence_length_'. (1 +/- 'offset_ratio') * 'sequence_length_'
  /// \param offset_ratio The offset ratio of the generated length
  /// \return random sequence length
  size_t GetRandomSequenceLength(double offset_ratio);

  /// Helper function to update the inputs
  /// \param inputs The vector of pointers to InferInput objects for all
  /// possible inputs, potentially including optional inputs with no provided
  /// data
  /// \param valid_inputs The vector of pointers to InferInput objects to be
  /// used for inference request.
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputs(
      const std::vector<cb::InferInput*>& inputs,
      std::vector<cb::InferInput*>& valid_inputs, const int stream_index,
      const int step_index);

  /// Helper function to update the shared memory inputs
  /// \param inputs The vector of pointers to InferInput objects
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputsSharedMemory(
      const std::vector<cb::InferInput*>& inputs, const int stream_index,
      const int step_index);

 protected:
  // TODO REFACTOR all sequence related code should be in a single class. We
  // shouldn't have to have a shared uint64 reference passed to all threads
  // Current sequence id (for issuing new sequences)
  std::atomic<uint64_t>& curr_seq_id_;

  // TODO REFACTOR this created in load manager init in one case. Can we
  // decouple? Used to pick among multiple data streams
  std::uniform_int_distribution<uint64_t>& distribution_;

  // TODO REFACTOR is there a better way to do threading than to pass the same
  // cv/mutex into every thread by reference?
  // Used to wake up this thread if it has been put to sleep
  std::condition_variable& wake_signal_;
  std::mutex& wake_mutex_;

  // TODO REFACTOR is there a better way to communicate this than a shared bool
  // reference?
  // Used to pause execution of this thread
  bool& execute_;

  // Map from shared memory key to its starting address and size
  std::unordered_map<std::string, SharedMemoryData>& shared_memory_regions_;
  // Sequence stats for all sequences
  std::vector<std::shared_ptr<SequenceStat>>& sequence_stat_;
  std::shared_ptr<DataLoader> data_loader_;
  const std::shared_ptr<ModelParser> parser_;
  const std::shared_ptr<cb::ClientBackendFactory> factory_;

  const cb::BackendKind backend_kind_;
  const SharedMemoryType shared_memory_type_;
  const bool on_sequence_model_;
  const bool async_;
  const bool streaming_;
  const int32_t batch_size_;
  const bool using_json_data_;
  const uint64_t start_sequence_id_;
  const uint64_t sequence_id_range_;
  const size_t sequence_length_;

 private:
  std::default_random_engine rng_generator_;
};

}}  // namespace triton::perfanalyzer
