// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <random>
#include <thread>
#include "client_backend/client_backend.h"
#include "data_loader.h"
#include "perf_utils.h"


namespace triton { namespace perfanalyzer {

class LoadManager {
 public:
  virtual ~LoadManager();

  /// Check if the load manager is working as expected.
  /// \return cb::Error object indicating success or failure.
  cb::Error CheckHealth();

  /// Swap the content of the timestamp vector recorded by the load
  /// manager with a new timestamp vector
  /// \param new_timestamps The timestamp vector to be swapped.
  /// \return cb::Error object indicating success or failure.
  cb::Error SwapTimestamps(TimestampVector& new_timestamps);

  /// Get the sum of all contexts' stat
  /// \param contexts_stat Returned the accumulated stat from all contexts
  /// in load manager
  cb::Error GetAccumulatedClientStat(cb::InferStat* contexts_stat);

  /// \return the batch size used for the inference requests
  size_t BatchSize() const { return batch_size_; }

  /// Resets all worker thread states to beginning of schedule.
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error ResetWorkers()
  {
    return cb::Error(
        "resetting worker threads not supported for this load manager.");
  }

  /// Count the number of requests collected until now.
  uint64_t CountCollectedRequests();

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
    // The vector of pointers to InferInput objects to be
    // used for inference request.
    std::vector<cb::InferInput*> inputs_;
    // The vector of pointers to InferRequestedOutput objects
    // to be used with the inference request.
    std::vector<const cb::InferRequestedOutput*> outputs_;
    // If not empty, the expected output data in the same order as 'outputs_'
    std::vector<std::vector<std::pair<const uint8_t*, size_t>>>
        expected_outputs_;
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
    struct timespec start_time_;
    // Whether or not the request is at the end of a sequence.
    bool sequence_end_;
    // Whether or not the request is delayed as per schedule.
    bool delayed_;
  };

 protected:
  LoadManager(
      const bool async, const bool streaming, const int32_t batch_size,
      const size_t max_threads, const size_t sequence_length,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory);

  /// Helper funtion to retrieve the input data for the inferences
  /// \param string_length The length of the random strings to be generated
  /// for string inputs.
  /// \param string_data The string to be used as string inputs for model.
  /// \param zero_input Whether to use zero for model inputs.
  /// \param user_data The vector containing path/paths to user-provided data
  /// that can be a directory or path to a json data file.
  /// \return cb::Error object indicating success or failure.
  cb::Error InitManagerInputs(
      const size_t string_length, const std::string& string_data,
      const bool zero_input, std::vector<std::string>& user_data);

  /// Helper function to allocate and prepare shared memory.
  /// from shared memory.
  /// \return cb::Error object indicating success or failure.
  cb::Error InitSharedMemory();

  /// Helper function to prepare the InferContext for sending inference request.
  /// \param ctx The target InferContext object.
  /// \return cb::Error object indicating success or failure.
  cb::Error PrepareInfer(InferContext* ctx);

  /// Helper function to prepare the InferContext for sending inference
  /// request in shared memory.
  /// \param ctx The target InferContext object.
  /// \return cb::Error object indicating success or failure.
  cb::Error PrepareSharedMemoryInfer(InferContext* ctx);

  /// Updates the input data to use for inference request
  /// \param inputs The vector of pointers to InferInput objects
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error UpdateInputs(
      std::vector<cb::InferInput*>& inputs, int stream_index, int step_index);

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
      const uint32_t seq_id, std::unique_ptr<cb::InferOptions>& options);
  void InitNewSequence(int sequence_id);

  /// Generate random sequence length based on 'offset_ratio' and
  /// 'sequence_length_'. (1 +/- 'offset_ratio') * 'sequence_length_'
  /// \param offset_ratio The offset ratio of the generated length
  /// \return random sequence length
  size_t GetRandomLength(double offset_ratio);

  /// Stops all the worker threads generating the request load.
  void StopWorkerThreads();

 private:
  /// Helper function to update the inputs
  /// \param inputs The vector of pointers to InferInput objects
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputs(
      const std::vector<cb::InferInput*>& inputs, const int stream_index,
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
  bool async_;
  bool streaming_;
  size_t batch_size_;
  size_t max_threads_;
  size_t sequence_length_;
  SharedMemoryType shared_memory_type_;
  size_t output_shm_size_;
  bool on_sequence_model_;

  const uint64_t start_sequence_id_;
  const uint64_t sequence_id_range_;

  std::shared_ptr<ModelParser> parser_;
  std::shared_ptr<cb::ClientBackendFactory> factory_;

  bool using_json_data_;
  bool using_shared_memory_;

  std::default_random_engine rng_generator_;
  std::uniform_int_distribution<uint64_t> distribution_;

  std::unique_ptr<DataLoader> data_loader_;
  std::unique_ptr<cb::ClientBackend> backend_;

  // Map from shared memory key to its starting address and size
  std::unordered_map<std::string, std::pair<uint8_t*, size_t>>
      shared_memory_regions_;

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

  // Holds the status of the inflight sequence
  struct SequenceStat {
    SequenceStat(uint64_t seq_id)
        : seq_id_(seq_id), data_stream_id_(0), remaining_queries_(0)
    {
    }
    // The unique correlation id allocated to the sequence
    uint64_t seq_id_;
    // The data stream id providing data for the sequence
    uint64_t data_stream_id_;
    // The number of queries remaining to complete the sequence
    size_t remaining_queries_;
    // A lock to protect sequence data
    std::mutex mtx_;
  };

  std::vector<std::shared_ptr<SequenceStat>> sequence_stat_;
  std::atomic<uint64_t> next_seq_id_;

  // Worker threads that loads the server with inferences
  std::vector<std::thread> threads_;
  // Contains the statistics on the current working threads
  std::vector<std::shared_ptr<ThreadStat>> threads_stat_;

  // Use condition variable to pause/continue worker threads
  std::condition_variable wake_signal_;
  std::mutex wake_mutex_;
};

}}  // namespace triton::perfanalyzer
