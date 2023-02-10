// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <mutex>
#include <vector>
#include "data_loader.h"
#include "infer_data.h"
#include "infer_data_manager.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

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
  // TODO REFACTOR TMA-1046 -- This should be in the InferContext class
  // The statistics of the InferContext
  std::vector<cb::InferStat> contexts_stat_;
  // A vector of request timestamps <start_time, end_time>
  // Request latency will be end_time - start_time
  TimestampVector request_timestamps_;
  // A lock to protect thread data
  std::mutex mu_;
};

/// The properties of an asynchronous request required in
/// the callback to effectively interpret the response.
struct AsyncRequestProperties {
  AsyncRequestProperties() : sequence_end_(false), delayed_(true) {}
  // The timestamp of when the request was started.
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  // Whether or not the request is at the end of a sequence.
  bool sequence_end_;
  // Whether or not the request is delayed as per schedule.
  bool delayed_;
};

/// Sends inference requests to the server
class InferContext {
 public:
  InferContext(
      const uint32_t id, const bool async, const bool streaming,
      const bool on_sequence_model, const bool using_json_data,
      const int32_t batch_size, const uint64_t start_sequence_id,
      const uint64_t sequence_id_range, const size_t sequence_length,
      std::atomic<uint64_t>& curr_seq_id,
      std::uniform_int_distribution<uint64_t>& distribution,
      std::shared_ptr<ThreadStat> thread_stat,
      std::vector<std::shared_ptr<SequenceStat>>& sequence_stat,
      std::shared_ptr<DataLoader> data_loader,
      std::shared_ptr<ModelParser> parser,
      std::shared_ptr<cb::ClientBackendFactory> factory,
      const std::shared_ptr<InferDataManager>& infer_data_manager)
      : id_(id), async_(async), streaming_(streaming),
        on_sequence_model_(on_sequence_model),
        using_json_data_(using_json_data), batch_size_(batch_size),
        start_sequence_id_(start_sequence_id),
        sequence_id_range_(sequence_id_range),
        sequence_length_(sequence_length), curr_seq_id_(curr_seq_id),
        distribution_(distribution), thread_stat_(thread_stat),
        sequence_stat_(sequence_stat), data_loader_(data_loader),
        parser_(parser), factory_(factory), data_step_id_(id),
        infer_data_manager_(infer_data_manager)
  {
    thread_stat_->status_ = factory_->CreateClientBackend(&infer_backend_);
    infer_data_.options_.reset(new cb::InferOptions(parser_->ModelName()));
    infer_data_.options_->model_version_ = parser_->ModelVersion();
    infer_data_.options_->model_signature_name_ = parser_->ModelSignatureName();

    thread_stat_->contexts_stat_.emplace_back();
  }

  InferContext(InferContext&&) = delete;
  InferContext(const InferContext&) = delete;

  // Initialize the context. Must be done before any inferences are sent
  void Init();

  // Send a single inference request to the server
  void SendInferRequest(bool delayed = false);

  // Send a single sequence inference request to the server
  void SendSequenceInferRequest(uint32_t seq_index, bool delayed = false);

  // Finish the active sequence at the given seq_stat_index
  void CompleteOngoingSequence(uint32_t seq_stat_index);

  // Returns the total number of async requests that have been sent by this
  // object and have not returned
  uint GetNumOngoingRequests() { return total_ongoing_requests_; }

  // Register a function that will get called after every async request returns
  void RegisterAsyncCallbackFinalize(std::function<void(uint32_t)> callback)
  {
    async_callback_finalize_func_ = callback;
  }

  // TODO REFACTOR TMA-1043 this should be in memory class
  void SetNumActiveThreads(size_t num_threads)
  {
    num_active_threads_ = num_threads;
  }

 protected:
  /// A helper function to issue inference request to the server.
  /// \param request_id The unique id to be associated with the request.
  /// \param delayed Whether the request fell behind its scheduled time.
  void SendRequest(const uint64_t request_id, const bool delayed);


  /// Update inputs based on custom json data
  void UpdateJsonData();

  /// Update inputs based on custom json data for the given sequence
  void UpdateSeqJsonData(std::shared_ptr<SequenceStat> seq_stat);

  void SetInferSequenceOptions(
      const uint32_t seq_stat_index,
      std::unique_ptr<cb::InferOptions>& options);
  void InitNewSequence(int seq_stat_index);

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

  cb::Error ValidateOutputs(const cb::InferResult* result_ptr);

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

  uint64_t GetNextSeqId(int seq_stat_index);

  /// Generate random sequence length based on 'offset_ratio' and
  /// 'sequence_length_'. (1 +/- 'offset_ratio') * 'sequence_length_'
  /// \param offset_ratio The offset ratio of the generated length
  /// \return random sequence length
  size_t GetRandomSequenceLength(double offset_ratio);

  // Callback function for handling asynchronous requests
  void AsyncCallbackFuncImpl(cb::InferResult* result);

  const bool async_;
  const bool streaming_;
  const bool on_sequence_model_;
  const bool using_json_data_;
  const int32_t batch_size_;
  const uint64_t start_sequence_id_;
  const uint64_t sequence_id_range_;
  const size_t sequence_length_;

  std::shared_ptr<ThreadStat> thread_stat_;
  std::vector<std::shared_ptr<SequenceStat>>& sequence_stat_;
  std::shared_ptr<DataLoader> data_loader_;
  std::shared_ptr<ModelParser> parser_;
  std::shared_ptr<cb::ClientBackendFactory> factory_;
  const std::shared_ptr<InferDataManager> infer_data_manager_;

  // TODO REFACTOR TMA-1019 this created in load manager init in one case. Can
  // we decouple? Used to pick among multiple data streams. Note this probably
  // gets absorbed into the new sequence class when it is created
  std::uniform_int_distribution<uint64_t>& distribution_;

  // TODO REFACTOR TMA-1019 all sequence related code should be in a single
  // class. We shouldn't have to have a shared uint64 reference passed to all
  // threads Current sequence id (for issuing new sequences)
  std::atomic<uint64_t>& curr_seq_id_;

  uint64_t request_id_ = 0;
  std::map<std::string, AsyncRequestProperties> async_req_map_;
  std::atomic<uint> total_ongoing_requests_{0};
  size_t data_step_id_;

  // Function pointer to the async callback function implementation
  std::function<void(cb::InferResult*)> async_callback_func_ = std::bind(
      &InferContext::AsyncCallbackFuncImpl, this, std::placeholders::_1);

  // Function pointer to registered async callbacks
  std::function<void(uint32_t)> async_callback_finalize_func_ = nullptr;

 private:
  const uint32_t id_;

  size_t GetNumActiveThreads() { return num_active_threads_; }

  std::default_random_engine rng_generator_;
  size_t num_active_threads_{0};

  // The backend to communicate with the server
  std::unique_ptr<cb::ClientBackend> infer_backend_;
  InferData infer_data_;
};

}}  // namespace triton::perfanalyzer
