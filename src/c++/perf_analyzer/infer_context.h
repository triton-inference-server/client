// Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "client_backend/client_backend.h"
#include "data_loader.h"
#include "idle_timer.h"
#include "iinfer_data_manager.h"
#include "infer_data.h"
#include "perf_utils.h"
#include "request_record.h"
#include "sequence_manager.h"

namespace triton { namespace perfanalyzer {

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

  // Tracks the amount of time this thread spent sleeping or waiting
  IdleTimer idle_timer;

  // A vector of request records
  std::vector<RequestRecord> request_records_;
  // A lock to protect thread data
  std::mutex mu_;
  // The number of sent requests by this thread.
  std::atomic<size_t> num_sent_requests_{0};
};

#ifndef DOCTEST_CONFIG_DISABLE
class NaggyMockInferContext;
#endif

/// Sends inference requests to the server
class InferContext {
 public:
  InferContext(
      const size_t thread_id, const uint32_t id, const bool async,
      const bool streaming, const bool on_sequence_model,
      const bool using_json_data, const int32_t batch_size,
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<DataLoader> data_loader,
      std::shared_ptr<ModelParser> parser,
      std::shared_ptr<cb::ClientBackendFactory> factory, const bool& execute,
      const std::shared_ptr<IInferDataManager>& infer_data_manager,
      std::shared_ptr<SequenceManager> sequence_manager)
      : thread_id_(thread_id), id_(id), async_(async), streaming_(streaming),
        on_sequence_model_(on_sequence_model),
        using_json_data_(using_json_data), batch_size_(batch_size),
        thread_stat_(thread_stat), data_loader_(data_loader), parser_(parser),
        factory_(factory), data_step_id_(thread_id), execute_(execute),
        infer_data_manager_(infer_data_manager),
        sequence_manager_(sequence_manager)
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

  // Returns the number of responses for the current request
  uint64_t GetNumResponsesForCurrentRequest() { return num_responses_; }

  // Register a function that will get called after every async request returns
  void RegisterAsyncCallbackFinalize(std::function<void(uint32_t)> callback)
  {
    async_callback_finalize_func_ = callback;
  }

  void RegisterWorkerCallback(std::function<void(uint32_t)> worker_callback)
  {
    worker_callback_ = worker_callback;
  }

  // TODO REFACTOR TMA-1043 this should be in memory class
  void SetNumActiveThreads(size_t num_threads)
  {
    num_active_threads_ = num_threads;
  }

  bool HasReceivedFinalResponse() { return has_received_final_response_; }

 protected:
  /// A helper function to issue inference request to the server.
  /// \param request_id The unique id to be associated with the request.
  /// \param delayed Whether the request fell behind its scheduled time.
  /// \param sequence_id Sequence ID of the request. Note that the default of
  /// `0` means the request is not a sequence.
  virtual void SendRequest(
      const uint64_t request_id, const bool delayed,
      const uint64_t sequence_id = 0);

  /// Update inputs based on custom json data
  void UpdateJsonData();

  /// Update inputs based on custom json data for the given sequence
  void UpdateSeqJsonData(size_t seq_stat_index);

  cb::Error ValidateOutputs(const cb::InferResult* result_ptr);

  // Callback function for handling asynchronous requests
  void AsyncCallbackFuncImpl(cb::InferResult* result);

  bool async_{false};
  bool streaming_{false};
  const bool on_sequence_model_{false};
  bool using_json_data_{false};
  const int32_t batch_size_{0};

  std::shared_ptr<ThreadStat> thread_stat_;
  std::shared_ptr<DataLoader> data_loader_;
  std::shared_ptr<ModelParser> parser_;
  std::shared_ptr<cb::ClientBackendFactory> factory_;
  std::shared_ptr<IInferDataManager> infer_data_manager_;

  uint64_t request_id_ = 0;
  std::map<std::string, RequestRecord> async_req_map_;
  std::atomic<uint> total_ongoing_requests_{0};
  size_t data_step_id_;

  // Function pointer to the async callback function implementation
  std::function<void(cb::InferResult*)> async_callback_func_ = std::bind(
      &InferContext::AsyncCallbackFuncImpl, this, std::placeholders::_1);

  // Function pointer to registered async callbacks
  std::function<void(uint32_t)> async_callback_finalize_func_ = nullptr;

 private:
  const RequestRecord::RequestInput GetInputs();

  const RequestRecord::ResponseOutput GetOutputs(
      const cb::InferResult& infer_result);

  const uint32_t id_{0};
  const size_t thread_id_{0};

  size_t GetNumActiveThreads() { return num_active_threads_; }

  size_t num_active_threads_{0};

  // The backend to communicate with the server
  std::unique_ptr<cb::ClientBackend> infer_backend_;
  InferData infer_data_;

  // FIXME: update build to use C++17 instead of C++14. This is a workaround
  // since C++14 doesn't have std::optional, but C++17 does.
  const bool execute_placeholder_{false};
  std::reference_wrapper<const bool> execute_{execute_placeholder_};

  std::shared_ptr<SequenceManager> sequence_manager_{nullptr};
  uint64_t num_responses_{0};
  std::function<void(uint32_t)> worker_callback_{nullptr};
  bool has_received_final_response_{false};

#ifndef DOCTEST_CONFIG_DISABLE
  friend NaggyMockInferContext;

 public:
  InferContext() = default;
#endif
};

}}  // namespace triton::perfanalyzer
