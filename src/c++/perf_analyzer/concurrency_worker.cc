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

#include <algorithm>
#include <thread>

#include "client_backend/client_backend.h"
#include "concurrency_worker.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

// Function for worker threads.
// If the model is non-sequence model, each worker uses only one context
// to maintain concurrency assigned to worker.
// If the model is sequence model, each worker has to use multiples contexts
// to maintain (sequence) concurrency assigned to worker.
void
ConcurrencyWorker::Infer()
{
  ReserveContexts();

  // run inferencing until receiving exit signal to maintain server load.
  do {
    HandleExecuteOff();

    if (HandleNoConcurrency()) {
      return;
    }

    CreateContextsAsNecessary();

    if (!thread_stat_->status_.IsOk()) {
      return;
    }

    PrepAndSendInferRequests();

    if (!thread_stat_->status_.IsOk()) {
      return;
    }

    WaitForResponses();

    if (HandleExitConditions()) {
      return;
    }

  } while (true);
}

void
ConcurrencyWorker::ReserveContexts()
{
  // Reserve the vectors in case of sequence models. In non-sequence or
  // synchronous mode only one context will be opened hence no need of
  // reserving.
  if (on_sequence_model_ && async_) {
    thread_stat_->contexts_stat_.reserve(max_concurrency_);
    ctxs_.reserve(max_concurrency_);
  }
}

void
ConcurrencyWorker::HandleExecuteOff()
{
  if (on_sequence_model_) {
    if (!execute_) {
      // Ensures the clean exit of the sequences
      auto status = CompleteOngoingSequences();
      if (thread_stat_->status_.IsOk()) {
        thread_stat_->status_ = status;
      }
      WaitForOngoingRequests();
      SyncClientStats();

      // Reconstruct 'free_ctx_ids_' because CompleteOngoingSequences()
      // has destructive side affects
      ResetFreeCtxIds();

      // Wait if no request should be sent and it is not exiting
      thread_config_->is_paused_ = true;
      std::unique_lock<std::mutex> lock(wake_mutex_);
      wake_signal_.wait(lock, [this]() { return early_exit || execute_; });
    }
  }
  thread_config_->is_paused_ = false;
}

bool
ConcurrencyWorker::HandleNoConcurrency()
{
  // Only interact with synchronous mechanism if the worker should wait
  if (thread_config_->concurrency_ == 0) {
    // Wait if no request should be sent and it is not exiting
    std::unique_lock<std::mutex> lock(wake_mutex_);
    wake_signal_.wait(lock, [this]() {
      return early_exit || (thread_config_->concurrency_ > 0);
    });
    // Stop executing if concurrency is 0 and early exit is requested
    if (early_exit && thread_config_->concurrency_ == 0) {
      return true;
    }
  }
  return false;
}

void
ConcurrencyWorker::CreateContextsAsNecessary()
{
  // If the model is non-sequence model, use one InferContext to
  // maintain concurrency for this thread.
  size_t active_ctx_cnt = on_sequence_model_ ? thread_config_->concurrency_ : 1;

  while (active_ctx_cnt > ctxs_.size()) {
    CreateContext();
  }
  ResetFreeCtxIds();
}

void
ConcurrencyWorker::PrepAndSendInferRequests()
{
  // Create async requests such that the number of ongoing requests
  // matches the concurrency level
  // Non-sequence model is 'num_reqs' * 1 ctx
  // Sequence model is 1 request of 1 sequence * 'active_ctx_cnt' ctxs_
  while (total_ongoing_requests_ < (int)thread_config_->concurrency_ &&
         early_exit == false) {
    PrepAndSendInferRequest();
  }
}

void
ConcurrencyWorker::PrepAndSendInferRequest()
{
  uint32_t ctx_id = GetCtxId();

  // Update the inputs if required for non-sequence
  if (using_json_data_ && (!on_sequence_model_)) {
    UpdateJsonData(
        std::static_pointer_cast<DataStepIdTracker>(thread_config_), ctx_id,
        active_threads_);
  }

  if (on_sequence_model_) {
    uint32_t seq_stat_index = GetSeqStatIndex(ctx_id);

    {
      std::lock_guard<std::mutex> guard(sequence_stat_[seq_stat_index]->mtx_);
      SetInferSequenceOptions(seq_stat_index, ctxs_[ctx_id]->options_);

      // Update the inputs if required
      if (using_json_data_) {
        UpdateSeqJsonData(ctx_id, sequence_stat_[seq_stat_index]);
      }
      sequence_stat_[seq_stat_index]->remaining_queries_--;
    }
  }
  bool is_delayed = false;
  SendRequest(
      ctxs_[ctx_id], ctx_id, request_id_++, is_delayed, async_callback_func_,
      async_req_map_, thread_stat_);

  // FIXME TMA-1023 we are clearly pushing and not popping in some cases
  if (!async_) {
    {
      std::lock_guard<std::mutex> lock(cb_mtx_);
      free_ctx_ids_.push(ctx_id);
    }
  }
  total_ongoing_requests_++;
}

void
ConcurrencyWorker::WaitForResponses()
{
  if (async_) {
    {
      // If async, then wait for signal from callback.
      std::unique_lock<std::mutex> lk(cb_mtx_);
      cb_cv_.wait(lk, [this] {
        if (notified_) {
          notified_ = false;
          return true;
        }
        return false;
      });
    }
  } else {
    // If synchronous, then all the requests have already been completed.
    total_ongoing_requests_ = 0;
  }
}

bool
ConcurrencyWorker::HandleExitConditions()
{
  if (early_exit || (!thread_stat_->cb_status_.IsOk())) {
    // Wait for all callbacks to complete.
    // Loop to ensure all the inflight requests have been completed.
    auto status = CompleteOngoingSequences();
    if (thread_stat_->status_.IsOk()) {
      thread_stat_->status_ = status;
    }
    WaitForOngoingRequests();
    return true;
  }
  return false;
}


void
ConcurrencyWorker::AsyncCallbackFinalize(uint32_t ctx_id)
{
  // avoid competition over 'cb_mtx_'
  {
    std::lock_guard<std::mutex> lk(cb_mtx_);
    free_ctx_ids_.push(ctx_id);
    notified_ = true;
  }

  total_ongoing_requests_--;

  cb_cv_.notify_all();
}

// Note that 'free_ctx_ids_' must be reconstruct after the call because
// this function doesn't utilize 'free_ctx_ids_' in the same way as in main
// loop
cb::Error
ConcurrencyWorker::CompleteOngoingSequences()
{
  if (!on_sequence_model_) {
    return cb::Error::Success;
  }

  for (size_t ctx_id = 0; ctx_id < ctxs_.size(); ++ctx_id) {
    size_t seq_stat_index = GetSeqStatIndex(ctx_id);

    std::lock_guard<std::mutex> guard(sequence_stat_[seq_stat_index]->mtx_);
    // Complete the sequence if there are remaining queries
    while (sequence_stat_[seq_stat_index]->remaining_queries_ != 0) {
      sequence_stat_[seq_stat_index]->remaining_queries_ = 1;
      SetInferSequenceOptions(seq_stat_index, ctxs_[ctx_id]->options_);

      // Update the inputs if required
      if (using_json_data_) {
        int step_id = data_loader_->GetTotalSteps(
                          sequence_stat_[seq_stat_index]->data_stream_id_) -
                      sequence_stat_[seq_stat_index]->remaining_queries_;

        RETURN_IF_ERROR(UpdateInputs(
            ctxs_[ctx_id]->inputs_, ctxs_[ctx_id]->valid_inputs_,
            sequence_stat_[seq_stat_index]->data_stream_id_, step_id));
        RETURN_IF_ERROR(UpdateValidationOutputs(
            ctxs_[ctx_id]->outputs_,
            sequence_stat_[seq_stat_index]->data_stream_id_, step_id,
            ctxs_[ctx_id]->expected_outputs_));
      }
      sequence_stat_[seq_stat_index]->remaining_queries_--;

      if (async_) {
        ctxs_[ctx_id]->options_->request_id_ = "0";
        if (streaming_) {
          RETURN_IF_ERROR(ctxs_[ctx_id]->infer_backend_->AsyncStreamInfer(
              *(ctxs_[ctx_id]->options_), ctxs_[ctx_id]->inputs_,
              ctxs_[ctx_id]->outputs_));
        } else {
          RETURN_IF_ERROR(ctxs_[ctx_id]->infer_backend_->AsyncInfer(
              async_callback_func_, *(ctxs_[ctx_id]->options_),
              ctxs_[ctx_id]->inputs_, ctxs_[ctx_id]->outputs_));
        }
        total_ongoing_requests_++;
      } else {
        cb::InferResult* results = nullptr;
        auto err = ctxs_[ctx_id]->infer_backend_->Infer(
            &results, *(ctxs_[ctx_id]->options_), ctxs_[ctx_id]->inputs_,
            ctxs_[ctx_id]->outputs_);
        if (results != nullptr) {
          delete results;
        }
        RETURN_IF_ERROR(err);
      }
    }
  }
  return cb::Error::Success;
}

void
ConcurrencyWorker::WaitForOngoingRequests()
{
  {
    while (total_ongoing_requests_ != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
}
}}  // namespace triton::perfanalyzer
