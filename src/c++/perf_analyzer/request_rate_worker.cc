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
#include "data_loader.h"
#include "perf_utils.h"
#include "request_rate_worker.h"

namespace triton { namespace perfanalyzer {

void
RequestRateWorker::Infer()
{
  CreateContext();
  if (!thread_stat_->status_.IsOk()) {
    return;
  }

  // run inferencing until receiving exit signal to maintain server load.
  do {
    HandleExecuteOff();

    bool is_delayed = SleepIfNecessary();

    PrepAndSendInferRequest(is_delayed);

    if (HandleExitConditions()) {
      return;
    }

  } while (true);
}

void
RequestRateWorker::CreateContext()
{
  ctxs_.push_back(std::make_shared<InferContext>());

  thread_stat_->status_ =
      factory_->CreateClientBackend(&(ctxs_.back()->infer_backend_));
  ctxs_.back()->options_.reset(new cb::InferOptions(parser_->ModelName()));
  ctxs_.back()->options_->model_version_ = parser_->ModelVersion();
  ctxs_.back()->options_->model_signature_name_ = parser_->ModelSignatureName();

  thread_stat_->contexts_stat_.emplace_back();

  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    thread_stat_->status_ = PrepareInfer(ctxs_.back().get());
  } else {
    thread_stat_->status_ = PrepareSharedMemoryInfer(ctxs_.back().get());
  }
  if (!thread_stat_->status_.IsOk()) {
    return;
  }

  if (streaming_) {
    // Decoupled models should not collect client side statistics
    thread_stat_->status_ = ctxs_.back()->infer_backend_->StartStream(
        async_callback_func_, (!parser_->IsDecoupled()));
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
  }
}

void
RequestRateWorker::HandleExecuteOff()
{
  // Should wait till main thread signals execution start
  if (!execute_) {
    if (on_sequence_model_) {
      // Finish off all the ongoing sequences for graceful exit
      for (size_t i = thread_config_->id_; i < sequence_stat_.size();
           i += thread_config_->stride_) {
        std::lock_guard<std::mutex> guard(sequence_stat_[i]->mtx_);
        sequence_stat_[i]->paused_ = true;
        if (sequence_stat_[i]->remaining_queries_ != 0) {
          ctxs_[ctx_id_]->options_->sequence_start_ = false;
          ctxs_[ctx_id_]->options_->sequence_end_ = true;
          ctxs_[ctx_id_]->options_->sequence_id_ = sequence_stat_[i]->seq_id_;

          bool is_delayed = false;
          uint32_t ctx_id = 0;
          SendRequest(
              ctxs_[ctx_id], ctx_id, request_id_++, is_delayed,
              async_callback_func_, async_req_map_, thread_stat_);
          sequence_stat_[i]->remaining_queries_ = 0;
        }
      }
    }
    // Ensures the clean measurements after thread is woken up.
    while (ctxs_[ctx_id_]->inflight_request_cnt_ != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    // Wait if no request should be sent and it is not exiting
    thread_config_->is_paused_ = true;
    std::unique_lock<std::mutex> lock(wake_mutex_);
    wake_signal_.wait(lock, [this]() { return early_exit || execute_; });

    if (on_sequence_model_) {
      for (size_t i = thread_config_->id_; i < sequence_stat_.size();
           i += thread_config_->stride_) {
        std::lock_guard<std::mutex> guard(sequence_stat_[i]->mtx_);
        sequence_stat_[i]->paused_ = false;
      }
    }
  }

  thread_config_->is_paused_ = false;
}

bool
RequestRateWorker::SleepIfNecessary()
{
  // Sleep if required
  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

  std::chrono::nanoseconds wait_time =
      (schedule_[thread_config_->index_] +
       (thread_config_->rounds_ * (*gen_duration_))) -
      (now - start_time_);

  thread_config_->index_ = (thread_config_->index_ + thread_config_->stride_);
  // Loop around the schedule to keep running
  thread_config_->rounds_ += (thread_config_->index_ / schedule_.size());
  thread_config_->index_ = thread_config_->index_ % schedule_.size();

  bool delayed = false;
  if (wait_time.count() < 0) {
    delayed = true;
  } else {
    std::this_thread::sleep_for(wait_time);
  }
  return delayed;
}

void
RequestRateWorker::PrepAndSendInferRequest(bool delayed)
{
  if (!on_sequence_model_) {
    // Update the inputs if required
    if (using_json_data_) {
      UpdateJsonData(
          std::static_pointer_cast<DataStepIdTracker>(thread_config_), ctx_id_,
          max_threads_);
    }
    SendRequest(
        ctxs_[ctx_id_], ctx_id_, request_id_++, delayed, async_callback_func_,
        async_req_map_, thread_stat_);
  } else {
    // Select one of the sequence at random for this request
    uint32_t seq_stat_index = rand() % sequence_stat_.size();
    // Need lock to protect the order of dispatch across worker threads.
    // This also helps in reporting the realistic latencies.
    std::lock_guard<std::mutex> guard(sequence_stat_[seq_stat_index]->mtx_);
    if (!early_exit && !sequence_stat_[seq_stat_index]->paused_) {
      SetInferSequenceOptions(seq_stat_index, ctxs_[ctx_id_]->options_);

      // Update the inputs if required
      if (using_json_data_) {
        UpdateSeqJsonData(ctx_id_, sequence_stat_[seq_stat_index]);
      }

      sequence_stat_[seq_stat_index]->remaining_queries_--;

      SendRequest(
          ctxs_[ctx_id_], ctx_id_, request_id_++, delayed, async_callback_func_,
          async_req_map_, thread_stat_);
    }
  }
}

bool
RequestRateWorker::HandleExitConditions()
{
  if (early_exit || (!thread_stat_->cb_status_.IsOk())) {
    if (on_sequence_model_) {
      // Finish off all the ongoing sequences for graceful exit
      for (size_t i = thread_config_->id_; i < sequence_stat_.size();
           i += thread_config_->stride_) {
        std::lock_guard<std::mutex> guard(sequence_stat_[i]->mtx_);
        sequence_stat_[i]->paused_ = true;
        if (sequence_stat_[i]->remaining_queries_ != 0) {
          ctxs_[ctx_id_]->options_->sequence_start_ = false;
          ctxs_[ctx_id_]->options_->sequence_end_ = true;
          ctxs_[ctx_id_]->options_->sequence_id_ = sequence_stat_[i]->seq_id_;

          bool is_delayed = false;
          SendRequest(
              ctxs_[ctx_id_], ctx_id_, request_id_++, is_delayed,
              async_callback_func_, async_req_map_, thread_stat_);
          sequence_stat_[i]->remaining_queries_ = 0;
        }
      }
    }
    if (async_) {
      // Loop to ensure all the inflight requests have been completed.
      while (ctxs_[ctx_id_]->inflight_request_cnt_ != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }
    return true;
  }
  return false;
}

}}  // namespace triton::perfanalyzer
