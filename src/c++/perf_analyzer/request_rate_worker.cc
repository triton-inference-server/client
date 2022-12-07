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
  create_context();
  if (!thread_stat_->status_.IsOk()) {
    return;
  }

  // run inferencing until receiving exit signal to maintain server load.
  do {
    handle_execute_off();

    bool is_delayed = sleep_if_necessary();

    send_infer_request(is_delayed);

    if (handle_exit_conditions()) {
      return;
    }

  } while (true);
}

void
RequestRateWorker::create_context()
{
  thread_stat_->status_ = factory_->CreateClientBackend(&(ctx_.infer_backend_));
  ctx_.options_.reset(new cb::InferOptions(parser_->ModelName()));
  ctx_.options_->model_version_ = parser_->ModelVersion();
  ctx_.options_->model_signature_name_ = parser_->ModelSignatureName();

  thread_stat_->contexts_stat_.emplace_back();

  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    thread_stat_->status_ = PrepareInfer(&ctx_);
  } else {
    thread_stat_->status_ = PrepareSharedMemoryInfer(&ctx_);
  }
  if (!thread_stat_->status_.IsOk()) {
    return;
  }

  if (streaming_) {
    // Decoupled models should not collect client side statistics
    thread_stat_->status_ = ctx_.infer_backend_->StartStream(
        async_callback_func_, (!parser_->IsDecoupled()));
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
  }
}

void
RequestRateWorker::handle_execute_off()
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
          ctx_.options_->sequence_start_ = false;
          ctx_.options_->sequence_end_ = true;
          ctx_.options_->sequence_id_ = sequence_stat_[i]->seq_id_;
          Request(
              ctx_, request_id_++, false /* delayed */, async_callback_func_,
              async_req_map_, thread_stat_);
          sequence_stat_[i]->remaining_queries_ = 0;
        }
      }
    }
    // Ensures the clean measurements after thread is woken up.
    while (ctx_.inflight_request_cnt_ != 0) {
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
RequestRateWorker::sleep_if_necessary()
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
RequestRateWorker::send_infer_request(bool delayed)
{
  // Update the inputs if required
  if (using_json_data_ && (!on_sequence_model_)) {
    int step_id = (thread_config_->non_sequence_data_step_id_ %
                   data_loader_->GetTotalStepsNonSequence()) *
                  batch_size_;
    thread_config_->non_sequence_data_step_id_ += max_threads_;
    thread_stat_->status_ =
        UpdateInputs(ctx_.inputs_, ctx_.valid_inputs_, 0, step_id);
    if (thread_stat_->status_.IsOk()) {
      thread_stat_->status_ = UpdateValidationOutputs(
          ctx_.outputs_, 0, step_id, ctx_.expected_outputs_);
    }
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
  }

  if (on_sequence_model_) {
    // Select one of the sequence at random for this request
    uint32_t seq_stat_index = rand() % sequence_stat_.size();
    // Need lock to protect the order of dispatch across worker threads.
    // This also helps in reporting the realistic latencies.
    std::lock_guard<std::mutex> guard(sequence_stat_[seq_stat_index]->mtx_);
    if (!early_exit && !sequence_stat_[seq_stat_index]->paused_) {
      SetInferSequenceOptions(seq_stat_index, ctx_.options_);

      // Update the inputs if required
      if (using_json_data_) {
        int step_id = data_loader_->GetTotalSteps(
                          sequence_stat_[seq_stat_index]->data_stream_id_) -
                      sequence_stat_[seq_stat_index]->remaining_queries_;
        thread_stat_->status_ = UpdateInputs(
            ctx_.inputs_, ctx_.valid_inputs_,
            sequence_stat_[seq_stat_index]->data_stream_id_, step_id);
        if (thread_stat_->status_.IsOk()) {
          thread_stat_->status_ = UpdateValidationOutputs(
              ctx_.outputs_, sequence_stat_[seq_stat_index]->data_stream_id_,
              step_id, ctx_.expected_outputs_);
        }
        if (!thread_stat_->status_.IsOk()) {
          return;
        }
      }

      Request(
          ctx_, request_id_++, delayed, async_callback_func_, async_req_map_,
          thread_stat_);
      sequence_stat_[seq_stat_index]->remaining_queries_--;
    }
  } else {
    Request(
        ctx_, request_id_++, delayed, async_callback_func_, async_req_map_,
        thread_stat_);
  }
}

bool
RequestRateWorker::handle_exit_conditions()
{
  if (early_exit || (!thread_stat_->cb_status_.IsOk())) {
    if (on_sequence_model_) {
      // Finish off all the ongoing sequences for graceful exit
      for (size_t i = thread_config_->id_; i < sequence_stat_.size();
           i += thread_config_->stride_) {
        std::lock_guard<std::mutex> guard(sequence_stat_[i]->mtx_);
        sequence_stat_[i]->paused_ = true;
        if (sequence_stat_[i]->remaining_queries_ != 0) {
          ctx_.options_->sequence_start_ = false;
          ctx_.options_->sequence_end_ = true;
          ctx_.options_->sequence_id_ = sequence_stat_[i]->seq_id_;
          Request(
              ctx_, request_id_++, false /* delayed */, async_callback_func_,
              async_req_map_, thread_stat_);
          sequence_stat_[i]->remaining_queries_ = 0;
        }
      }
    }
    if (async_) {
      // Loop to ensure all the inflight requests have been completed.
      while (ctx_.inflight_request_cnt_ != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }
    return true;
  }
  return false;
}

void
RequestRateWorker::Request(
    InferContext& context, const uint64_t request_id, const bool delayed,
    cb::OnCompleteFn callback_func,
    std::map<std::string, AsyncRequestProperties>& async_req_map_,
    std::shared_ptr<ThreadStat> thread_stat_)
{
  if (async_) {
    context.options_->request_id_ = std::to_string(request_id);
    {
      std::lock_guard<std::mutex> lock(thread_stat_->mu_);
      auto it =
          async_req_map_
              .emplace(context.options_->request_id_, AsyncRequestProperties())
              .first;
      it->second.start_time_ = std::chrono::system_clock::now();
      it->second.sequence_end_ = context.options_->sequence_end_;
      it->second.delayed_ = delayed;
    }
    if (streaming_) {
      thread_stat_->status_ = context.infer_backend_->AsyncStreamInfer(
          *(context.options_), context.valid_inputs_, context.outputs_);
    } else {
      thread_stat_->status_ = context.infer_backend_->AsyncInfer(
          callback_func, *(context.options_), context.valid_inputs_,
          context.outputs_);
    }
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
    context.inflight_request_cnt_++;
  } else {
    std::chrono::time_point<std::chrono::system_clock> start_time_sync,
        end_time_sync;
    start_time_sync = std::chrono::system_clock::now();
    cb::InferResult* results = nullptr;
    thread_stat_->status_ = context.infer_backend_->Infer(
        &results, *(context.options_), context.valid_inputs_, context.outputs_);
    if (results != nullptr) {
      if (thread_stat_->status_.IsOk()) {
        thread_stat_->status_ = ValidateOutputs(context, results);
      }
      delete results;
    }
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
    end_time_sync = std::chrono::system_clock::now();
    {
      // Add the request timestamp to thread Timestamp vector with proper
      // locking
      std::lock_guard<std::mutex> lock(thread_stat_->mu_);
      thread_stat_->request_timestamps_.emplace_back(std::make_tuple(
          start_time_sync, end_time_sync, context.options_->sequence_end_,
          delayed));
      thread_stat_->status_ = context.infer_backend_->ClientInferStat(
          &(thread_stat_->contexts_stat_[0]));
      if (!thread_stat_->status_.IsOk()) {
        return;
      }
    }
  }
}

void
RequestRateWorker::async_callback_func_impl(cb::InferResult* result)
{
  std::shared_ptr<cb::InferResult> result_ptr(result);
  if (thread_stat_->cb_status_.IsOk()) {
    // Add the request timestamp to thread Timestamp vector with
    // proper locking
    std::lock_guard<std::mutex> lock(thread_stat_->mu_);
    thread_stat_->cb_status_ = result_ptr->RequestStatus();
    if (thread_stat_->cb_status_.IsOk()) {
      std::chrono::time_point<std::chrono::system_clock> end_time_async;
      end_time_async = std::chrono::system_clock::now();
      std::string request_id;
      thread_stat_->cb_status_ = result_ptr->Id(&request_id);
      const auto& it = async_req_map_.find(request_id);
      if (it != async_req_map_.end()) {
        thread_stat_->request_timestamps_.emplace_back(std::make_tuple(
            it->second.start_time_, end_time_async, it->second.sequence_end_,
            it->second.delayed_));
        ctx_.infer_backend_->ClientInferStat(
            &(thread_stat_->contexts_stat_[0]));
        thread_stat_->cb_status_ = ValidateOutputs(ctx_, result);
        async_req_map_.erase(request_id);
      } else {
        return;
      }
    }
  }
  ctx_.inflight_request_cnt_--;
}

}}  // namespace triton::perfanalyzer
