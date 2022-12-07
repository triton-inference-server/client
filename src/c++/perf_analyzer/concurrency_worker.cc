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
  uint32_t seq_stat_index = 0, ctx_id = 0;

  // Reserve the vectors in case of sequence models. In non-sequence or
  // synchronous mode only one context will be opened hence no need of
  // reserving.
  if (on_sequence_model_ && async_) {
    thread_stat_->contexts_stat_.reserve(max_concurrency_);
    ctxs_.reserve(max_concurrency_);
  }

  uint64_t request_id = 0;

  // run inferencing until receiving exit signal to maintain server load.
  do {
    if (on_sequence_model_) {
      if (!execute_) {
        // Ensures the clean exit of the sequences
        auto status = complete_ongoing_sequence_func();
        if (thread_stat_->status_.IsOk()) {
          thread_stat_->status_ = status;
        }
        while (total_ongoing_requests_ != 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        // Make sure all threads are in sync with the client's stats
        //
        for (size_t i = 0; i < ctxs_.size(); ++i) {
          ctxs_[i]->infer_backend_->ClientInferStat(
              &(thread_stat_->contexts_stat_[i]));
        }
        // Reconstruct 'free_ctx_ids_' because complete_ongoing_sequence_func()
        // has destructive side affects
        free_ctx_ids_ = std::queue<int>();
        for (size_t i = 0; i < ctxs_.size(); ++i) {
          free_ctx_ids_.push(i);
        }
        // Wait if no request should be sent and it is not exiting
        thread_config_->is_paused_ = true;
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_signal_.wait(lock, [this]() { return early_exit || execute_; });
      }
    }

    thread_config_->is_paused_ = false;

    // Only interact with synchronous mechanism if the worker should wait
    if (thread_config_->concurrency_ == 0) {
      // Wait if no request should be sent and it is not exiting
      std::unique_lock<std::mutex> lock(wake_mutex_);
      // FIXME this was waiting on thread_config before
      wake_signal_.wait(lock, [this]() {
        return early_exit || (thread_config_->concurrency_ > 0);
      });
      // Stop executing if concurrency is 0 and early exit is requested
      if (early_exit && thread_config_->concurrency_ == 0) {
        break;
      }
    }

    size_t num_reqs = thread_config_->concurrency_;

    // If the model is non-sequence model, use one InferContext to
    // maintain concurrency for this thread.
    size_t active_ctx_cnt = on_sequence_model_ ? num_reqs : 1;

    while (active_ctx_cnt > ctxs_.size()) {
      {
        std::lock_guard<std::mutex> lock(cb_mtx_);
        free_ctx_ids_.push(ctxs_.size());
      }
      ctxs_.emplace_back(new InferContext());
      thread_stat_->status_ =
          factory_->CreateClientBackend(&(ctxs_.back()->infer_backend_));
      ctxs_.back()->options_.reset(new cb::InferOptions(parser_->ModelName()));
      ctxs_.back()->options_->model_version_ = parser_->ModelVersion();
      ctxs_.back()->options_->model_signature_name_ =
          parser_->ModelSignatureName();
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

    // Create async requests such that the number of ongoing requests
    // matches the concurrency level
    // Non-sequence model is 'num_reqs' * 1 ctx
    // Sequence model is 1 request of 1 sequence * 'active_ctx_cnt' ctxs_
    while (total_ongoing_requests_ < (int)num_reqs && early_exit == false) {
      // Update the inputs if required for non-sequence
      if (using_json_data_ && (!on_sequence_model_)) {
        int step_id = (thread_config_->non_sequence_data_step_id_ %
                       data_loader_->GetTotalStepsNonSequence()) *
                      batch_size_;
        thread_config_->non_sequence_data_step_id_ += active_threads_;
        // There will be only one ctx in non-sequence case
        thread_stat_->status_ = UpdateInputs(
            ctxs_[ctx_id]->inputs_, ctxs_[ctx_id]->valid_inputs_, 0, step_id);
        if (thread_stat_->status_.IsOk()) {
          thread_stat_->status_ = UpdateValidationOutputs(
              ctxs_[ctx_id]->outputs_, 0, step_id,
              ctxs_[ctx_id]->expected_outputs_);
        }
        if (!thread_stat_->status_.IsOk()) {
          return;
        }
      }

      if (on_sequence_model_) {
        size_t offset = 0;
        for (size_t i = 0; i < thread_config_->thread_id_; i++) {
          offset += threads_config_[i]->concurrency_;
        }

        // Find the next available context id to use for this request
        {
          std::lock_guard<std::mutex> lk(cb_mtx_);
          ctx_id = free_ctx_ids_.front();
          free_ctx_ids_.pop();
        }
        seq_stat_index = offset + ctx_id;

        {
          std::lock_guard<std::mutex> guard(
              sequence_stat_[seq_stat_index]->mtx_);
          SetInferSequenceOptions(seq_stat_index, ctxs_[ctx_id]->options_);

          // Update the inputs if required
          if (using_json_data_) {
            int step_id = data_loader_->GetTotalSteps(
                              sequence_stat_[seq_stat_index]->data_stream_id_) -
                          sequence_stat_[seq_stat_index]->remaining_queries_;

            thread_stat_->status_ = UpdateInputs(
                ctxs_[ctx_id]->inputs_, ctxs_[ctx_id]->valid_inputs_,
                sequence_stat_[seq_stat_index]->data_stream_id_, step_id);
            if (thread_stat_->status_.IsOk()) {
              thread_stat_->status_ = UpdateValidationOutputs(
                  ctxs_[ctx_id]->outputs_,
                  sequence_stat_[seq_stat_index]->data_stream_id_, step_id,
                  ctxs_[ctx_id]->expected_outputs_);
            }
            if (!thread_stat_->status_.IsOk()) {
              return;
            }
          }
          sequence_stat_[seq_stat_index]->remaining_queries_--;
        }
      }
      if (async_) {
        ctxs_[ctx_id]->options_->request_id_ = std::to_string(request_id++);
        {
          std::lock_guard<std::mutex> lock(thread_stat_->mu_);
          auto it = async_req_map_
                        .emplace(
                            ctxs_[ctx_id]->options_->request_id_,
                            AsyncRequestProperties())
                        .first;
          it->second.start_time_ = std::chrono::system_clock::now();
          it->second.ctx_id_ = ctx_id;
          it->second.sequence_end_ = ctxs_[ctx_id]->options_->sequence_end_;
        }
        if (streaming_) {
          thread_stat_->status_ =
              ctxs_[ctx_id]->infer_backend_->AsyncStreamInfer(
                  *(ctxs_[ctx_id]->options_), ctxs_[ctx_id]->valid_inputs_,
                  ctxs_[ctx_id]->outputs_);
        } else {
          thread_stat_->status_ = ctxs_[ctx_id]->infer_backend_->AsyncInfer(
              async_callback_func_, *(ctxs_[ctx_id]->options_),
              ctxs_[ctx_id]->valid_inputs_, ctxs_[ctx_id]->outputs_);
        }
        if (!thread_stat_->status_.IsOk()) {
          return;
        }
      } else {
        std::chrono::time_point<std::chrono::system_clock> start_time_sync,
            end_time_sync;
        start_time_sync = std::chrono::system_clock::now();
        cb::InferResult* results = nullptr;
        thread_stat_->status_ = ctxs_[ctx_id]->infer_backend_->Infer(
            &results, *(ctxs_[ctx_id]->options_), ctxs_[ctx_id]->valid_inputs_,
            ctxs_[ctx_id]->outputs_);
        if (results != nullptr) {
          if (thread_stat_->status_.IsOk()) {
            thread_stat_->status_ = ValidateOutputs(*ctxs_[ctx_id], results);
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
              start_time_sync, end_time_sync,
              ctxs_[ctx_id]->options_->sequence_end_, false /* delayed */));
          thread_stat_->status_ =
              ctxs_[ctx_id]->infer_backend_->ClientInferStat(
                  &(thread_stat_->contexts_stat_[ctx_id]));
          if (!thread_stat_->status_.IsOk()) {
            return;
          }
        }
        {
          std::lock_guard<std::mutex> lock(cb_mtx_);
          free_ctx_ids_.push(ctx_id);
        }
      }
      total_ongoing_requests_++;
    }

    if (async_) {
      {
        // If async, then wait for signal from callback.
        std::unique_lock<std::mutex> lk(cb_mtx_);
        // FIXME was waiting on notified before?
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

    if (early_exit || (!thread_stat_->cb_status_.IsOk())) {
      // Wait for all callbacks to complete.
      // Loop to ensure all the inflight requests have been completed.
      auto status = complete_ongoing_sequence_func();
      if (thread_stat_->status_.IsOk()) {
        thread_stat_->status_ = status;
      }
      while (total_ongoing_requests_ != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
      // end loop
      break;
    }
  } while (true);
}

void
ConcurrencyWorker::async_callback_func_impl(cb::InferResult* result)
{
  uint32_t ctx_id = 0;
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
            false /* delayed */));
        ctx_id = it->second.ctx_id_;
        ctxs_[ctx_id]->infer_backend_->ClientInferStat(
            &(thread_stat_->contexts_stat_[ctx_id]));
        thread_stat_->cb_status_ = ValidateOutputs(*ctxs_[ctx_id], result);
        async_req_map_.erase(request_id);
      }
    }
  }
  // avoid competition over 'cb_mtx_'
  {
    std::lock_guard<std::mutex> lk(cb_mtx_);
    free_ctx_ids_.push(ctx_id);
    notified_ = true;
  }

  total_ongoing_requests_--;

  cb_cv_.notify_all();
}

// Specify the function as lambda here to work around the possible callback
// lifecycle issue when making this a class member function.
// Note that 'free_ctx_ids_' must be reconstruct after the call because
// this function doesn't utilize 'free_ctx_ids_' in the same way as in main
// loop
cb::Error
ConcurrencyWorker::complete_ongoing_sequence_func()
{
  if (!on_sequence_model_) {
    return cb::Error::Success;
  }
  size_t offset = 0;
  for (size_t i = 0; i < thread_config_->thread_id_; i++) {
    offset += threads_config_[i]->concurrency_;
  }

  for (size_t ctx_id = 0; ctx_id < ctxs_.size(); ++ctx_id) {
    size_t seq_stat_index = offset + ctx_id;

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

}}  // namespace triton::perfanalyzer
