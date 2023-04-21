// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    if (HandleExitConditions()) {
      return;
    }

    SendInferRequests();

    if (HandleExitConditions()) {
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
      CompleteOngoingSequences();
      WaitForOngoingRequests();

      // Reconstruct 'free_ctx_ids_' because CompleteOngoingSequences()
      // has destructive side affects
      ResetFreeCtxIds();

      // Wait if no request should be sent and it is not exiting
      thread_config_->is_paused_ = true;
      std::unique_lock<std::mutex> lock(wake_mutex_);
      wake_signal_.wait(lock, [this]() { return early_exit || execute_; });

      // TODO REFACTOR TMA-1043 - memory manager should be handling this instead
      // of here
      for (auto ctx : ctxs_) {
        ctx->SetNumActiveThreads(active_threads_);
      }
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

  if (active_ctx_cnt > ctxs_.size()) {
    while (active_ctx_cnt > ctxs_.size()) {
      CreateContext();
    }
    ResetFreeCtxIds();
  }

  // TODO REFACTOR TMA-1043 -- this shouldn't be handled here
  for (auto ctx : ctxs_) {
    ctx->SetNumActiveThreads(active_threads_);
  }
}

void
ConcurrencyWorker::SendInferRequests()
{
  while (free_ctx_ids_.size() && execute_ && !ShouldExit()) {
    uint32_t ctx_id = GetCtxId();
    SendInferRequest(ctx_id);
    RestoreFreeCtxId(ctx_id);
  }
}

void
ConcurrencyWorker::RestoreFreeCtxId(uint32_t ctx_id)
{
  if (!async_) {
    {
      std::lock_guard<std::mutex> lock(cb_mtx_);
      free_ctx_ids_.push(ctx_id);
    }
  }
}

void
ConcurrencyWorker::WaitForResponses()
{
  if (async_) {
    {
      // If async, then wait for signal from callback.
      std::unique_lock<std::mutex> lk(cb_mtx_);
      thread_stat_->idle_timer.Start();
      cb_cv_.wait(lk, [this] {
        if (notified_) {
          notified_ = false;
          return true;
        }
        return false;
      });
      thread_stat_->idle_timer.Stop();
    }
  }
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

  cb_cv_.notify_all();
}

void
ConcurrencyWorker::CompleteOngoingSequences()
{
  if (on_sequence_model_) {
    for (size_t ctx_id = 0; ctx_id < ctxs_.size(); ++ctx_id) {
      size_t seq_stat_index = GetSeqStatIndex(ctx_id);
      ctxs_[ctx_id]->CompleteOngoingSequence(seq_stat_index);
    }
  }
}

void
ConcurrencyWorker::ResetFreeCtxIds()
{
  std::lock_guard<std::mutex> lock(cb_mtx_);
  free_ctx_ids_ = std::queue<int>();

  for (size_t i = 0; i < thread_config_->concurrency_; ++i) {
    if (on_sequence_model_) {
      free_ctx_ids_.push(i);
    } else {
      free_ctx_ids_.push(0);
    }
  }
}

uint32_t
ConcurrencyWorker::GetSeqStatIndex(uint32_t ctx_id)
{
  return (thread_config_->seq_stat_index_offset_ + ctx_id);
}

uint32_t
ConcurrencyWorker::GetCtxId()
{
  uint32_t ctx_id;
  // Find the next available context id to use for this request
  std::lock_guard<std::mutex> lk(cb_mtx_);
  {
    if (free_ctx_ids_.size() < 1) {
      throw std::runtime_error("free ctx id list is empty");
    }
    ctx_id = free_ctx_ids_.front();
    free_ctx_ids_.pop();
  }
  return ctx_id;
}

}}  // namespace triton::perfanalyzer
