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

#include "request_rate_worker.h"

#include <algorithm>
#include <thread>

#include "client_backend/client_backend.h"
#include "data_loader.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

void
RequestRateWorker::Infer()
{
  CreateCtxIdTracker();
  CreateContexts();

  // run inferencing until receiving exit signal to maintain server load.
  do {
    HandleExecuteOff();

    bool is_delayed = SleepIfNecessary();
    uint32_t ctx_id = GetCtxId();
    SendInferRequest(ctx_id, is_delayed);
    RestoreFreeCtxId(ctx_id);

    if (HandleExitConditions()) {
      return;
    }

  } while (true);
}

void
RequestRateWorker::CreateCtxIdTracker()
{
  bool is_concurrency = false;

  ctx_id_tracker_ = CtxIdTrackerFactory::CreateTracker(
      is_concurrency, on_sequence_model_, serial_sequences_);
}

void
RequestRateWorker::CreateContexts()
{
  size_t active_ctx_cnt =
      on_sequence_model_ ? thread_config_->num_sequences_ : 1;
  while (ctxs_.size() < active_ctx_cnt) {
    CreateContext();
  }

  ResetFreeCtxIds();
}

void
RequestRateWorker::ResetFreeCtxIds()
{
  std::lock_guard<std::mutex> lock(cb_mtx_);
  ctx_id_tracker_->Reset(ctxs_.size());
}

void
RequestRateWorker::SetSchedule(RateSchedulePtr_t schedule)
{
  schedule_ = schedule;
}

std::chrono::nanoseconds
RequestRateWorker::GetNextTimestamp()
{
  return schedule_->Next();
}


uint32_t
RequestRateWorker::GetSeqStatIndex(uint32_t ctx_id)
{
  return (thread_config_->seq_stat_index_offset_ + ctx_id);
}

void
RequestRateWorker::HandleExecuteOff()
{
  // Should wait till main thread signals execution start
  if (!execute_) {
    CompleteOngoingSequences();
    WaitForOngoingRequests();

    // Reset Ctx IDs because CompleteOngoingSequences()
    // has destructive side affects
    ResetFreeCtxIds();

    // Wait if no request should be sent and it is not exiting
    thread_config_->is_paused_ = true;
    std::unique_lock<std::mutex> lock(wake_mutex_);
    wake_signal_.wait(lock, [this]() { return exiting_ || execute_; });
  }

  thread_config_->is_paused_ = false;
}

bool
RequestRateWorker::SleepIfNecessary()
{
  WaitForFreeCtx();

  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  std::chrono::nanoseconds next_timestamp = GetNextTimestamp();
  std::chrono::nanoseconds current_timestamp = now - start_time_;
  std::chrono::nanoseconds wait_time = next_timestamp - current_timestamp;

  bool delayed = false;
  if (wait_time.count() < 0) {
    delayed = true;
  } else {
    thread_stat_->idle_timer.Start();
    std::this_thread::sleep_for(wait_time);
    thread_stat_->idle_timer.Stop();
  }
  return delayed;
}

void
RequestRateWorker::WaitForFreeCtx()
{
  if (!ctx_id_tracker_->IsAvailable()) {
    notified_ = false;
    // wait for signal from callback.
    std::unique_lock<std::mutex> lk(cb_mtx_);
    thread_stat_->idle_timer.Start();
    cb_cv_.wait(lk, [this] {
      if (notified_ || exiting_) {
        notified_ = false;
        return true;
      }
      return false;
    });
    thread_stat_->idle_timer.Stop();
  }
}

}}  // namespace triton::perfanalyzer
