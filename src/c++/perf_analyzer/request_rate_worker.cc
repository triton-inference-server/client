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
    uint32_t ctx_id = GetCtxId();
    PrepAndSendInferRequest(ctx_id, is_delayed);

    if (HandleExitConditions()) {
      return;
    }

  } while (true);
}


void
RequestRateWorker::CompleteOngoingSequences()
{
  if (on_sequence_model_) {
    for (size_t ctx_id = 0; ctx_id < ctxs_.size(); ++ctx_id) {
      // Finish off all the ongoing sequences for graceful exit
      for (size_t i = thread_config_->id_; i < sequence_stat_.size();
           i += thread_config_->stride_) {
        CompleteOngoingSequence(ctx_id, i);
      }
    }
  }
}

void
RequestRateWorker::HandleExecuteOff()
{
  // Should wait till main thread signals execution start
  if (!execute_) {
    CompleteOngoingSequences();
    WaitForOngoingRequests();

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
RequestRateWorker::UpdateJsonData(uint32_t ctx_id)
{
  LoadWorker::UpdateJsonData(
      std::static_pointer_cast<DataStepIdTracker>(thread_config_), ctx_id,
      max_threads_);
}

}}  // namespace triton::perfanalyzer
