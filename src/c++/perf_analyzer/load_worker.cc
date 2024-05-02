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

#include "load_worker.h"

#include <algorithm>
#include <thread>

#include "client_backend/client_backend.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

bool
LoadWorker::ShouldExit()
{
  return early_exit || !thread_stat_->cb_status_.IsOk() ||
         !thread_stat_->status_.IsOk() ||
         (thread_stat_->max_requests_ &&
          thread_stat_->num_sent_requests_ >= thread_stat_->max_requests_);
}

bool
LoadWorker::HandleExitConditions()
{
  if (ShouldExit()) {
    thread_stat_->idle_timer.Start();
    CompleteOngoingSequences();
    WaitForOngoingRequests();
    return true;
  }
  return false;
}

void
LoadWorker::CompleteOngoingSequences()
{
  if (on_sequence_model_) {
    for (size_t ctx_id = 0; ctx_id < ctxs_.size(); ++ctx_id) {
      size_t seq_stat_index = GetSeqStatIndex(ctx_id);
      ctxs_[ctx_id]->CompleteOngoingSequence(seq_stat_index);
    }
  }
}

void
LoadWorker::WaitForOngoingRequests()
{
  while (GetNumOngoingRequests() != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

uint
LoadWorker::GetNumOngoingRequests()
{
  uint num = 0;
  for (auto ctx : ctxs_) {
    num += ctx->GetNumOngoingRequests();
  }
  return num;
}

void
LoadWorker::CreateContext()
{
  auto ctx = CreateInferContext();
  ctx->Init();
  CreateContextFinalize(ctx);
  ctxs_.push_back(ctx);
}

uint32_t
LoadWorker::GetCtxId()
{
  std::lock_guard<std::mutex> lk(cb_mtx_);
  return ctx_id_tracker_->Get();
}


void
LoadWorker::RestoreFreeCtxId(uint32_t ctx_id)
{
  if (!async_) {
    {
      std::lock_guard<std::mutex> lock(cb_mtx_);
      ctx_id_tracker_->Restore(ctx_id);
    }
  }
}

void
LoadWorker::AsyncCallbackFinalize(uint32_t ctx_id)
{
  // avoid competition over 'cb_mtx_'
  {
    std::lock_guard<std::mutex> lk(cb_mtx_);
    ctx_id_tracker_->Restore(ctx_id);
    notified_ = true;
  }

  cb_cv_.notify_all();
}

}}  // namespace triton::perfanalyzer
