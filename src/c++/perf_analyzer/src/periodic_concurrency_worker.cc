// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "periodic_concurrency_worker.h"

namespace triton { namespace perfanalyzer {

void
PeriodicConcurrencyWorker::Infer()
{
  CreateCtxIdTracker();
  ReserveContexts();
  RunInference();
}

std::shared_ptr<InferContext>
PeriodicConcurrencyWorker::CreateInferContext()
{
  std::shared_ptr infer_context{std::make_shared<InferContext>(
      id_, ctxs_.size(), async_, streaming_, on_sequence_model_,
      using_json_data_, batch_size_, thread_stat_, data_loader_, parser_,
      factory_, execute_, infer_data_manager_, sequence_manager_)};
  infer_context->RegisterWorkerCallback(worker_callback_);
  return infer_context;
}

void
PeriodicConcurrencyWorker::WorkerCallback(uint32_t infer_context_id)
{
  if (ctxs_.at(infer_context_id)->GetNumResponsesForCurrentRequest() ==
      request_period_) {
    period_completed_callback_();
  }
  if (ctxs_.at(infer_context_id)->HasReceivedFinalResponse()) {
    bool has_not_completed_period{
        ctxs_.at(infer_context_id)->GetNumResponsesForCurrentRequest() <
        request_period_};
    if (has_not_completed_period) {
      throw std::runtime_error(
          "Request received final response before request period was reached. "
          "Request period must be at most the total number of responses "
          "received by any request.");
    }
    request_completed_callback_();
  }
}

}}  // namespace triton::perfanalyzer
