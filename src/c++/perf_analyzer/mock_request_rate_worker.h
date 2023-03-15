// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gmock/gmock.h"
#include "request_rate_worker.h"

namespace triton { namespace perfanalyzer {

class MockRequestRateWorker : public RequestRateWorker {
 public:
  MockRequestRateWorker(
      uint32_t id, std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config,
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<DataLoader> data_loader,
      const std::shared_ptr<cb::ClientBackendFactory> factory,
      const bool on_sequence_model, const bool async, const size_t max_threads,
      const bool using_json_data, const bool streaming,
      const int32_t batch_size, std::condition_variable& wake_signal,
      std::mutex& wake_mutex, bool& execute,
      std::chrono::steady_clock::time_point& start_time,
      const std::shared_ptr<IInferDataManager>& infer_data_manager,
      std::shared_ptr<SequenceManager> sequence_manager)
      : RequestRateWorker(
            id, thread_stat, thread_config, parser, data_loader, factory,
            on_sequence_model, async, max_threads, using_json_data, streaming,
            batch_size, wake_signal, wake_mutex, execute, start_time,
            infer_data_manager, sequence_manager)
  {
    ON_CALL(*this, Infer()).WillByDefault([this]() -> void {
      return RequestRateWorker::Infer();
    });
  }

  MOCK_METHOD(void, Infer, (), (override));

  void SendInferRequest()
  {
    if (!context_created) {
      CreateContext();
      context_created = true;
    }
    if (thread_stat_->status_.IsOk()) {
      LoadWorker::SendInferRequest(0, false);
    }
  }

  void EmptyInfer() { thread_config_->is_paused_ = true; }

 private:
  bool context_created{false};
  std::shared_ptr<ThreadConfig>& thread_config_{
      RequestRateWorker::thread_config_};
};

}}  // namespace triton::perfanalyzer
