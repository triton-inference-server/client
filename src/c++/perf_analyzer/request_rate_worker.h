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
#pragma once

#include <memory>

#include "ischeduler.h"
#include "load_worker.h"
#include "model_parser.h"
#include "sequence_manager.h"

namespace triton { namespace perfanalyzer {


#ifndef DOCTEST_CONFIG_DISABLE
class NaggyMockRequestRateWorker;
class TestRequestRateManager;
class TestCustomLoadManager;
#endif

/// Worker thread for RequestRateManager
///
/// If the model is non-sequence model, each worker uses only one context
/// to maintain concurrency assigned to worker.
/// If the model is sequence model, each worker has to use multiples contexts
/// to maintain (sequence) concurrency assigned to worker.
///
class RequestRateWorker : public LoadWorker, public IScheduler {
 public:
  struct ThreadConfig {
    ThreadConfig(uint32_t index, uint32_t stride)
        : id_(index), stride_(stride), is_paused_(false)
    {
    }

    uint32_t id_;
    uint32_t stride_;
    bool is_paused_;
  };

  RequestRateWorker(
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
      : LoadWorker(
            id, thread_stat, parser, data_loader, factory, on_sequence_model,
            async, streaming, batch_size, using_json_data, wake_signal,
            wake_mutex, execute, infer_data_manager, sequence_manager),
        thread_config_(thread_config), max_threads_(max_threads),
        start_time_(start_time)
  {
  }

  void Infer() override;

  /// Provides the schedule that should be followed
  ///
  void SetSchedule(RateSchedulePtr_t schedule) override;

 private:
  RateSchedulePtr_t schedule_;

  const size_t max_threads_;
  std::chrono::steady_clock::time_point& start_time_;

  std::shared_ptr<ThreadConfig> thread_config_;

  std::chrono::nanoseconds GetNextTimestamp();

  // Request Rate Worker only ever has a single context
  uint32_t GetCtxId() { return 0; }

  uint32_t GetSeqStatIndex(uint32_t ctx_id) override
  {
    return (rand() % sequence_manager_->GetNumSequenceStatuses());
  }

  void CompleteOngoingSequences() override;

  void HandleExecuteOff();

  // Sleep until it is time for the next part of the schedule
  // Returns true if the request was delayed
  bool SleepIfNecessary();

  void CreateContextFinalize(std::shared_ptr<InferContext> ctx) override
  {
    ctx->SetNumActiveThreads(max_threads_);
  }

#ifndef DOCTEST_CONFIG_DISABLE
  friend NaggyMockRequestRateWorker;
  friend TestCustomLoadManager;
  friend TestRequestRateManager;

#endif
};


}}  // namespace triton::perfanalyzer
