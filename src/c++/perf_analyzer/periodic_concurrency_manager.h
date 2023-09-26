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
#pragma once

#include <future>

#include "concurrency_manager.h"
#include "periodic_concurrency_worker.h"

namespace triton { namespace perfanalyzer {

/// @brief Concurrency manager for periodically increasing concurrency by a step
/// amount based on the number of responses received (request period) by the
/// latest N (step or start concurrency for first-issued concurrent requests)
/// concurrent requests/workers.
class PeriodicConcurrencyManager : public ConcurrencyManager {
 public:
  PeriodicConcurrencyManager(
      const bool async, const bool streaming, const int32_t batch_size,
      const size_t max_threads, const size_t max_concurrency,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const Range<uint64_t> concurrency_range, const uint64_t request_period)
      : ConcurrencyManager(
            async, streaming, batch_size, max_threads, max_concurrency,
            shared_memory_type, output_shm_size, parser, factory),
        concurrency_range_(concurrency_range), request_period_(request_period)
  {
  }

  std::vector<RequestRecord> RunExperiment();

 private:
  std::shared_ptr<IWorker> MakeWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<PeriodicConcurrencyWorker::ThreadConfig> thread_config)
      override;

  void MaybeAddConcurrentRequests();

  void AddConcurrentRequests(uint64_t num_concurrent_requests);

  void AddConcurrentRequest(size_t seq_stat_index_offset);

  void PeriodCompletedCallback();

  void RequestCompletedCallback();

  void WaitForRequestsToFinish();

  std::vector<RequestRecord> GetRequestRecords();

  Range<uint64_t> concurrency_range_{1, 1, 1};
  uint64_t request_period_{0};
  uint64_t steps_completed_{0};
  uint64_t num_incomplete_periods_{0};
  uint64_t num_completed_requests_{0};
  std::mutex period_completed_callback_mutex_{};
  std::mutex request_completed_callback_mutex_{};
  std::promise<bool> all_requests_completed_promise_{};
  std::function<void()> period_completed_callback_{
      std::bind(&PeriodicConcurrencyManager::PeriodCompletedCallback, this)};
  std::function<void()> request_completed_callback_{
      std::bind(&PeriodicConcurrencyManager::RequestCompletedCallback, this)};
};

}}  // namespace triton::perfanalyzer
