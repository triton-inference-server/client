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

#include "periodic_concurrency_manager.h"

namespace triton { namespace perfanalyzer {

std::vector<RequestRecord>
PeriodicConcurrencyManager::RunExperiment()
{
  AddConcurrentRequests(concurrency_range_.start);
  WaitForRequestsToFinish();
  return GetRequestRecords();
}

std::shared_ptr<IWorker>
PeriodicConcurrencyManager::MakeWorker(
    std::shared_ptr<ThreadStat> thread_stat,
    std::shared_ptr<PeriodicConcurrencyWorker::ThreadConfig> thread_config)
{
  uint32_t id = workers_.size();

  auto worker = std::make_shared<PeriodicConcurrencyWorker>(
      id, thread_stat, thread_config, parser_, data_loader_, factory_,
      on_sequence_model_, async_, max_concurrency_, using_json_data_,
      streaming_, batch_size_, wake_signal_, wake_mutex_, active_threads_,
      execute_, infer_data_manager_, sequence_manager_, request_period_,
      period_completed_callback_, request_completed_callback_);
  return worker;
};

void
PeriodicConcurrencyManager::AddConcurrentRequests(
    uint64_t num_concurrent_requests)
{
  for (size_t i = 0; i < num_concurrent_requests; i++) {
    threads_stat_.emplace_back(std::make_shared<ThreadStat>());
    threads_config_.emplace_back(
        std::make_shared<ConcurrencyWorker::ThreadConfig>(
            threads_config_.size(), 1, i));
    workers_.emplace_back(
        MakeWorker(threads_stat_.back(), threads_config_.back()));
    threads_.emplace_back(&IWorker::Infer, workers_.back());
    active_threads_++;
  }
  num_incomplete_periods_ = num_concurrent_requests;
}

void
PeriodicConcurrencyManager::PeriodCompletedCallback()
{
  std::lock_guard<std::mutex> lock(period_completed_callback_mutex_);

  num_incomplete_periods_--;

  if (num_incomplete_periods_ == 0) {
    steps_completed_++;
    if (steps_completed_ * concurrency_range_.step < concurrency_range_.end) {
      AddConcurrentRequests(concurrency_range_.step);
    }
  }
}

void
PeriodicConcurrencyManager::RequestCompletedCallback()
{
  std::lock_guard<std::mutex> lock(request_completed_callback_mutex_);

  num_completed_requests_++;

  if (num_completed_requests_ == concurrency_range_.end) {
    all_requests_completed_promise_.set_value(true);
  }
}

void
PeriodicConcurrencyManager::WaitForRequestsToFinish()
{
  std::future<bool> all_requests_completed_future{
      all_requests_completed_promise_.get_future()};
  all_requests_completed_future.get();
}

std::vector<RequestRecord>
PeriodicConcurrencyManager::GetRequestRecords()
{
  std::vector<RequestRecord> request_records{};

  for (const auto& thread_stat : threads_stat_) {
    request_records.insert(
        request_records.end(), thread_stat->request_records_.cbegin(),
        thread_stat->request_records_.cend());
  }

  return request_records;
}

}}  // namespace triton::perfanalyzer
