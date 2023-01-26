// Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "request_rate_manager.h"

namespace triton { namespace perfanalyzer {

RequestRateManager::~RequestRateManager()
{
  // The destruction of derived class should wait for all the request generator
  // threads to finish
  StopWorkerThreads();
}

cb::Error
RequestRateManager::Create(
    const bool async, const bool streaming,
    const uint64_t measurement_window_ms, Distribution request_distribution,
    const int32_t batch_size, const size_t max_threads,
    const uint32_t num_of_sequences, const size_t sequence_length,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const uint64_t start_sequence_id, const uint64_t sequence_id_range,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    std::unique_ptr<LoadManager>* manager)
{
  std::unique_ptr<RequestRateManager> local_manager(new RequestRateManager(
      async, streaming, request_distribution, batch_size, measurement_window_ms,
      max_threads, num_of_sequences, sequence_length, shared_memory_type,
      output_shm_size, start_sequence_id, sequence_id_range, parser, factory));

  *manager = std::move(local_manager);

  return cb::Error::Success;
}

RequestRateManager::RequestRateManager(
    const bool async, const bool streaming, Distribution request_distribution,
    int32_t batch_size, const uint64_t measurement_window_ms,
    const size_t max_threads, const uint32_t num_of_sequences,
    const size_t sequence_length, const SharedMemoryType shared_memory_type,
    const size_t output_shm_size, const uint64_t start_sequence_id,
    const uint64_t sequence_id_range,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory)
    : LoadManager(
          async, streaming, batch_size, max_threads, sequence_length,
          shared_memory_type, output_shm_size, start_sequence_id,
          sequence_id_range, parser, factory),
      request_distribution_(request_distribution), execute_(false)
{
  if (on_sequence_model_) {
    for (uint64_t i = 0; i < num_of_sequences; i++) {
      sequence_stat_.emplace_back(new SequenceStat(0));
    }
  }
  gen_duration_.reset(
      new std::chrono::nanoseconds(2 * measurement_window_ms * 1000 * 1000));

  threads_config_.reserve(max_threads);
}

cb::Error
RequestRateManager::ChangeRequestRate(const double request_rate)
{
  PauseWorkers();
  // Can safely update the schedule
  GenerateSchedule(request_rate);
  ResumeWorkers();

  return cb::Error::Success;
}

cb::Error
RequestRateManager::ResetWorkers()
{
  PauseWorkers();
  ResumeWorkers();

  return cb::Error::Success;
}

void
RequestRateManager::GenerateSchedule(const double request_rate)
{
  std::function<std::chrono::nanoseconds(std::mt19937&)> distribution;
  if (request_distribution_ == Distribution::POISSON) {
    distribution = ScheduleDistribution<Distribution::POISSON>(request_rate);
  } else if (request_distribution_ == Distribution::CONSTANT) {
    distribution = ScheduleDistribution<Distribution::CONSTANT>(request_rate);
  } else {
    return;
  }
  schedule_.clear();

  std::chrono::nanoseconds next_timestamp(0);

  std::mt19937 schedule_rng;
  while (next_timestamp < *gen_duration_) {
    schedule_.emplace_back(next_timestamp);
    next_timestamp = schedule_.back() + distribution(schedule_rng);
  }
}

void
RequestRateManager::PauseWorkers()
{
  // Pause all the threads
  execute_ = false;

  if (threads_.empty()) {
    while (threads_.size() < max_threads_) {
      // Launch new thread for inferencing
      threads_stat_.emplace_back(new ThreadStat());
      threads_config_.emplace_back(
          new RequestRateWorker::ThreadConfig(threads_.size(), max_threads_));

      auto worker = MakeWorker(threads_stat_.back(), threads_config_.back());

      threads_.emplace_back(&IWorker::Infer, worker);
    }
  }

  // Wait to see all threads are paused.
  for (auto& thread_config : threads_config_) {
    while (!thread_config->is_paused_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void
RequestRateManager::ResumeWorkers()
{
  // Reset all the thread counters
  for (auto& thread_config : threads_config_) {
    thread_config->index_ = thread_config->id_;
    thread_config->rounds_ = 0;
  }

  // Update the start_time_ to point to current time
  start_time_ = std::chrono::steady_clock::now();

  // Wake up all the threads to begin execution
  execute_ = true;
  wake_signal_.notify_all();
}

std::shared_ptr<IWorker>
RequestRateManager::MakeWorker(
    std::shared_ptr<ThreadStat> thread_stat,
    std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config)
{
  return std::make_shared<RequestRateWorker>(
      thread_stat, thread_config, parser_, data_loader_, backend_->Kind(),
      factory_, sequence_length_, start_sequence_id_, sequence_id_range_,
      on_sequence_model_, async_, max_threads_, using_json_data_, streaming_,
      shared_memory_type_, batch_size_, sequence_stat_, shared_memory_regions_,
      wake_signal_, wake_mutex_, execute_, curr_seq_id_, start_time_, schedule_,
      gen_duration_, distribution_, memory_manager_);
}


}}  // namespace triton::perfanalyzer
