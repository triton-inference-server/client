// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    const uint64_t measurement_window_ms, const size_t max_trials,
    Distribution request_distribution, const int32_t batch_size,
    const size_t max_threads, const uint32_t num_of_sequences,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    std::unique_ptr<LoadManager>* manager,
    const std::unordered_map<std::string, cb::RequestParameter>&
        request_parameters)
{
  std::unique_ptr<RequestRateManager> local_manager(new RequestRateManager(
      async, streaming, request_distribution, batch_size, measurement_window_ms,
      max_trials, max_threads, num_of_sequences, shared_memory_type,
      output_shm_size, serial_sequences, parser, factory, request_parameters));

  *manager = std::move(local_manager);

  return cb::Error::Success;
}

RequestRateManager::RequestRateManager(
    const bool async, const bool streaming, Distribution request_distribution,
    int32_t batch_size, const uint64_t measurement_window_ms,
    const size_t max_trials, const size_t max_threads,
    const uint32_t num_of_sequences, const SharedMemoryType shared_memory_type,
    const size_t output_shm_size, const bool serial_sequences,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    const std::unordered_map<std::string, cb::RequestParameter>&
        request_parameters)
    : LoadManager(
          async, streaming, batch_size, max_threads, shared_memory_type,
          output_shm_size, parser, factory, request_parameters),
      request_distribution_(request_distribution), execute_(false),
      num_of_sequences_(num_of_sequences), serial_sequences_(serial_sequences)
{
  gen_duration_.reset(new std::chrono::nanoseconds(
      max_trials * measurement_window_ms * NANOS_PER_MILLIS));

  threads_config_.reserve(max_threads);
}

void
RequestRateManager::InitManagerFinalize()
{
  if (on_sequence_model_) {
    sequence_manager_->InitSequenceStatuses(num_of_sequences_);
  }
}

cb::Error
RequestRateManager::ChangeRequestRate(
    const double request_rate, const size_t request_count)
{
  PauseWorkers();
  ConfigureThreads(request_count);
  // Can safely update the schedule
  GenerateSchedule(request_rate);
  ResumeWorkers();

  return cb::Error::Success;
}

void
RequestRateManager::GenerateSchedule(const double request_rate)
{
  std::chrono::nanoseconds max_duration;
  std::function<std::chrono::nanoseconds(std::mt19937&)> distribution;

  if (request_distribution_ == Distribution::POISSON) {
    distribution = ScheduleDistribution<Distribution::POISSON>(request_rate);
    // Poisson distribution needs to generate a schedule for the maximum
    // possible duration to make sure that it is as random and as close to the
    // desired rate as possible
    max_duration = *gen_duration_;
  } else if (request_distribution_ == Distribution::CONSTANT) {
    distribution = ScheduleDistribution<Distribution::CONSTANT>(request_rate);
    // Constant distribution only needs one entry per worker -- that one value
    // can be repeated over and over to emulate a full schedule of any length
    max_duration = std::chrono::nanoseconds(1);
  } else {
    return;
  }

  auto worker_schedules = CreateWorkerSchedules(max_duration, distribution);
  GiveSchedulesToWorkers(worker_schedules);
}

std::vector<RateSchedulePtr_t>
RequestRateManager::CreateWorkerSchedules(
    std::chrono::nanoseconds max_duration,
    std::function<std::chrono::nanoseconds(std::mt19937&)> distribution)
{
  std::mt19937 schedule_rng;

  std::vector<RateSchedulePtr_t> worker_schedules =
      CreateEmptyWorkerSchedules();
  std::vector<size_t> thread_ids{CalculateThreadIds()};

  std::chrono::nanoseconds next_timestamp(0);
  size_t thread_id_index = 0;
  size_t worker_index = 0;


  // Generate schedule until we hit max_duration, but also make sure that all
  // worker schedules follow the thread id distribution
  //
  while (next_timestamp < max_duration ||
         thread_id_index % thread_ids.size() != 0) {
    next_timestamp = next_timestamp + distribution(schedule_rng);
    worker_index = thread_ids[thread_id_index];
    thread_id_index = ++thread_id_index % thread_ids.size();
    worker_schedules[worker_index]->intervals.emplace_back(next_timestamp);
  }

  SetScheduleDurations(worker_schedules);

  return worker_schedules;
}

std::vector<RateSchedulePtr_t>
RequestRateManager::CreateEmptyWorkerSchedules()
{
  std::vector<RateSchedulePtr_t> worker_schedules;
  for (size_t i = 0; i < workers_.size(); i++) {
    worker_schedules.push_back(std::make_shared<RateSchedule>());
  }
  return worker_schedules;
}

std::vector<size_t>
RequestRateManager::CalculateThreadIds()
{
  std::vector<size_t> thread_ids{};
  // Determine number of ids to loop over for time stamps
  size_t num_ids = 0;
  if (on_sequence_model_) {
    num_ids = num_of_sequences_;
  } else {
    num_ids = max_threads_;
  }

  for (size_t i = 0; i < num_ids; i++) {
    size_t t = i % DetermineNumThreads();
    thread_ids.push_back(t);
  }
  return thread_ids;
}

void
RequestRateManager::SetScheduleDurations(
    std::vector<RateSchedulePtr_t>& schedules)
{
  RateSchedulePtr_t last_schedule = schedules.back();

  std::chrono::nanoseconds duration = last_schedule->intervals.back();

  for (auto schedule : schedules) {
    duration = std::max(schedule->intervals.back(), duration);
  }

  for (auto schedule : schedules) {
    schedule->duration = duration;
  }
}


void
RequestRateManager::GiveSchedulesToWorkers(
    const std::vector<RateSchedulePtr_t>& worker_schedules)
{
  for (size_t i = 0; i < workers_.size(); i++) {
    auto w = std::dynamic_pointer_cast<IScheduler>(workers_[i]);
    w->SetSchedule(worker_schedules[i]);
  }
}

void
RequestRateManager::PauseWorkers()
{
  // Pause all the threads
  execute_ = false;

  // Wait to see all threads are paused.
  for (auto& thread_config : threads_config_) {
    while (!thread_config->is_paused_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void
RequestRateManager::ConfigureThreads(const size_t request_count)
{
  if (threads_.empty()) {
    size_t num_of_threads = DetermineNumThreads();
    while (workers_.size() < num_of_threads) {
      // Launch new thread for inferencing
      threads_stat_.emplace_back(new ThreadStat());
      threads_config_.emplace_back(new ThreadConfig(workers_.size()));

      workers_.push_back(
          MakeWorker(threads_stat_.back(), threads_config_.back()));
    }
    // Compute the number of sequences for each thread (take floor)
    // and spread the remaining value
    size_t avg_num_seqs = num_of_sequences_ / workers_.size();
    size_t num_seqs_add_one = num_of_sequences_ % workers_.size();
    size_t seq_offset = 0;

    size_t avg_req_count = request_count / workers_.size();
    size_t req_count_add_one = request_count % workers_.size();


    for (size_t i = 0; i < workers_.size(); i++) {
      size_t num_of_seq = avg_num_seqs + (i < num_seqs_add_one ? 1 : 0);
      threads_config_[i]->num_sequences_ = num_of_seq;
      threads_config_[i]->seq_stat_index_offset_ = seq_offset;
      seq_offset += num_of_seq;

      size_t thread_num_reqs = avg_req_count + (i < req_count_add_one ? 1 : 0);
      threads_config_[i]->num_requests_ = thread_num_reqs;

      threads_.emplace_back(&IWorker::Infer, workers_[i]);
    }
  }
}

void
RequestRateManager::ResumeWorkers()
{
  // Update the start_time_ to point to current time
  start_time_ = std::chrono::steady_clock::now();

  // Wake up all the threads to begin execution
  execute_ = true;
  wake_signal_.notify_all();
}

std::shared_ptr<IWorker>
RequestRateManager::MakeWorker(
    std::shared_ptr<ThreadStat> thread_stat,
    std::shared_ptr<ThreadConfig> thread_config)
{
  size_t id = workers_.size();
  size_t num_of_threads = DetermineNumThreads();
  return std::make_shared<RequestRateWorker>(
      id, thread_stat, thread_config, parser_, data_loader_, factory_,
      on_sequence_model_, async_, num_of_threads, using_json_data_, streaming_,
      batch_size_, wake_signal_, wake_mutex_, execute_, start_time_,
      serial_sequences_, infer_data_manager_, sequence_manager_);
}

size_t
RequestRateManager::DetermineNumThreads()
{
  size_t num_of_threads = max_threads_;
  if (on_sequence_model_) {
    num_of_threads = std::min(max_threads_, num_of_sequences_);
  }
  return num_of_threads;
}


}}  // namespace triton::perfanalyzer
