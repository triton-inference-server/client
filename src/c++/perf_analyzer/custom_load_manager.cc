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

#include "custom_load_manager.h"

#include <fstream>

#include "constants.h"

namespace triton { namespace perfanalyzer {

cb::Error
CustomLoadManager::Create(
    const bool async, const bool streaming,
    const uint64_t measurement_window_ms, const size_t max_trials,
    const std::string& request_intervals_file, const int32_t batch_size,
    const size_t max_threads, const uint32_t num_of_sequences,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    std::unique_ptr<LoadManager>* manager,
    const std::unordered_map<std::string, cb::RequestParameter>&
        request_parameters)
{
  std::unique_ptr<CustomLoadManager> local_manager(new CustomLoadManager(
      async, streaming, request_intervals_file, batch_size,
      measurement_window_ms, max_trials, max_threads, num_of_sequences,
      shared_memory_type, output_shm_size, serial_sequences, parser, factory,
      request_parameters));

  *manager = std::move(local_manager);

  return cb::Error::Success;
}

CustomLoadManager::CustomLoadManager(
    const bool async, const bool streaming,
    const std::string& request_intervals_file, int32_t batch_size,
    const uint64_t measurement_window_ms, const size_t max_trials,
    const size_t max_threads, const uint32_t num_of_sequences,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    const std::unordered_map<std::string, cb::RequestParameter>&
        request_parameters)
    : RequestRateManager(
          async, streaming, Distribution::CUSTOM, batch_size,
          measurement_window_ms, max_trials, max_threads, num_of_sequences,
          shared_memory_type, output_shm_size, serial_sequences, parser,
          factory, request_parameters),
      request_intervals_file_(request_intervals_file)
{
}

cb::Error
CustomLoadManager::InitCustomIntervals(const size_t request_count)
{
  PauseWorkers();
  ConfigureThreads(request_count);
  auto status = GenerateSchedule();
  ResumeWorkers();
  return status;
}

cb::Error
CustomLoadManager::GenerateSchedule()
{
  if (request_intervals_file_.empty()) {
    return cb::Error::Success;
  }

  RETURN_IF_ERROR(
      ReadTimeIntervalsFile(request_intervals_file_, &custom_intervals_));

  auto worker_schedules = CreateWorkerSchedules();
  GiveSchedulesToWorkers(worker_schedules);
  return cb::Error::Success;
}

std::vector<RateSchedulePtr_t>
CustomLoadManager::CreateWorkerSchedules()
{
  std::vector<RateSchedulePtr_t> worker_schedules =
      CreateEmptyWorkerSchedules();
  std::vector<size_t> thread_ids{CalculateThreadIds()};

  size_t thread_id_index = 0;
  size_t worker_index = 0;
  size_t intervals_index = 0;

  std::chrono::nanoseconds next_timestamp(0);

  bool started = false;

  // Keep filling the schedule until both the thread_ids (which can differ if
  // sequences are enabled) and the intervals are both at the end of their
  // lists. This effectively finds the least common multiple of the two sizes
  // and makes sure that the schedule is complete and can be repeated
  // indefinitely
  //
  while (!started || thread_id_index != 0 || intervals_index != 0) {
    started = true;
    next_timestamp += custom_intervals_[intervals_index];
    worker_index = thread_ids[thread_id_index];
    worker_schedules[worker_index]->intervals.emplace_back(next_timestamp);

    thread_id_index = (thread_id_index + 1) % thread_ids.size();
    intervals_index = (intervals_index + 1) % custom_intervals_.size();
  }

  SetScheduleDurations(worker_schedules);

  return worker_schedules;
}

cb::Error
CustomLoadManager::GetCustomRequestRate(double* request_rate)
{
  if (custom_intervals_.empty()) {
    return cb::Error("The custom intervals vector is empty", pa::GENERIC_ERROR);
  }
  uint64_t total_time_ns = 0;
  for (auto interval : custom_intervals_) {
    total_time_ns += interval.count();
  }

  *request_rate =
      (custom_intervals_.size() * NANOS_PER_SECOND) / (total_time_ns);
  return cb::Error::Success;
}

cb::Error
CustomLoadManager::ReadTimeIntervalsFile(
    const std::string& path, NanoIntervals* contents)
{
  std::ifstream in(path);
  if (!in) {
    return cb::Error("failed to open file '" + path + "'", pa::GENERIC_ERROR);
  }

  std::string current_string;
  while (std::getline(in, current_string)) {
    std::chrono::nanoseconds curent_time_interval_ns(
        std::stol(current_string) * 1000);
    contents->push_back(curent_time_interval_ns);
  }
  in.close();

  if (contents->size() == 0) {
    return cb::Error("file '" + path + "' is empty", pa::GENERIC_ERROR);
  }
  return cb::Error::Success;
}

}}  // namespace triton::perfanalyzer
