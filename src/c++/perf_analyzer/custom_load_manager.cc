// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

namespace triton { namespace perfanalyzer {

cb::Error
CustomLoadManager::Create(
    const bool async, const bool streaming,
    const uint64_t measurement_window_ms,
    const std::string& request_intervals_file, const int32_t batch_size,
    const size_t max_threads, const uint32_t num_of_sequences,
    const size_t sequence_length, const size_t string_length,
    const std::string& string_data, const bool zero_input,
    std::vector<std::string>& user_data,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const uint64_t start_sequence_id, const uint64_t sequence_id_range,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    std::unique_ptr<LoadManager>* manager)
{
  std::unique_ptr<CustomLoadManager> local_manager(new CustomLoadManager(
      async, streaming, request_intervals_file, batch_size,
      measurement_window_ms, max_threads, num_of_sequences, sequence_length,
      shared_memory_type, output_shm_size, start_sequence_id, sequence_id_range,
      parser, factory));

  local_manager->threads_config_.reserve(max_threads);

  RETURN_IF_ERROR(local_manager->InitManagerInputs(
      string_length, string_data, zero_input, user_data));

  if (local_manager->shared_memory_type_ !=
      SharedMemoryType::NO_SHARED_MEMORY) {
    RETURN_IF_ERROR(local_manager->InitSharedMemory());
  }

  *manager = std::move(local_manager);

  return cb::Error::Success;
}

CustomLoadManager::CustomLoadManager(
    const bool async, const bool streaming,
    const std::string& request_intervals_file, int32_t batch_size,
    const uint64_t measurement_window_ms, const size_t max_threads,
    const uint32_t num_of_sequences, const size_t sequence_length,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const uint64_t start_sequence_id, const uint64_t sequence_id_range,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory)
    : RequestRateManager(
          async, streaming, Distribution::CUSTOM, batch_size,
          measurement_window_ms, max_threads, num_of_sequences, sequence_length,
          shared_memory_type, output_shm_size, start_sequence_id,
          sequence_id_range, parser, factory),
      request_intervals_file_(request_intervals_file)
{
}

cb::Error
CustomLoadManager::InitCustomIntervals()
{
  schedule_.clear();
  schedule_.emplace_back(0);
  if (!request_intervals_file_.empty()) {
    RETURN_IF_ERROR(
        ReadTimeIntervalsFile(request_intervals_file_, &custom_intervals_));
    size_t index = 0;
    while (schedule_.back() < *gen_duration_) {
      std::chrono::nanoseconds next_timestamp(
          schedule_.back() + custom_intervals_[index++]);
      schedule_.emplace_back(next_timestamp);
      if (index == custom_intervals_.size()) {
        index = 0;
      }
    }
  }
  return cb::Error::Success;
}

cb::Error
CustomLoadManager::GetCustomRequestRate(double* request_rate)
{
  if (custom_intervals_.empty()) {
    return cb::Error("The custom intervals vector is empty");
  }
  uint64_t total_time_ns = 0;
  for (auto interval : custom_intervals_) {
    total_time_ns += interval.count();
  }

  *request_rate =
      (custom_intervals_.size() * 1000 * 1000 * 1000) / (total_time_ns);
  return cb::Error::Success;
}

}}  // namespace triton::perfanalyzer
