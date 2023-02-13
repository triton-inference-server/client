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

#include "load_manager.h"

#include <algorithm>

#include "client_backend/client_backend.h"

namespace triton { namespace perfanalyzer {


cb::Error
LoadManager::CheckHealth()
{
  // Check thread status to make sure that the load setting is
  // consistent to the one being reported
  // If some thread return early, main thread will return and
  // the worker thread's error message will be reported
  // when derived class destructor gets called.
  for (auto& thread_stat : threads_stat_) {
    if (!thread_stat->status_.IsOk()) {
      return cb::Error(
          "Failed to maintain requested inference load."
          " Worker thread(s) failed to generate concurrent requests.",
          pa::GENERIC_ERROR);
    }
    if (!thread_stat->cb_status_.IsOk()) {
      return cb::Error(
          "Failed to retrieve results from inference request.",
          pa::GENERIC_ERROR);
    }
  }
  return cb::Error::Success;
}

cb::Error
LoadManager::SwapTimestamps(TimestampVector& new_timestamps)
{
  TimestampVector total_timestamp;
  // Gather request timestamps with proper locking from all the worker
  // threads
  for (auto& thread_stat : threads_stat_) {
    std::lock_guard<std::mutex> lock(thread_stat->mu_);
    total_timestamp.insert(
        total_timestamp.end(), thread_stat->request_timestamps_.begin(),
        thread_stat->request_timestamps_.end());
    thread_stat->request_timestamps_.clear();
  }
  // Swap the results
  total_timestamp.swap(new_timestamps);
  return cb::Error::Success;
}

uint64_t
LoadManager::CountCollectedRequests()
{
  uint64_t num_of_requests = 0;
  for (auto& thread_stat : threads_stat_) {
    std::lock_guard<std::mutex> lock(thread_stat->mu_);
    num_of_requests += thread_stat->request_timestamps_.size();
  }
  return num_of_requests;
}

cb::Error
LoadManager::GetAccumulatedClientStat(cb::InferStat* contexts_stat)
{
  contexts_stat->completed_request_count = 0;
  contexts_stat->cumulative_receive_time_ns = 0;
  contexts_stat->cumulative_send_time_ns = 0;
  contexts_stat->cumulative_total_request_time_ns = 0;

  for (auto& thread_stat : threads_stat_) {
    std::lock_guard<std::mutex> lock(thread_stat->mu_);
    for (auto& context_stat : thread_stat->contexts_stat_) {
      contexts_stat->completed_request_count +=
          context_stat.completed_request_count;
      contexts_stat->cumulative_total_request_time_ns +=
          context_stat.cumulative_total_request_time_ns;
      contexts_stat->cumulative_send_time_ns +=
          context_stat.cumulative_send_time_ns;
      contexts_stat->cumulative_receive_time_ns +=
          context_stat.cumulative_receive_time_ns;
    }
  }
  return cb::Error::Success;
}

LoadManager::LoadManager(
    const bool async, const bool streaming, const int32_t batch_size,
    const size_t max_threads, const size_t sequence_length,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const uint64_t start_sequence_id, const uint64_t sequence_id_range,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory)
    : async_(async), streaming_(streaming), batch_size_(batch_size),
      max_threads_(max_threads), sequence_length_(sequence_length),
      start_sequence_id_(start_sequence_id),
      sequence_id_range_(sequence_id_range), parser_(parser), factory_(factory),
      using_json_data_(false), curr_seq_id_(0)
{
  on_sequence_model_ =
      ((parser_->SchedulerType() == ModelParser::SEQUENCE) ||
       (parser_->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE));

  data_loader_.reset(new DataLoader(batch_size_));

  infer_data_manager_ = std::make_shared<pa::InferDataManager>(
      batch_size_, shared_memory_type, output_shm_size, parser_, factory_,
      data_loader_);
}

void
LoadManager::InitManager(
    const size_t string_length, const std::string& string_data,
    const bool zero_input, std::vector<std::string>& user_data)
{
  auto status =
      InitManagerInputs(string_length, string_data, zero_input, user_data);
  THROW_IF_ERROR(status, "Failed to init manager inputs");

  THROW_IF_ERROR(
      infer_data_manager_->InitSharedMemory(),
      "Unable to init shared memory in memory manager");
}

cb::Error
LoadManager::InitManagerInputs(
    const size_t string_length, const std::string& string_data,
    const bool zero_input, std::vector<std::string>& user_data)
{
  RETURN_IF_ERROR(factory_->CreateClientBackend(&backend_));

  // Read provided data
  if (!user_data.empty()) {
    if (IsDirectory(user_data[0])) {
      RETURN_IF_ERROR(data_loader_->ReadDataFromDir(
          parser_->Inputs(), parser_->Outputs(), user_data[0]));
    } else {
      using_json_data_ = true;
      for (const auto& json_file : user_data) {
        RETURN_IF_ERROR(data_loader_->ReadDataFromJSON(
            parser_->Inputs(), parser_->Outputs(), json_file));
      }
      distribution_ = std::uniform_int_distribution<uint64_t>(
          0, data_loader_->GetDataStreamsCount() - 1);
      std::cout << " Successfully read data for "
                << data_loader_->GetDataStreamsCount() << " stream/streams";
      if (data_loader_->GetDataStreamsCount() == 1) {
        std::cout << " with " << data_loader_->GetTotalSteps(0)
                  << " step/steps";
      }
      std::cout << "." << std::endl;
    }
  } else {
    RETURN_IF_ERROR(data_loader_->GenerateData(
        parser_->Inputs(), zero_input, string_length, string_data));
  }

  // Reserve the required vector space
  threads_stat_.reserve(max_threads_);

  return cb::Error::Success;
}

void
LoadManager::StopWorkerThreads()
{
  early_exit = true;
  // wake up all threads
  wake_signal_.notify_all();

  size_t cnt = 0;
  for (auto& thread : threads_) {
    thread.join();
    if (!threads_stat_[cnt]->status_.IsOk()) {
      std::cerr << "Thread [" << cnt
                << "] had error: " << (threads_stat_[cnt]->status_)
                << std::endl;
    }
    if (!threads_stat_[cnt]->cb_status_.IsOk()) {
      std::cerr << "Thread [" << cnt
                << "] had error: " << (threads_stat_[cnt]->cb_status_)
                << std::endl;
    }
    cnt++;
  }
  threads_.clear();
}

void
LoadManager::UnpauseAllSequences()
{
  for (auto seq : sequence_stat_) {
    std::lock_guard<std::mutex> guard(seq->mtx_);
    seq->paused_ = false;
  }
}

}}  // namespace triton::perfanalyzer
