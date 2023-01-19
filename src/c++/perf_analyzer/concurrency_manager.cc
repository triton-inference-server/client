// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "concurrency_manager.h"
#include <queue>

namespace triton { namespace perfanalyzer {

ConcurrencyManager::~ConcurrencyManager()
{
  // The destruction of derived class should wait for all the request generator
  // threads to finish
  StopWorkerThreads();
}

cb::Error
ConcurrencyManager::Create(
    const bool async, const bool streaming, const int32_t batch_size,
    const size_t max_threads, const size_t max_concurrency,
    const size_t sequence_length, const SharedMemoryType shared_memory_type,
    const size_t output_shm_size, const uint64_t start_sequence_id,
    const uint64_t sequence_id_range,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    std::unique_ptr<LoadManager>* manager)
{
  std::unique_ptr<ConcurrencyManager> local_manager(new ConcurrencyManager(
      async, streaming, batch_size, max_threads, max_concurrency,
      sequence_length, shared_memory_type, output_shm_size, start_sequence_id,
      sequence_id_range, parser, factory));

  *manager = std::move(local_manager);

  return cb::Error::Success;
}

ConcurrencyManager::ConcurrencyManager(
    const bool async, const bool streaming, const int32_t batch_size,
    const size_t max_threads, const size_t max_concurrency,
    const size_t sequence_length, const SharedMemoryType shared_memory_type,
    const size_t output_shm_size, const uint64_t start_sequence_id,
    const uint64_t sequence_id_range,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory)
    : LoadManager(
          async, streaming, batch_size, max_threads, sequence_length,
          shared_memory_type, output_shm_size, start_sequence_id,
          sequence_id_range, parser, factory),
      execute_(true), max_concurrency_(max_concurrency)
{
  if (on_sequence_model_) {
    for (uint64_t i = 0; i < max_concurrency_; i++) {
      sequence_stat_.emplace_back(new SequenceStat(0));
    }
  }

  threads_config_.reserve(max_threads);
}

cb::Error
ConcurrencyManager::ChangeConcurrencyLevel(
    const size_t concurrent_request_count)
{
  PauseSequenceWorkers();
  ReconfigThreads(concurrent_request_count);
  ResumeSequenceWorkers();

  std::cout << "Request concurrency: " << concurrent_request_count << std::endl;
  return cb::Error::Success;
}

void
ConcurrencyManager::PauseSequenceWorkers()
{
  if (on_sequence_model_) {
    execute_ = false;
    // Wait to see all threads are paused.
    for (auto& thread_config : threads_config_) {
      while (!thread_config->is_paused_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  }
}

void
ConcurrencyManager::ReconfigThreads(const size_t concurrent_request_count)
{
  // Always prefer to create new threads if the maximum limit has not been met
  //
  // While operating in synchronous mode, each context can send only one
  // request at a time, hence the number of worker threads should be equal to
  // the requested concurrency levels.
  //
  while ((concurrent_request_count > threads_.size()) &&
         (threads_.size() < max_threads_)) {
    // Launch new thread for inferencing
    threads_stat_.emplace_back(new ThreadStat());
    threads_config_.emplace_back(
        new ConcurrencyWorker::ThreadConfig(threads_config_.size()));

    auto worker = MakeWorker(threads_stat_.back(), threads_config_.back());

    threads_.emplace_back(&IWorker::Infer, worker);
  }

  {
    // Make sure all threads are reconfigured before they are woken up
    std::lock_guard<std::mutex> lock(wake_mutex_);

    // Compute the new concurrency level for each thread (take floor)
    // and spread the remaining value
    size_t avg_concurrency = concurrent_request_count / threads_.size();
    size_t threads_add_one = concurrent_request_count % threads_.size();

    active_threads_ = 0;
    for (size_t i = 0; i < threads_stat_.size(); i++) {
      threads_config_[i]->concurrency_ =
          avg_concurrency + (i < threads_add_one ? 1 : 0);
      if (threads_config_[i]->concurrency_) {
        active_threads_++;
      }
    }
  }
}

void
ConcurrencyManager::ResumeSequenceWorkers()
{
  if (on_sequence_model_) {
    UnpauseAllSequences();
    execute_ = true;
  }

  // Make sure all threads will check their updated concurrency level
  wake_signal_.notify_all();
}

std::shared_ptr<IWorker>
ConcurrencyManager::MakeWorker(
    std::shared_ptr<ThreadStat> thread_stat,
    std::shared_ptr<ConcurrencyWorker::ThreadConfig> thread_config)
{
  return std::make_shared<ConcurrencyWorker>(
      thread_stat, thread_config, parser_, data_loader_, backend_->Kind(),
      factory_, sequence_length_, start_sequence_id_, sequence_id_range_,
      on_sequence_model_, async_, max_concurrency_, using_json_data_,
      streaming_, shared_memory_type_, batch_size_, threads_config_,
      sequence_stat_, shared_memory_regions_, wake_signal_, wake_mutex_,
      active_threads_, execute_, curr_seq_id_, distribution_);
}

}}  // namespace triton::perfanalyzer
