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
#include <random>

#include "load_worker.h"
#include "sequence_manager.h"

namespace triton { namespace perfanalyzer {


#ifndef DOCTEST_CONFIG_DISABLE
class NaggyMockConcurrencyWorker;
#endif

/// Worker thread for the ConcurrencyManager
///
/// The worker maintains concurrency in different ways:
///   For sequence models, multiple contexts must be created for multiple
///   concurrent sequences.
///
///   For non-sequence models, one context can send out multiple requests
///   at the same time. Thus it uses one single context as every infer context
///   creates a worker thread implicitly.
///
class ConcurrencyWorker : public LoadWorker {
 public:
  struct ThreadConfig {
    ThreadConfig(size_t thread_id)
        : thread_id_(thread_id), concurrency_(0), seq_stat_index_offset_(0),
          is_paused_(false)
    {
    }

    // ID of corresponding worker thread
    size_t thread_id_;

    // The concurrency level that the worker should produce
    size_t concurrency_;

    // The starting sequence stat index for this worker
    size_t seq_stat_index_offset_;

    // Whether or not the thread is issuing new inference requests
    bool is_paused_;
  };

  ConcurrencyWorker(
      uint32_t id, std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config,
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<DataLoader> data_loader,
      const std::shared_ptr<cb::ClientBackendFactory> factory,
      const bool on_sequence_model, const bool async,
      const size_t max_concurrency, const bool using_json_data,
      const bool streaming, const int32_t batch_size,
      std::condition_variable& wake_signal, std::mutex& wake_mutex,
      size_t& active_threads, bool& execute,
      const std::shared_ptr<IInferDataManager>& infer_data_manager,
      std::shared_ptr<SequenceManager> sequence_manager)
      : LoadWorker(
            id, thread_stat, parser, data_loader, factory, on_sequence_model,
            async, streaming, batch_size, using_json_data, wake_signal,
            wake_mutex, execute, infer_data_manager, sequence_manager),
        thread_config_(thread_config), max_concurrency_(max_concurrency),
        active_threads_(active_threads)
  {
  }

  void Infer() override;

 private:
  const size_t max_concurrency_;
  // TODO REFACTOR TMA-1020 can we decouple this thread from the total count of
  // threads?
  size_t& active_threads_;

  std::shared_ptr<ThreadConfig> thread_config_;

  void CompleteOngoingSequences() override;

  // Reserve vector size for contexts
  void ReserveContexts();

  // Handle the case where execute_ is false
  void HandleExecuteOff();

  // Handle the case where this thread is configured to do nothing
  // Returns true if an exit condition was met
  bool HandleNoConcurrency();

  // Create and populate contexts if needed
  void CreateContextsAsNecessary();

  // Send out the desired concurrency of requests
  void SendInferRequests();

  void WaitForResponses();

  void ResetFreeCtxIds();

  uint32_t GetSeqStatIndex(uint32_t ctx_id) override;

  void CreateContextFinalize(std::shared_ptr<InferContext> ctx) override
  {
    ctx->RegisterAsyncCallbackFinalize(std::bind(
        &ConcurrencyWorker::AsyncCallbackFinalize, this,
        std::placeholders::_1));
  }

#ifndef DOCTEST_CONFIG_DISABLE
  friend NaggyMockConcurrencyWorker;
#endif
};

}}  // namespace triton::perfanalyzer
