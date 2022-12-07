// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "load_worker.h"
#include "model_parser.h"

namespace triton { namespace perfanalyzer {

/// Interface for RequestRateWorker
///
class IRequestRateWorker {
 public:
  struct ThreadConfig {
    ThreadConfig(uint32_t index, uint32_t stride)
        : index_(index), id_(index), stride_(stride), is_paused_(false),
          rounds_(0), non_sequence_data_step_id_(index)
    {
    }

    uint32_t index_;
    uint32_t id_;
    uint32_t stride_;
    bool is_paused_;
    uint64_t rounds_;
    int non_sequence_data_step_id_;
  };

  virtual void Infer(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config) = 0;
};

/// Worker thread for RequestRateManager
///
/// If the model is non-sequence model, each worker uses only one context
/// to maintain concurrency assigned to worker.
/// If the model is sequence model, each worker has to use multiples contexts
/// to maintain (sequence) concurrency assigned to worker.
///
class RequestRateWorker : public LoadWorker, public IRequestRateWorker {
 public:
  RequestRateWorker(
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<DataLoader> data_loader, cb::BackendKind backend_kind,
      const std::shared_ptr<cb::ClientBackendFactory> factory,
      const size_t sequence_length, const uint64_t start_sequence_id,
      const uint64_t sequence_id_range, const bool on_sequence_model,
      const bool async, const size_t max_threads, const bool using_json_data,
      const bool streaming, const SharedMemoryType shared_memory_type,
      const int32_t batch_size,
      std::vector<std::shared_ptr<SequenceStat>>& sequence_stat,
      std::unordered_map<std::string, SharedMemoryData>& shared_memory_regions,
      std::condition_variable& wake_signal, std::mutex& wake_mutex,
      bool& execute, std::atomic<uint64_t>& curr_seq_id,
      std::chrono::steady_clock::time_point& start_time,
      std::vector<std::chrono::nanoseconds>& schedule,
      std::shared_ptr<std::chrono::nanoseconds> gen_duration,
      std::uniform_int_distribution<uint64_t>& distribution)
      : LoadWorker(
            parser, data_loader, factory, sequence_stat, shared_memory_regions,
            backend_kind, shared_memory_type, on_sequence_model, async,
            streaming, batch_size, using_json_data, sequence_length,
            start_sequence_id, sequence_id_range, curr_seq_id, distribution,
            wake_signal, wake_mutex, execute),
        max_threads_(max_threads), start_time_(start_time), schedule_(schedule),
        gen_duration_(gen_duration)
  {
  }

  void Infer(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config) override;

 private:
  const size_t max_threads_;
  std::chrono::steady_clock::time_point& start_time_;
  // TODO REFACTOR - why can't we just pass every thread its own personal
  // schedule instead of passing in the full schedule and making each thread
  // self-calculate where it should be?
  std::vector<std::chrono::nanoseconds>& schedule_;
  std::shared_ptr<std::chrono::nanoseconds> gen_duration_;

  /// A helper function to issue inference request to the server.
  /// \param context InferContext to use for sending the request.
  /// \param request_id The unique id to be associated with the request.
  /// \param delayed Whether the request fell behind its scheduled time.
  /// \param callback_func The callback function to use with asynchronous
  /// request.
  /// \param async_req_map The map from ongoing request_id to the
  /// request information needed to correctly interpret the details.
  /// \param thread_stat The runnning status of the worker thread
  void Request(
      std::shared_ptr<InferContext> context, const uint64_t request_id,
      const bool delayed, cb::OnCompleteFn callback_func,
      std::shared_ptr<std::map<std::string, AsyncRequestProperties>>
          async_req_map,
      std::shared_ptr<ThreadStat> thread_stat);
};


}}  // namespace triton::perfanalyzer
