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

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "data_loader.h"
#include "infer_manager.h"
#include "iworker.h"
#include "model_parser.h"

namespace triton { namespace perfanalyzer {

/// Abstract base class for worker threads
///
class LoadWorker : public IWorker {
 protected:
  LoadWorker(
      uint32_t id, std::shared_ptr<ThreadStat> thread_stat,
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<DataLoader> data_loader,
      const std::shared_ptr<cb::ClientBackendFactory> factory,
      std::vector<std::shared_ptr<SequenceStat>>& sequence_stat,
      std::unordered_map<std::string, SharedMemoryData>& shared_memory_regions,
      const cb::BackendKind backend_kind,
      const SharedMemoryType shared_memory_type, const bool on_sequence_model,
      const bool async, const bool streaming, const int32_t batch_size,
      const bool using_json_data, const size_t sequence_length,
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      std::atomic<uint64_t>& curr_seq_id,
      std::uniform_int_distribution<uint64_t>& distribution,
      std::condition_variable& wake_signal, std::mutex& wake_mutex,
      bool& execute)
      : thread_stat_(thread_stat), parser_(parser), data_loader_(data_loader),
        factory_(factory), sequence_stat_(sequence_stat),
        shared_memory_regions_(shared_memory_regions),
        backend_kind_(backend_kind), shared_memory_type_(shared_memory_type),
        on_sequence_model_(on_sequence_model), async_(async),
        streaming_(streaming), batch_size_(batch_size),
        using_json_data_(using_json_data), sequence_length_(sequence_length),
        start_sequence_id_(start_sequence_id),
        sequence_id_range_(sequence_id_range), wake_signal_(wake_signal),
        wake_mutex_(wake_mutex), execute_(execute)
  {
    infer_manager_ = std::make_unique<InferManager>(
        id, async, streaming, on_sequence_model, using_json_data, batch_size,
        backend_kind, shared_memory_type, start_sequence_id, sequence_id_range,
        sequence_length, curr_seq_id, distribution, thread_stat, sequence_stat,
        data_loader, parser, factory);
  }

  virtual ~LoadWorker() = default;

 protected:
  // FIXMETKG - clean this up? dynamic dispatch?
  void PrepAndSendInferRequest(uint32_t ctx_id, bool delayed = false)
  {
    if (on_sequence_model_) {
      uint32_t seq_stat_index = GetSeqStatIndex(ctx_id);
      infer_manager_->PrepAndSendSequenceInferRequest(
          ctx_id, seq_stat_index, delayed);
    } else {
      infer_manager_->PrepAndSendInferRequest(ctx_id, delayed);
    }
  }


  // Detect and handle the case where this thread needs to exit
  // Returns true if an exit condition was met
  bool HandleExitConditions();

  virtual uint32_t GetSeqStatIndex(uint32_t ctx_id) = 0;
  virtual void CompleteOngoingSequences() = 0;

  void WaitForOngoingRequests();

  // Code to execute at the end of the async callback function
  virtual void AsyncCallbackFinalize(uint32_t ctx_id) = 0;

  virtual size_t GetNumActiveThreads() = 0;

  std::unique_ptr<InferManager> infer_manager_;

  // TODO REFACTOR TMA-1017 is there a better way to do threading than to pass
  // the same cv/mutex into every thread by reference? Used to wake up this
  // thread if it has been put to sleep
  std::condition_variable& wake_signal_;
  std::mutex& wake_mutex_;

  // TODO REFACTOR TMA-1017 is there a better way to communicate this than a
  // shared bool reference? Used to pause execution of this thread
  bool& execute_;

  // Stats for this thread
  std::shared_ptr<ThreadStat> thread_stat_;

  // Map from shared memory key to its starting address and size
  std::unordered_map<std::string, SharedMemoryData>& shared_memory_regions_;
  // Sequence stats for all sequences
  std::vector<std::shared_ptr<SequenceStat>>& sequence_stat_;
  std::shared_ptr<DataLoader> data_loader_;
  const std::shared_ptr<ModelParser> parser_;
  const std::shared_ptr<cb::ClientBackendFactory> factory_;

  const cb::BackendKind backend_kind_;
  const SharedMemoryType shared_memory_type_;
  const bool on_sequence_model_;
  const bool async_;
  const bool streaming_;
  const int32_t batch_size_;
  const bool using_json_data_;
  const uint64_t start_sequence_id_;
  const uint64_t sequence_id_range_;
  const size_t sequence_length_;
};

}}  // namespace triton::perfanalyzer
