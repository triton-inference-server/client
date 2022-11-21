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

#include <memory>
#include <queue>
#include <random>

#include "concurrency_manager.h"
#include "model_parser.h"

namespace triton { namespace perfanalyzer {

///  FIXME
class ConcurrencyWorker {
 public:
  ~ConcurrencyWorker();
  ConcurrencyWorker(ConcurrencyWorker&) = delete;

  ConcurrencyWorker(
      const std::shared_ptr<ModelParser>& parser,
      std::shared_ptr<DataLoader> data_loader, cb::BackendKind backend_kind,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const size_t sequence_length, const uint64_t start_sequence_id,
      const uint64_t sequence_id_range, const bool on_sequence_model,
      const bool async, const size_t max_concurrency,
      const bool using_json_data, const bool streaming,
      const SharedMemoryType shared_memory_type, const int32_t batch_size,
      std::vector<std::shared_ptr<ConcurrencyManager::ThreadConfig>>&
          threads_config,
      std::vector<std::shared_ptr<LoadManager::SequenceStat>>& sequence_stat,
      std::condition_variable& wake_signal, std::mutex& wake_mutex,
      size_t& active_threads, bool& execute, std::atomic<uint64_t>& curr_seq_id)
      : parser_(parser), data_loader_(data_loader), backend_kind_(backend_kind),
        factory_(factory), sequence_length_(sequence_length),
        start_sequence_id_(start_sequence_id),
        sequence_id_range_(sequence_id_range),
        on_sequence_model_(on_sequence_model), async_(async),
        max_concurrency_(max_concurrency), using_json_data_(using_json_data),
        streaming_(streaming), shared_memory_type_(shared_memory_type),
        batch_size_(batch_size), threads_config_(threads_config),
        sequence_stat_(sequence_stat), wake_signal_(wake_signal),
        wake_mutex_(wake_mutex), active_threads_(active_threads),
        execute_(execute), curr_seq_id_(curr_seq_id)
  {
  }

  // FIXME underscores. Likely should be in constructor?
  void Infer(
      std::shared_ptr<ConcurrencyManager::ThreadStat> thread_stat,
      std::shared_ptr<ConcurrencyManager::ThreadConfig> thread_config);

 private:
  // FIXME -- should come into constructor? Also, same name as function
  // variables?
  std::shared_ptr<DataLoader> data_loader_;
  cb::BackendKind backend_kind_;  // FIXME changed from unique_ptr of backend_
  const std::shared_ptr<cb::ClientBackendFactory>& factory_;
  const std::shared_ptr<ModelParser>& parser_;

  // Map from shared memory key to its starting address and size
  // FIXME -- does this need to be shared amongst all threads? Reference?
  std::unordered_map<std::string, LoadManager::SharedMemoryData>
      shared_memory_regions_;
  std::default_random_engine rng_generator_;
  std::uniform_int_distribution<uint64_t> distribution_;
  // Current sequence id (for issuing new sequences)
  std::atomic<uint64_t>&
      curr_seq_id_;  // FIXME -- this needs to be shared across?
  const uint64_t start_sequence_id_;
  const uint64_t sequence_id_range_;
  const size_t sequence_length_;
  const bool on_sequence_model_;
  const bool async_;
  const size_t max_concurrency_;
  const bool using_json_data_;
  const bool streaming_;
  const SharedMemoryType shared_memory_type_;
  const int32_t batch_size_;

  // FIXME -- these need to be a reference? Should not be member variable?
  std::vector<std::shared_ptr<ConcurrencyManager::ThreadConfig>>&
      threads_config_;
  std::vector<std::shared_ptr<LoadManager::SequenceStat>>& sequence_stat_;
  std::condition_variable& wake_signal_;
  std::mutex& wake_mutex_;
  size_t& active_threads_;
  bool& execute_;

  // FIXME - remove these functions from LoadManager

  /// Helper function to prepare the LoadManager::InferContext for sending
  /// inference request. \param ctx The target LoadManager::InferContext object.
  /// \return cb::Error object indicating success or failure.
  cb::Error PrepareInfer(LoadManager::InferContext* ctx);

  /// Helper function to prepare the LoadManager::InferContext for sending
  /// inference request in shared memory. \param ctx The target
  /// LoadManager::InferContext object. \return cb::Error object indicating
  /// success or failure.
  cb::Error PrepareSharedMemoryInfer(LoadManager::InferContext* ctx);

  /// Updates the input data to use for inference request
  /// \param inputs The vector of pointers to InferInput objects for all
  /// possible inputs, potentially including optional inputs with no provided
  /// data
  /// \param valid_inputs The vector of pointers to InferInput objects to be
  /// used for inference request.
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error UpdateInputs(
      const std::vector<cb::InferInput*>& inputs,
      std::vector<cb::InferInput*>& valid_inputs, int stream_index,
      int step_index);

  /// Updates the expected output data to use for inference request. Empty
  /// vector will be returned if there is no expected output associated to the
  /// step.
  /// \param outputs The vector of outputs to get the expected data
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \param data The vector of pointer and size of the expected outputs
  /// \return cb::Error object indicating success or failure.
  cb::Error UpdateValidationOutputs(
      const std::vector<const cb::InferRequestedOutput*>& outputs,
      int stream_index, int step_index,
      std::vector<std::vector<std::pair<const uint8_t*, size_t>>>& data);

  cb::Error ValidateOutputs(
      const LoadManager::InferContext& ctx, const cb::InferResult* result_ptr);

  void SetInferSequenceOptions(
      const uint32_t seq_stat_index,
      std::unique_ptr<cb::InferOptions>& options);
  void InitNewSequence(int seq_stat_index);
  uint64_t GetNextSeqId(int seq_stat_index);
  /// Generate random sequence length based on 'offset_ratio' and
  /// 'sequence_length_'. (1 +/- 'offset_ratio') * 'sequence_length_'
  /// \param offset_ratio The offset ratio of the generated length
  /// \return random sequence length
  size_t GetRandomSequenceLength(double offset_ratio);

  /// Helper function to update the inputs
  /// \param inputs The vector of pointers to InferInput objects for all
  /// possible inputs, potentially including optional inputs with no provided
  /// data
  /// \param valid_inputs The vector of pointers to InferInput objects to be
  /// used for inference request.
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputs(
      const std::vector<cb::InferInput*>& inputs,
      std::vector<cb::InferInput*>& valid_inputs, const int stream_index,
      const int step_index);

  /// Helper function to update the shared memory inputs
  /// \param inputs The vector of pointers to InferInput objects
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputsSharedMemory(
      const std::vector<cb::InferInput*>& inputs, const int stream_index,
      const int step_index);
};

}}  // namespace triton::perfanalyzer
