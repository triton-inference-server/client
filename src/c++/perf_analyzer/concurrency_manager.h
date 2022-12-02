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
#pragma once

#include "concurrency_worker.h"
#include "load_manager.h"

namespace triton { namespace perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class TestConcurrencyManager;
#endif

//==============================================================================
/// ConcurrencyManager is a helper class to send inference requests to inference
/// server consistently, based on the specified setting, so that the
/// perf_analyzer can measure performance under different concurrency.
///
/// An instance of concurrency manager will be created at the beginning of the
/// perf_analyzer and it will be used to simulate different load level in
/// respect to number of concurrent infer requests and to collect per-request
/// statistic.
///
/// Detail:
/// Concurrency Manager will maintain the number of concurrent requests by
/// spawning worker threads that keep sending randomly generated requests to the
/// server. The worker threads will record the start time and end
/// time of each request into a shared vector.
///
class ConcurrencyManager : public LoadManager {
 public:
  ~ConcurrencyManager();

  /// Create a concurrency manager that is responsible to maintain specified
  /// load on inference server.
  /// \param async Whether to use asynchronous or synchronous API for infer
  /// request.
  /// \param streaming Whether to use gRPC streaming API for infer request
  /// \param batch_size The batch size used for each request.
  /// \param max_threads The maximum number of working threads to be spawned.
  /// \param max_concurrency The maximum concurrency which will be requested.
  /// \param sequence_length The base length of each sequence.
  /// \param string_length The length of the string to create for input.
  /// \param string_data The data to use for generating string input.
  /// \param zero_input Whether to fill the input tensors with zero.
  /// \param user_data The vector containing path/paths to user-provided data
  /// that can be a directory or path to a json data file.
  /// \param shared_memory_type The type of shared memory to use for inputs.
  /// \param output_shm_size The size in bytes of the shared memory to
  /// allocate for the output.
  /// \param parser The ModelParser object to get the model details.
  /// \param factory The ClientBackendFactory object used to create
  /// client to the server.
  /// \param manager Returns a new ConcurrencyManager object.
  /// \return cb::Error object indicating success or failure.
  static cb::Error Create(
      const bool async, const bool streaming, const int32_t batch_size,
      const size_t max_threads, const size_t max_concurrency,
      const size_t sequence_length, const size_t string_length,
      const std::string& string_data, const bool zero_input,
      std::vector<std::string>& user_data,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      std::unique_ptr<LoadManager>* manager);

  /// Adjusts the number of concurrent requests to be the same as
  /// 'concurrent_request_count' (by creating or pausing threads)
  /// \param concurent_request_count The number of concurrent requests.
  /// \return cb::Error object indicating success or failure.
  cb::Error ChangeConcurrencyLevel(const size_t concurrent_request_count);

 private:
  ConcurrencyManager(
      const bool async, const bool streaming, const int32_t batch_size,
      const size_t max_threads, const size_t max_concurrency,
      const size_t sequence_length, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const uint64_t start_sequence_id,
      const uint64_t sequence_id_range, const size_t string_length,
      const std::string& string_data, const bool zero_input,
      std::vector<std::string>& user_data,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory);

  // Pause all worker threads that are working on sequences
  //
  void PauseSequenceWorkers();

  // Create new threads (if necessary), and then reconfigure all worker threads
  // to handle the new concurrent request count
  //
  void ReconfigThreads(size_t concurrent_request_count);

  // Restart all worker threads that were working on sequences
  //
  void ResumeSequenceWorkers();

  // The number of worker threads with non-zero concurrencies
  size_t active_threads_;

  bool execute_;

  size_t max_concurrency_;
  std::vector<std::shared_ptr<ConcurrencyWorker::ThreadConfig>> threads_config_;

#ifndef DOCTEST_CONFIG_DISABLE
  friend TestConcurrencyManager;

 protected:
  ConcurrencyManager() = default;
#endif
};

}}  // namespace triton::perfanalyzer
