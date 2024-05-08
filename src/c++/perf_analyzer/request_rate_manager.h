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
#pragma once

#include <condition_variable>

#include "load_manager.h"
#include "request_rate_worker.h"

namespace triton { namespace perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class TestRequestRateManager;
#endif

//==============================================================================
/// RequestRateManager is a helper class to send inference requests to
/// inference server in accordance with a Poisson distribution. This
/// distribution models the real-world traffic patterns.
///
/// An instance of this load manager will be created at the beginning of the
/// perf analyzer and it will be used to simulate load with different target
/// requests per second values and to collect per-request statistic.
///
/// Detail:
/// Request Rate Manager will try to follow a pre-computed schedule while
/// issuing requests to the server and maintain a constant request rate. The
/// manager will spawn max_threads many worker thread to meet the timeline
/// imposed by the schedule. The worker threads will record the start time and
/// end time of each request into a shared vector which will be used to report
/// the observed latencies in serving requests. Additionally, they will report a
/// vector of the number of requests missed their schedule.
///
class RequestRateManager : public LoadManager {
 public:
  ~RequestRateManager();

  /// Create an object of realistic load manager that is responsible to maintain
  /// specified load on inference server.
  /// \param async Whether to use asynchronous or synchronous API for infer
  /// request.
  /// \param streaming Whether to use gRPC streaming API for infer request
  /// \param measurement_window_ms The time window for measurements.
  /// \param max_trials The maximum number of windows that will be measured
  /// \param request_distribution The kind of distribution to use for drawing
  /// out intervals between successive requests.
  /// \param batch_size The batch size used for each request.
  /// \param max_threads The maximum number of working threads to be spawned.
  /// \param num_of_sequences The number of concurrent sequences that must be
  /// maintained on the server.
  /// \param string_length The length of the string to create for input.
  /// \param string_data The data to use for generating string input.
  /// \param zero_input Whether to fill the input tensors with zero.
  /// \param user_data The vector containing path/paths to user-provided data
  /// that can be a directory or path to a json data file.
  /// \param shared_memory_type The type of shared memory to use for inputs.
  /// \param output_shm_size The size of the shared memory to allocate for the
  /// output.
  /// \param serial_sequences Enable serial sequence mode.
  /// \param parser The ModelParser object to get the model details.
  /// \param factory The ClientBackendFactory object used to create
  /// client to the server.
  /// \param manager Returns a new ConcurrencyManager object.
  /// \param request_parameters Custom request parameters to send to the server
  /// \return cb::Error object indicating success or failure.
  static cb::Error Create(
      const bool async, const bool streaming,
      const uint64_t measurement_window_ms, const size_t max_trials,
      Distribution request_distribution, const int32_t batch_size,
      const size_t max_threads, const uint32_t num_of_sequences,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      std::unique_ptr<LoadManager>* manager,
      const std::unordered_map<std::string, cb::RequestParameter>&
          request_parameters);

  /// Adjusts the rate of issuing requests to be the same as 'request_rate'
  /// \param request_rate The rate at which requests must be issued to the
  /// server.
  /// \return cb::Error object indicating success or failure.
  cb::Error ChangeRequestRate(
      const double target_request_rate, const size_t request_count = 0);

 protected:
  RequestRateManager(
      const bool async, const bool streaming, Distribution request_distribution,
      const int32_t batch_size, const uint64_t measurement_window_ms,
      const size_t max_trials, const size_t max_threads,
      const uint32_t num_of_sequences,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::unordered_map<std::string, cb::RequestParameter>&
          request_parameters);

  void InitManagerFinalize() override;

  /// Generates and update the request schedule as per the given request rate.
  /// \param request_rate The request rate to use for new schedule.
  void GenerateSchedule(const double request_rate);

  std::vector<RateSchedulePtr_t> CreateWorkerSchedules(
      std::chrono::nanoseconds duration,
      std::function<std::chrono::nanoseconds(std::mt19937&)> distribution);

  std::vector<RateSchedulePtr_t> CreateEmptyWorkerSchedules();

  std::vector<size_t> CalculateThreadIds();

  void SetScheduleDurations(std::vector<RateSchedulePtr_t>& schedules);

  void GiveSchedulesToWorkers(
      const std::vector<RateSchedulePtr_t>& worker_schedules);

  // Pauses the worker threads
  void PauseWorkers();

  void ConfigureThreads(const size_t request_count = 0);

  // Resets the counters and resumes the worker threads
  void ResumeWorkers();

  // Makes a new worker
  virtual std::shared_ptr<IWorker> MakeWorker(
      std::shared_ptr<ThreadStat>, std::shared_ptr<ThreadConfig>);

  size_t DetermineNumThreads();

  std::vector<std::shared_ptr<ThreadConfig>> threads_config_;

  std::shared_ptr<std::chrono::nanoseconds> gen_duration_;
  Distribution request_distribution_;
  std::chrono::steady_clock::time_point start_time_;
  bool execute_;
  const size_t num_of_sequences_{0};
  const bool serial_sequences_{false};

#ifndef DOCTEST_CONFIG_DISABLE
  friend TestRequestRateManager;

 public:
  RequestRateManager() = default;
#endif
};

}}  // namespace triton::perfanalyzer
