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

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <vector>
#include "client_backend/client_backend.h"
#include "metrics.h"

namespace triton { namespace perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class TestMetricsManager;
#endif

class MetricsManager {
 public:
  MetricsManager(
      std::shared_ptr<clientbackend::ClientBackend> client_backend,
      uint64_t metrics_interval_ms);

  /// Ends the background thread, redundant in case StopQueryingMetrics() isn't
  /// called
  ~MetricsManager();

  /// Starts background thread that queries metrics on an interval
  void StartQueryingMetrics();

  /// Checks if background thread threw exception and propogates it if so
  void CheckQueryingStatus();

  /// Puts the latest-collected metrics from background thread into vector
  /// output parameter to be used by main thread
  void GetLatestMetrics(std::vector<Metrics>& metrics_per_timestamp);

  /// Ends the background thread
  void StopQueryingMetrics();

 private:
  void QueryMetricsEveryNMilliseconds();
  void CheckForMissingMetrics(const Metrics& metrics);
  void CheckForMetricIntervalTooShort(
      const std::chrono::nanoseconds& remainder,
      const std::chrono::nanoseconds& duration);

  std::shared_ptr<clientbackend::ClientBackend> client_backend_{nullptr};
  uint64_t metrics_interval_ms_{0};
  std::mutex metrics_mutex_{};
  std::vector<Metrics> metrics_{};
  bool should_keep_querying_{false};
  std::future<void> query_loop_future_{};
  std::mutex query_loop_mutex_{};
  std::condition_variable query_loop_cv_{};
  bool has_given_missing_metrics_warning_{false};
  bool has_given_metric_interval_warning_{false};

#ifndef DOCTEST_CONFIG_DISABLE
  friend TestMetricsManager;

 protected:
  MetricsManager() = default;
#endif
};

}}  // namespace triton::perfanalyzer
