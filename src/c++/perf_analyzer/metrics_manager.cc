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

#include "metrics_manager.h"
#include <iostream>
#include <stdexcept>
#include <utility>
#include "constants.h"
#include "perf_analyzer_exception.h"

namespace triton { namespace perfanalyzer {

MetricsManager::MetricsManager(
    std::shared_ptr<clientbackend::ClientBackend> client_backend,
    uint64_t metrics_interval_ms)
    : client_backend_(client_backend), metrics_interval_ms_(metrics_interval_ms)
{
}

MetricsManager::~MetricsManager()
{
  if (query_loop_future_.valid()) {
    StopQueryingMetrics();
  }
}

void
MetricsManager::StartQueryingMetrics()
{
  should_keep_querying_ = true;
  query_loop_future_ =
      std::async(&MetricsManager::QueryMetricsEveryNMilliseconds, this);
}

void
MetricsManager::QueryMetricsEveryNMilliseconds()
{
  while (should_keep_querying_) {
    const auto& start{std::chrono::system_clock::now()};

    Metrics metrics{};
    clientbackend::Error err{client_backend_->Metrics(metrics)};
    if (err.IsOk() == false) {
      throw PerfAnalyzerException(err.Message(), err.Err());
    }

    CheckForMissingMetrics(metrics);

    {
      std::lock_guard<std::mutex> metrics_lock{metrics_mutex_};
      metrics_.push_back(std::move(metrics));
    }

    const auto& end{std::chrono::system_clock::now()};
    const auto& duration{end - start};
    const auto& remainder{std::chrono::milliseconds(metrics_interval_ms_) -
                          duration};

    CheckForMetricIntervalTooShort(remainder, duration);

    {
      std::unique_lock<std::mutex> query_loop_lock{query_loop_mutex_};
      query_loop_cv_.wait_for(query_loop_lock, remainder);
    }
  }
}

void
MetricsManager::CheckForMissingMetrics(const Metrics& metrics)
{
  if (has_given_missing_metrics_warning_) {
    return;
  }
  if (metrics.gpu_utilization_per_gpu.empty()) {
    std::cerr << "WARNING: Unable to parse 'nv_gpu_utilization' metric."
              << std::endl;
    has_given_missing_metrics_warning_ = true;
  }
  if (metrics.gpu_power_usage_per_gpu.empty()) {
    std::cerr << "WARNING: Unable to parse 'nv_gpu_power_usage' metric."
              << std::endl;
    has_given_missing_metrics_warning_ = true;
  }
  if (metrics.gpu_memory_used_bytes_per_gpu.empty()) {
    std::cerr << "WARNING: Unable to parse 'nv_gpu_memory_used_bytes' metric."
              << std::endl;
    has_given_missing_metrics_warning_ = true;
  }
  if (metrics.gpu_memory_total_bytes_per_gpu.empty()) {
    std::cerr << "WARNING: Unable to parse 'nv_gpu_memory_total_bytes' metric."
              << std::endl;
    has_given_missing_metrics_warning_ = true;
  }
}

void
MetricsManager::CheckForMetricIntervalTooShort(
    const std::chrono::nanoseconds& remainder,
    const std::chrono::nanoseconds& duration)
{
  if (has_given_metric_interval_warning_) {
    return;
  }
  if (remainder < std::chrono::nanoseconds::zero()) {
    std::cerr << "WARNING: Triton metrics endpoint latency ("
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                     .count()
              << "ms) is larger than the querying interval ("
              << metrics_interval_ms_
              << "ms). Please try a larger querying interval "
                 "via `--triton-metrics-interval`."
              << std::endl;
    has_given_metric_interval_warning_ = true;
  }
}

void
MetricsManager::CheckQueryingStatus()
{
  if (query_loop_future_.valid() &&
      query_loop_future_.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready) {
    query_loop_future_.get();
  }
}

void
MetricsManager::GetLatestMetrics(std::vector<Metrics>& metrics)
{
  if (metrics.empty() == false) {
    throw PerfAnalyzerException(
        "MetricsManager::GetLatestMetrics() must be passed an empty vector.",
        GENERIC_ERROR);
  }
  std::lock_guard<std::mutex> metrics_lock{metrics_mutex_};
  metrics_.swap(metrics);
}

void
MetricsManager::StopQueryingMetrics()
{
  should_keep_querying_ = false;
  query_loop_cv_.notify_one();
  if (query_loop_future_.valid()) {
    query_loop_future_.get();
  }
}

}}  // namespace triton::perfanalyzer
