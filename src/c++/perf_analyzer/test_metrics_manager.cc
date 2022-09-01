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

#include <iostream>
#include <sstream>
#include <streambuf>
#include "doctest.h"
#include "metrics_manager.h"

namespace triton { namespace perfanalyzer {

class TestMetricsManager : public MetricsManager {
 public:
  void CheckForMissingMetrics(const Metrics& metrics)
  {
    MetricsManager::CheckForMissingMetrics(metrics);
  }

  void CheckForMetricIntervalTooShort(
      const std::chrono::nanoseconds& remainder,
      const std::chrono::nanoseconds& duration)
  {
    MetricsManager::CheckForMetricIntervalTooShort(remainder, duration);
  }

  bool& has_given_missing_metrics_warning_()
  {
    return MetricsManager::has_given_missing_metrics_warning_;
  }

  bool& has_given_metric_interval_warning_()
  {
    return MetricsManager::has_given_metric_interval_warning_;
  }

  uint64_t& metrics_interval_ms_()
  {
    return MetricsManager::metrics_interval_ms_;
  }
};

TEST_CASE("testing the CheckForMissingMetrics function")
{
  TestMetricsManager tmm{};
  Metrics metrics{};
  std::stringstream captured_cerr;
  std::streambuf* old_cerr{std::cerr.rdbuf(captured_cerr.rdbuf())};

  SUBCASE("no metrics missing, called once, nothing printed")
  {
    metrics.gpu_utilization_per_gpu["gpu0"] = 0.5;
    metrics.gpu_power_usage_per_gpu["gpu0"] = 50.0;
    metrics.gpu_memory_used_bytes_per_gpu["gpu0"] = 1000;
    tmm.CheckForMissingMetrics(metrics);
    CHECK(captured_cerr.str() == "");
  }

  SUBCASE("no metrics missing, called twice, nothing printed")
  {
    metrics.gpu_utilization_per_gpu["gpu0"] = 0.5;
    metrics.gpu_power_usage_per_gpu["gpu0"] = 50.0;
    metrics.gpu_memory_used_bytes_per_gpu["gpu0"] = 1000;
    tmm.CheckForMissingMetrics(metrics);
    tmm.CheckForMissingMetrics(metrics);
    CHECK(captured_cerr.str() == "");
  }

  SUBCASE("all metrics missing, called once, printed once")
  {
    tmm.CheckForMissingMetrics(metrics);
    CHECK(
        captured_cerr.str() ==
        "WARNING: Unable to parse 'nv_gpu_utilization' metric.\n"
        "WARNING: Unable to parse 'nv_gpu_power_usage' metric.\n"
        "WARNING: Unable to parse 'nv_gpu_memory_used_bytes' metric.\n");
  }

  SUBCASE("all metrics missing, called twice, printed once")
  {
    tmm.CheckForMissingMetrics(metrics);
    tmm.CheckForMissingMetrics(metrics);
    CHECK(
        captured_cerr.str() ==
        "WARNING: Unable to parse 'nv_gpu_utilization' metric.\n"
        "WARNING: Unable to parse 'nv_gpu_power_usage' metric.\n"
        "WARNING: Unable to parse 'nv_gpu_memory_used_bytes' metric.\n");
  }

  std::cerr.rdbuf(old_cerr);
  tmm.has_given_missing_metrics_warning_() = false;
}

TEST_CASE("testing the CheckForMetricIntervalTooShort function")
{
  TestMetricsManager tmm{};
  tmm.metrics_interval_ms_() = 5;
  std::stringstream captured_cerr;
  std::streambuf* old_cerr{std::cerr.rdbuf(captured_cerr.rdbuf())};

  SUBCASE("valid interval, called once, nothing printed")
  {
    std::chrono::nanoseconds remainder{2000000};
    std::chrono::nanoseconds duration{3000000};
    tmm.CheckForMetricIntervalTooShort(remainder, duration);
    CHECK(captured_cerr.str() == "");
  }

  SUBCASE("valid interval, called twice, nothing printed")
  {
    std::chrono::nanoseconds remainder{2000000};
    std::chrono::nanoseconds duration{3000000};
    tmm.CheckForMetricIntervalTooShort(remainder, duration);
    tmm.CheckForMetricIntervalTooShort(remainder, duration);
    CHECK(captured_cerr.str() == "");
  }

  SUBCASE("invalid interval, called once, printed once")
  {
    std::chrono::nanoseconds remainder{-2000000};
    std::chrono::nanoseconds duration{7000000};
    tmm.CheckForMetricIntervalTooShort(remainder, duration);
    CHECK(
        captured_cerr.str() ==
        "WARNING: Triton metrics endpoint latency (7ms) is larger than the "
        "querying interval (5ms). Please try a larger querying interval via "
        "`--triton-metrics-interval`.\n");
  }

  SUBCASE("invalid interval, called twice, printed once")
  {
    std::chrono::nanoseconds remainder{-2000000};
    std::chrono::nanoseconds duration{7000000};
    tmm.CheckForMetricIntervalTooShort(remainder, duration);
    tmm.CheckForMetricIntervalTooShort(remainder, duration);
    CHECK(
        captured_cerr.str() ==
        "WARNING: Triton metrics endpoint latency (7ms) is larger than the "
        "querying interval (5ms). Please try a larger querying interval via "
        "`--triton-metrics-interval`.\n");
  }

  std::cerr.rdbuf(old_cerr);
  tmm.has_given_metric_interval_warning_() = false;
}

}}  // namespace triton::perfanalyzer
