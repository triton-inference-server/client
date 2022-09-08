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

  uint64_t& metrics_interval_ms_{MetricsManager::metrics_interval_ms_};
};

TEST_CASE("testing the CheckForMissingMetrics function")
{
  TestMetricsManager tmm{};
  Metrics metrics{};
  std::stringstream captured_cerr;
  std::streambuf* old_cerr{std::cerr.rdbuf(captured_cerr.rdbuf())};

  // check that no warning gets printed when all metrics are present
  metrics.gpu_utilization_per_gpu["gpu0"] = 0.5;
  metrics.gpu_power_usage_per_gpu["gpu0"] = 50.0;
  metrics.gpu_memory_used_bytes_per_gpu["gpu0"] = 1000;
  metrics.gpu_memory_total_bytes_per_gpu["gpu0"] = 10000;
  tmm.CheckForMissingMetrics(metrics);
  CHECK(captured_cerr.str() == "");

  // check that still no warning gets printed on a subsequent call
  tmm.CheckForMissingMetrics(metrics);
  CHECK(captured_cerr.str() == "");

  // check that warning gets printed when missing metrics
  metrics.gpu_utilization_per_gpu.clear();
  metrics.gpu_power_usage_per_gpu.clear();
  metrics.gpu_memory_used_bytes_per_gpu.clear();
  metrics.gpu_memory_total_bytes_per_gpu.clear();
  tmm.CheckForMissingMetrics(metrics);
  CHECK(
      captured_cerr.str() ==
      "WARNING: Unable to parse 'nv_gpu_utilization' metric.\n"
      "WARNING: Unable to parse 'nv_gpu_power_usage' metric.\n"
      "WARNING: Unable to parse 'nv_gpu_memory_used_bytes' metric.\n"
      "WARNING: Unable to parse 'nv_gpu_memory_total_bytes' metric.\n");

  // check that no additional warning gets printed on a subsequent call
  tmm.CheckForMissingMetrics(metrics);
  CHECK(
      captured_cerr.str() ==
      "WARNING: Unable to parse 'nv_gpu_utilization' metric.\n"
      "WARNING: Unable to parse 'nv_gpu_power_usage' metric.\n"
      "WARNING: Unable to parse 'nv_gpu_memory_used_bytes' metric.\n"
      "WARNING: Unable to parse 'nv_gpu_memory_total_bytes' metric.\n");

  std::cerr.rdbuf(old_cerr);
}

TEST_CASE("testing the CheckForMetricIntervalTooShort function")
{
  TestMetricsManager tmm{};
  tmm.metrics_interval_ms_ = 5;
  std::chrono::nanoseconds remainder{};
  std::chrono::nanoseconds duration{};
  std::stringstream captured_cerr;
  std::streambuf* old_cerr{std::cerr.rdbuf(captured_cerr.rdbuf())};

  // check that no warning gets printed when interval is long enough
  remainder = std::chrono::nanoseconds(2000000);
  duration = std::chrono::nanoseconds(3000000);
  tmm.CheckForMetricIntervalTooShort(remainder, duration);
  CHECK(captured_cerr.str() == "");

  // check that still no warning gets printed on a subsequent call
  tmm.CheckForMetricIntervalTooShort(remainder, duration);
  CHECK(captured_cerr.str() == "");

  // check that warning gets printed when interval is too short
  remainder = std::chrono::nanoseconds(-2000000);
  duration = std::chrono::nanoseconds(7000000);
  tmm.CheckForMetricIntervalTooShort(remainder, duration);
  CHECK(
      captured_cerr.str() ==
      "WARNING: Triton metrics endpoint latency (7ms) is larger than the "
      "querying interval (5ms). Please try a larger querying interval via "
      "`--triton-metrics-interval`.\n");

  // check that no additional warning gets printed on a subsequent call
  tmm.CheckForMetricIntervalTooShort(remainder, duration);
  CHECK(
      captured_cerr.str() ==
      "WARNING: Triton metrics endpoint latency (7ms) is larger than the "
      "querying interval (5ms). Please try a larger querying interval via "
      "`--triton-metrics-interval`.\n");

  std::cerr.rdbuf(old_cerr);
}

}}  // namespace triton::perfanalyzer
