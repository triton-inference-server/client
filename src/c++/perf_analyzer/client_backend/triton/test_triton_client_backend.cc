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

#include <cstdint>
#include <map>
#include <string>
#include "../../doctest.h"
#include "triton_client_backend.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritonremote {

class TestTritonClientBackend : public TritonClientBackend {
 public:
  template <typename T>
  void ParseAndStoreMetric(
      const std::string& metrics_endpoint_text, const std::string metric_id,
      std::map<std::string, T>& metric_per_gpu)
  {
    TritonClientBackend::ParseAndStoreMetric<T>(
        metrics_endpoint_text, metric_id, metric_per_gpu);
  }
};

TEST_CASE("testing the ParseAndStoreMetric function")
{
  TestTritonClientBackend ttcb{};

  SUBCASE("nv_gpu_utilization metric")
  {
    const std::string metrics_endpoint_text{R"(
# HELP nv_gpu_utilization GPU utilization rate [0.0 - 1.0)
# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_uuid="GPU-00000000-0000-0000-0000-000000000000"} 0.41
nv_gpu_utilization{gpu_uuid="GPU-00000000-0000-0000-0000-000000000001"} 0.77
    )"};
    const std::string metric_id{"nv_gpu_utilization"};
    std::map<std::string, double> gpu_utilization_per_gpu{};

    ttcb.ParseAndStoreMetric<double>(
        metrics_endpoint_text, metric_id, gpu_utilization_per_gpu);
    CHECK(gpu_utilization_per_gpu.size() == 2);
    CHECK(
        gpu_utilization_per_gpu["GPU-00000000-0000-0000-0000-000000000000"] ==
        doctest::Approx(0.41));
    CHECK(
        gpu_utilization_per_gpu["GPU-00000000-0000-0000-0000-000000000001"] ==
        doctest::Approx(0.77));
  }

  SUBCASE("nv_gpu_power_usage metric")
  {
    const std::string metrics_endpoint_text{R"(
# HELP nv_gpu_power_usage GPU power usage in watts
# TYPE nv_gpu_power_usage gauge
nv_gpu_power_usage{gpu_uuid="GPU-00000000-0000-0000-0000-000000000000"} 81.619
nv_gpu_power_usage{gpu_uuid="GPU-00000000-0000-0000-0000-000000000001"} 99.217
    )"};
    const std::string metric_id{"nv_gpu_power_usage"};
    std::map<std::string, double> gpu_power_usage_per_gpu{};

    ttcb.ParseAndStoreMetric<double>(
        metrics_endpoint_text, metric_id, gpu_power_usage_per_gpu);
    CHECK(gpu_power_usage_per_gpu.size() == 2);
    CHECK(
        gpu_power_usage_per_gpu["GPU-00000000-0000-0000-0000-000000000000"] ==
        doctest::Approx(81.619));
    CHECK(
        gpu_power_usage_per_gpu["GPU-00000000-0000-0000-0000-000000000001"] ==
        doctest::Approx(99.217));
  }

  SUBCASE("nv_gpu_memory_used_bytes metric")
  {
    const std::string metrics_endpoint_text{R"(
# HELP nv_gpu_memory_used_bytes GPU used memory, in bytes
# TYPE nv_gpu_memory_used_bytes gauge
nv_gpu_memory_used_bytes{gpu_uuid="GPU-00000000-0000-0000-0000-000000000000"} 50000000
nv_gpu_memory_used_bytes{gpu_uuid="GPU-00000000-0000-0000-0000-000000000001"} 75000000
    )"};
    const std::string metric_id{"nv_gpu_memory_used_bytes"};
    std::map<std::string, uint64_t> gpu_memory_used_bytes_per_gpu{};

    ttcb.ParseAndStoreMetric<uint64_t>(
        metrics_endpoint_text, metric_id, gpu_memory_used_bytes_per_gpu);
    CHECK(gpu_memory_used_bytes_per_gpu.size() == 2);
    CHECK(
        gpu_memory_used_bytes_per_gpu
            ["GPU-00000000-0000-0000-0000-000000000000"] == 50000000);
    CHECK(
        gpu_memory_used_bytes_per_gpu
            ["GPU-00000000-0000-0000-0000-000000000001"] == 75000000);
  }

  SUBCASE("nv_gpu_memory_total_bytes metric")
  {
    const std::string metrics_endpoint_text{R"(
# HELP nv_gpu_memory_total_bytes GPU total memory, in bytes
# TYPE nv_gpu_memory_total_bytes gauge
nv_gpu_memory_total_bytes{gpu_uuid="GPU-00000000-0000-0000-0000-000000000000"} 1000000000
nv_gpu_memory_total_bytes{gpu_uuid="GPU-00000000-0000-0000-0000-000000000001"} 2000000000
    )"};
    const std::string metric_id{"nv_gpu_memory_total_bytes"};
    std::map<std::string, uint64_t> gpu_memory_total_bytes_per_gpu{};

    ttcb.ParseAndStoreMetric<uint64_t>(
        metrics_endpoint_text, metric_id, gpu_memory_total_bytes_per_gpu);
    CHECK(gpu_memory_total_bytes_per_gpu.size() == 2);
    CHECK(
        gpu_memory_total_bytes_per_gpu
            ["GPU-00000000-0000-0000-0000-000000000000"] == 1000000000);
    CHECK(
        gpu_memory_total_bytes_per_gpu
            ["GPU-00000000-0000-0000-0000-000000000001"] == 2000000000);
  }
}

}}}}  // namespace triton::perfanalyzer::clientbackend::tritonremote
