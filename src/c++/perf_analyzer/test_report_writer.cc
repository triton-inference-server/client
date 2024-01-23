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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS"" AND ANY
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

#include <string>

#include "doctest.h"
#include "profile_data_collector.h"
#include "report_writer.h"

namespace triton { namespace perfanalyzer {

class TestReportWriter : ReportWriter {
 public:
  TestReportWriter() = default;
  TestReportWriter(std::vector<Experiment>& experiments)
      : ReportWriter(
            "", false, std::vector<pa::PerfStatus>{}, false, false, 0, nullptr,
            false, experiments, true)
  {
  }
  void WriteGpuMetrics(std::ostream& ofs, const Metrics& metrics)
  {
    ReportWriter::WriteGpuMetrics(ofs, metrics);
  }

  void WriteLlmMetrics(std::ostream& ofs)
  {
    ReportWriter::WriteLlmMetrics(ofs);
  }
};

TEST_CASE("testing WriteGpuMetrics")
{
  TestReportWriter trw{};
  Metrics m{};
  m.gpu_utilization_per_gpu["a"] = 1.0;
  m.gpu_power_usage_per_gpu["a"] = 2.2;
  m.gpu_memory_used_bytes_per_gpu["a"] = 3;
  m.gpu_memory_total_bytes_per_gpu["a"] = 4;
  std::ostringstream actual_output{};

  SUBCASE("single gpu complete output")
  {
    trw.WriteGpuMetrics(actual_output, m);
    const std::string expected_output{",a:1;,a:2.2;,a:3;,a:4;"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("single gpu missing data")
  {
    m.gpu_power_usage_per_gpu.erase("a");
    trw.WriteGpuMetrics(actual_output, m);
    const std::string expected_output{",a:1;,,a:3;,a:4;"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("multi-gpu")
  {
    m.gpu_utilization_per_gpu["z"] = 100.0;
    m.gpu_power_usage_per_gpu["z"] = 222.2;
    m.gpu_memory_used_bytes_per_gpu["z"] = 45;
    m.gpu_memory_total_bytes_per_gpu["z"] = 89;

    SUBCASE("multi gpu complete output")
    {
      trw.WriteGpuMetrics(actual_output, m);
      const std::string expected_output{
          ",a:1;z:100;,a:2.2;z:222.2;,a:3;z:45;,a:4;z:89;"};
      CHECK(actual_output.str() == expected_output);
    }

    SUBCASE("multi gpu missing data")
    {
      m.gpu_utilization_per_gpu.erase("z");
      m.gpu_power_usage_per_gpu.erase("a");
      trw.WriteGpuMetrics(actual_output, m);
      const std::string expected_output{",a:1;,z:222.2;,a:3;z:45;,a:4;z:89;"};
      CHECK(actual_output.str() == expected_output);
    }
  }
}

TEST_CASE("report_writer: WriteLlmMetrics")
{
  // Create a dummy request records
  using std::chrono::system_clock;
  using std::chrono::time_point;

  Experiment experiment;
  auto clock_epoch{time_point<system_clock>()};

  uint64_t seq_id1{123};
  auto request1{clock_epoch + std::chrono::microseconds(1)};
  auto response1{clock_epoch + std::chrono::microseconds(4)};
  auto response2{clock_epoch + std::chrono::microseconds(5)};

  RequestRecord rr1{
      request1, std::vector<time_point<system_clock>>{response1, response2},
      0,        false,
      seq_id1,  false};

  uint64_t seq_id2{456};
  auto request2{clock_epoch + std::chrono::microseconds(4)};
  auto response3{clock_epoch + std::chrono::microseconds(5)};
  auto response4{clock_epoch + std::chrono::microseconds(7)};

  RequestRecord rr2{
      request2, std::vector<time_point<system_clock>>{response3, response4},
      0,        false,
      seq_id2,  false};

  std::vector<RequestRecord> request_records{rr1, rr2};
  experiment.requests = std::move(request_records);
  std::vector<Experiment> experiments{experiment};

  // Avg first token latency
  // = ((request1 - response1) + (request2 - response3)) / 2
  // = (3 + 1) / 2 = 2 us
  // Avg token-to-token latency
  // = ((response2 - response1) + (response4 - response3)) / 2
  // = (1 + 2) / 2 = 1.5 us
  TestReportWriter trw(experiments);
  std::ostringstream actual_output{};
  trw.WriteLlmMetrics(actual_output);
  const std::string expected_output{",2,1.5"};
  CHECK(actual_output.str() == expected_output);
}

}}  // namespace triton::perfanalyzer
