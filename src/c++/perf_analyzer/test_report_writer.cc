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

#include <sstream>
#include "doctest.h"
#include "report_writer.h"

namespace triton { namespace perfanalyzer {

class TestReportWriter {
 public:
  static void WriteGpuMetrics(std::ostream& ofs, pa::PerfStatus& status)
  {
    ReportWriter rw{};
    rw.WriteGpuMetrics(ofs, status);
  }
};

TEST_CASE("testing WriteGpuMetrics")
{
  // Single Gpu setup
  PerfStatus s1{};
  Metrics m1{};
  m1.gpu_utilization_per_gpu.insert(std::pair<std::string, double>("a", 1.0));
  m1.gpu_power_usage_per_gpu.insert(std::pair<std::string, double>("a", 2.2));
  m1.gpu_memory_used_bytes_per_gpu.insert(
      std::pair<std::string, uint64_t>("a", 3));
  s1.metrics.push_back(m1);

  SUBCASE("single gpu complete output")
  {
    std::ostringstream actual_output{};
    TestReportWriter::WriteGpuMetrics(actual_output, s1);
    std::string expected_output{"a:1;,a:2.2;,a:3;,"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("single gpu missing data")
  {
    s1.metrics[0].gpu_power_usage_per_gpu.erase("a");
    std::ostringstream actual_output{};
    TestReportWriter::WriteGpuMetrics(actual_output, s1);
    std::string expected_output{"a:1;,,a:3;,"};
    CHECK(actual_output.str() == expected_output);
  }

  // Multi Gpu setup
  s1.metrics.clear();
  m1.gpu_utilization_per_gpu.insert(std::pair<std::string, double>("z", 100.0));
  m1.gpu_power_usage_per_gpu.insert(std::pair<std::string, double>("z", 222.2));
  m1.gpu_memory_used_bytes_per_gpu.insert(
      std::pair<std::string, uint64_t>("z", 45));
  s1.metrics.push_back(m1);

  SUBCASE("multi gpu complete output")
  {
    std::ostringstream actual_output{};
    TestReportWriter::WriteGpuMetrics(actual_output, s1);
    std::string expected_output{"a:1;z:100;,a:2.2;z:222.2;,a:3;z:45;,"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("multi gpu missing data")
  {
    s1.metrics[0].gpu_utilization_per_gpu.erase("z");
    s1.metrics[0].gpu_power_usage_per_gpu.erase("a");
    std::ostringstream actual_output{};
    TestReportWriter::WriteGpuMetrics(actual_output, s1);
    std::string expected_output{"a:1;,z:222.2;,a:3;z:45;,"};
    CHECK(actual_output.str() == expected_output);
  }
}

}}  // namespace triton::perfanalyzer
