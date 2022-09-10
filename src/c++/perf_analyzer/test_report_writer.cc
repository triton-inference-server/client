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
#include "report_writer.h"

namespace triton { namespace perfanalyzer {

class TestReportWriter : ReportWriter {
 public:
  void WriteGpuMetrics(std::ostream& ofs, const Metrics& metrics)
  {
    ReportWriter::WriteGpuMetrics(ofs, metrics);
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
    const std::string expected_output{"a:1;,a:2.2;,a:3;,a:4;,"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("single gpu missing data")
  {
    m.gpu_power_usage_per_gpu.erase("a");
    trw.WriteGpuMetrics(actual_output, m);
    const std::string expected_output{"a:1;,,a:3;,a:4;,"};
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
          "a:1;z:100;,a:2.2;z:222.2;,a:3;z:45;,a:4;z:89;,"};
      CHECK(actual_output.str() == expected_output);
    }

    SUBCASE("multi gpu missing data")
    {
      m.gpu_utilization_per_gpu.erase("z");
      m.gpu_power_usage_per_gpu.erase("a");
      trw.WriteGpuMetrics(actual_output, m);
      const std::string expected_output{"a:1;,z:222.2;,a:3;z:45;,a:4;z:89;,"};
      CHECK(actual_output.str() == expected_output);
    }
  }
}

}}  // namespace triton::perfanalyzer
