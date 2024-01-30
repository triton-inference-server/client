// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "request_record.h"

namespace triton { namespace perfanalyzer {

class TestReportWriter : ReportWriter {
 public:
  TestReportWriter() = default;
  TestReportWriter(const std::shared_ptr<ProfileDataCollector>& collector)
      : ReportWriter(
            "", false, std::vector<pa::PerfStatus>{}, false, false, 0, nullptr,
            false, collector, true)
  {
  }
  void WriteGPUMetrics(std::ostream& ofs, const Metrics& metrics)
  {
    ReportWriter::WriteGPUMetrics(ofs, metrics);
  }

  void WriteLLMMetrics(std::ostream& ofs)
  {
    ReportWriter::WriteLLMMetrics(ofs);
  }
};

TEST_CASE("testing WriteGPUMetrics")
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
    trw.WriteGPUMetrics(actual_output, m);
    const std::string expected_output{",a:1;,a:2.2;,a:3;,a:4;"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("single gpu missing data")
  {
    m.gpu_power_usage_per_gpu.erase("a");
    trw.WriteGPUMetrics(actual_output, m);
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
      trw.WriteGPUMetrics(actual_output, m);
      const std::string expected_output{
          ",a:1;z:100;,a:2.2;z:222.2;,a:3;z:45;,a:4;z:89;"};
      CHECK(actual_output.str() == expected_output);
    }

    SUBCASE("multi gpu missing data")
    {
      m.gpu_utilization_per_gpu.erase("z");
      m.gpu_power_usage_per_gpu.erase("a");
      trw.WriteGPUMetrics(actual_output, m);
      const std::string expected_output{",a:1;,z:222.2;,a:3;z:45;,a:4;z:89;"};
      CHECK(actual_output.str() == expected_output);
    }
  }
}

RequestRecord
GenerateRequestRecord(
    uint64_t sequence_id, uint64_t request_timestamp,
    const std::vector<uint64_t>& response_timestamps)
{
  using std::chrono::system_clock;
  using std::chrono::time_point;

  auto clock_epoch{time_point<system_clock>()};
  auto request{clock_epoch + std::chrono::microseconds(request_timestamp)};

  std::vector<time_point<system_clock>> responses;
  for (const auto& t : response_timestamps) {
    responses.push_back(clock_epoch + std::chrono::microseconds(t));
  }

  RequestRecord request_record{request, responses,   0,
                               false,   sequence_id, false};
  return request_record;
}

TEST_CASE("report_writer: WriteLLMMetrics")
{
  std::shared_ptr<ProfileDataCollector> collector;
  CHECK_NOTHROW_MESSAGE(
      pa::ProfileDataCollector::Create(&collector),
      "failed to create profile data collector");

  InferenceLoadMode infer_mode{};

  SUBCASE("request with zero response")
  {
    uint64_t sequence_id1{123};
    uint64_t request_timestamp1{1};
    std::vector<uint64_t> response_timestamps1{};
    RequestRecord rr1 = GenerateRequestRecord(
        sequence_id1, request_timestamp1, response_timestamps1);

    uint64_t sequence_id2{456};
    uint64_t request_timestamp2{2};
    std::vector<uint64_t> response_timestamps2{};
    RequestRecord rr2 = GenerateRequestRecord(
        sequence_id2, request_timestamp2, response_timestamps2);

    std::vector<RequestRecord> request_records{rr1, rr2};
    collector->AddData(infer_mode, std::move(request_records));

    // Avg first token latency = n/a
    // Avg token-to-token latency = n/a
    TestReportWriter trw(collector);
    std::ostringstream actual_output{};
    trw.WriteLLMMetrics(actual_output);
    const std::string expected_output{",n/a,n/a"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("requests with single response")
  {
    uint64_t sequence_id1{123};
    uint64_t request_timestamp1{1};
    std::vector<uint64_t> response_timestamps1{2};
    RequestRecord rr1 = GenerateRequestRecord(
        sequence_id1, request_timestamp1, response_timestamps1);

    uint64_t sequence_id2{456};
    uint64_t request_timestamp2{2};
    std::vector<uint64_t> response_timestamps2{9};
    RequestRecord rr2 = GenerateRequestRecord(
        sequence_id2, request_timestamp2, response_timestamps2);

    std::vector<RequestRecord> request_records{rr1, rr2};
    collector->AddData(infer_mode, std::move(request_records));

    // Avg first token latency
    // = ((response1[0] - request1) + (response2[0] - request2)) / 2
    // = ((2 - 1) + (9 - 2)) / 2 = 4 us
    //
    // Avg token-to-token latency = n/a
    TestReportWriter trw(collector);
    std::ostringstream actual_output{};
    trw.WriteLLMMetrics(actual_output);
    const std::string expected_output{",4,n/a"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("requests with multiple responses")
  {
    uint64_t sequence_id1{123};
    uint64_t request_timestamp1{1};
    std::vector<uint64_t> response_timestamps1{4, 5, 8, 10};
    RequestRecord rr1 = GenerateRequestRecord(
        sequence_id1, request_timestamp1, response_timestamps1);

    uint64_t sequence_id2{456};
    uint64_t request_timestamp2{2};
    std::vector<uint64_t> response_timestamps2{6, 7, 10, 12};
    RequestRecord rr2 = GenerateRequestRecord(
        sequence_id2, request_timestamp2, response_timestamps2);

    std::vector<RequestRecord> request_records{rr1, rr2};
    collector->AddData(infer_mode, std::move(request_records));

    // Avg first token latency
    // = ((response1[0] - request1) + (response2[0] - request2)) / 2
    // = ((4 - 1) + (6 - 2)) / 2 = 3.5 us
    //
    // Avg token-to-token latency
    // = ((res1[i] - res1[i - 1]) + ... + (res2[i]] - res2[i - 1]) + ...) / 6
    // = ((5-4) + (8-5) + (10-8) + (7-6) + (10-7) + (12-10)) / 6 = 2 us
    TestReportWriter trw(collector);
    std::ostringstream actual_output{};
    trw.WriteLLMMetrics(actual_output);
    const std::string expected_output{",3.5,2"};
    CHECK(actual_output.str() == expected_output);
  }
}

}}  // namespace triton::perfanalyzer
