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
#include "mock_model_parser.h"
#include "mock_report_writer.h"
#include "profile_data_collector.h"
#include "report_writer.h"
#include "request_record.h"

namespace triton { namespace perfanalyzer {

TEST_CASE("report_writer: WriteGPUMetrics")
{
  MockReportWriter mrw{};
  Metrics m{};
  m.gpu_utilization_per_gpu["a"] = 1.0;
  m.gpu_power_usage_per_gpu["a"] = 2.2;
  m.gpu_memory_used_bytes_per_gpu["a"] = 3;
  m.gpu_memory_total_bytes_per_gpu["a"] = 4;
  std::ostringstream actual_output{};

  SUBCASE("single gpu complete output")
  {
    mrw.WriteGPUMetrics(actual_output, m);
    const std::string expected_output{",a:1;,a:2.2;,a:3;,a:4;"};
    CHECK(actual_output.str() == expected_output);
  }

  SUBCASE("single gpu missing data")
  {
    m.gpu_power_usage_per_gpu.erase("a");
    mrw.WriteGPUMetrics(actual_output, m);
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
      mrw.WriteGPUMetrics(actual_output, m);
      const std::string expected_output{
          ",a:1;z:100;,a:2.2;z:222.2;,a:3;z:45;,a:4;z:89;"};
      CHECK(actual_output.str() == expected_output);
    }

    SUBCASE("multi gpu missing data")
    {
      m.gpu_utilization_per_gpu.erase("z");
      m.gpu_power_usage_per_gpu.erase("a");
      mrw.WriteGPUMetrics(actual_output, m);
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

  InferenceLoadMode infer_mode;
  std::ostringstream actual_output;
  std::string expected_output;

  SUBCASE("requests with zero response")
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

    // Avg first token latency = N/A
    // Avg token-to-token latency = N/A
    expected_output = ",N/A,N/A";
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
    // Avg token-to-token latency = N/A
    expected_output = ",4,N/A";
  }

  SUBCASE("requests with many responses")
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
    expected_output = ",3.5,2";
  }

  SUBCASE("requests with mixture of responses")
  {
    // zero response
    uint64_t sequence_id1{123};
    uint64_t request_timestamp1{1};
    std::vector<uint64_t> response_timestamps1{};
    RequestRecord rr1 = GenerateRequestRecord(
        sequence_id1, request_timestamp1, response_timestamps1);

    // single response
    uint64_t sequence_id2{456};
    uint64_t request_timestamp2{2};
    std::vector<uint64_t> response_timestamps2{8};
    RequestRecord rr2 = GenerateRequestRecord(
        sequence_id2, request_timestamp2, response_timestamps2);

    // many responses
    uint64_t sequence_id3{456};
    uint64_t request_timestamp3{4};
    std::vector<uint64_t> response_timestamps3{6, 7, 10, 12};
    RequestRecord rr3 = GenerateRequestRecord(
        sequence_id3, request_timestamp3, response_timestamps3);

    std::vector<RequestRecord> request_records{rr1, rr2, rr3};
    collector->AddData(infer_mode, std::move(request_records));

    // Avg first token latency
    // = ((response2[0] - request2) + (response3[0] - request3)) / 2
    // = ((8 - 2) + (6 - 4)) / 2 = 4 us
    //
    // Avg token-to-token latency
    // = (... + (response3[i] - response3[i - 1]) + ...) / 3
    // = ((7 - 6) + (10 - 7) + (12 - 10)) / 3 = 2 us
    expected_output = ",4,2";
  }

  PerfStatus status;
  status.concurrency = infer_mode.concurrency;
  status.request_rate = infer_mode.request_rate;

  MockReportWriter mrw;
  mrw.collector_ = collector;
  mrw.WriteLLMMetrics(actual_output, status);
  CHECK(actual_output.str() == expected_output);
}

TEST_CASE("report_writer: GenerateReport")
{
  std::string filename{"temp.csv"};
  std::vector<PerfStatus> summary;
  std::shared_ptr<ModelParser> mmp{nullptr};
  std::shared_ptr<ProfileDataCollector> collector{nullptr};
  CHECK_NOTHROW_MESSAGE(
      pa::ProfileDataCollector::Create(&collector),
      "failed to create profile data collector");

  // default parameters
  bool target_concurrency{true};
  bool verbose_csv{false};
  bool include_server_stats{false};
  int32_t percentile{90};
  bool should_output_metrics{false};
  bool should_output_llm_metrics{false};
  bool is_sequence_model{false};
  bool is_decoupled_model{false};

  std::ostringstream actual_output;
  std::string expected_output;

  SUBCASE("single experiment")
  {
    mmp = std::make_shared<MockModelParser>(
        is_sequence_model, is_decoupled_model);

    ClientSideStats css;
    css.infer_per_sec = 150.123;
    css.avg_send_time_ns = 2000;
    css.avg_receive_time_ns = 3000;

    PerfStatus ps;
    ps.concurrency = 10;
    ps.client_stats = css;

    summary.push_back(ps);

    expected_output =
        "Concurrency,Inferences/Second,Client Send,Client "
        "Recv\n"
        "10,150.123,2,3\n";
  }

  SUBCASE("multiple LLM experiments")
  {
    // set parameters
    is_decoupled_model = true;
    should_output_llm_metrics = true;

    mmp = std::make_shared<MockModelParser>(
        is_sequence_model, is_decoupled_model);

    // first experiment
    ClientSideStats css1;
    css1.infer_per_sec = 150;
    css1.responses_per_sec = 123.456;
    css1.avg_send_time_ns = 2000;
    css1.avg_receive_time_ns = 3000;

    PerfStatus ps1;
    ps1.concurrency = 10;
    ps1.client_stats = css1;

    summary.push_back(ps1);

    InferenceLoadMode infer_mode1{ps1.concurrency, ps1.request_rate};
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

    std::vector<RequestRecord> request_records1{rr1, rr2};
    collector->AddData(infer_mode1, std::move(request_records1));

    // second experiment
    ClientSideStats css2;
    css2.infer_per_sec = 345.12;
    css2.responses_per_sec = 10.789;
    css2.avg_send_time_ns = 4000;
    css2.avg_receive_time_ns = 5000;

    PerfStatus ps2;
    ps2.concurrency = 30;
    ps2.client_stats = css2;

    summary.push_back(ps2);

    InferenceLoadMode infer_mode2{ps2.concurrency, ps2.request_rate};
    uint64_t sequence_id3{123};
    uint64_t request_timestamp3{1};
    std::vector<uint64_t> response_timestamps3{5, 8, 9, 11};
    RequestRecord rr3 = GenerateRequestRecord(
        sequence_id3, request_timestamp3, response_timestamps3);

    uint64_t sequence_id4{456};
    uint64_t request_timestamp4{2};
    std::vector<uint64_t> response_timestamps4{10, 15, 19, 22};
    RequestRecord rr4 = GenerateRequestRecord(
        sequence_id4, request_timestamp4, response_timestamps4);

    std::vector<RequestRecord> request_records2{rr3, rr4};
    collector->AddData(infer_mode2, std::move(request_records2));

    expected_output =
        "Concurrency,Inferences/Second,Response Throughput,Client Send,Client "
        "Recv,Avg First Token Latency,Avg Token-to-Token Latency\n"
        "10,150,123.456,2,3,3.5,2\n"
        "30,345.12,10.789,4,5,6,3\n";
  }

  MockReportWriter mrw{
      filename,
      summary,
      mmp,
      collector,
      target_concurrency,
      verbose_csv,
      include_server_stats,
      percentile,
      should_output_metrics,
      should_output_llm_metrics,
  };
  mrw.GenerateReport();

  // read from temp.csv
  std::ifstream input_file(filename);
  std::string line;
  while (std::getline(input_file, line)) {
    actual_output << line << "\n";
  }
  input_file.close();

  CHECK(actual_output.str() == expected_output);

  // clean up
  std::remove(filename.c_str());
}

}}  // namespace triton::perfanalyzer
