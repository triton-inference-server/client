// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gmock/gmock.h"
#include "inference_profiler.h"

namespace triton { namespace perfanalyzer {

class NaggyMockInferenceProfiler : public InferenceProfiler {
 public:
  NaggyMockInferenceProfiler()
  {
    ON_CALL(
        *this, ValidLatencyMeasurement(
                   testing::_, testing::_, testing::_, testing::_, testing::_,
                   testing::_))
        .WillByDefault(
            [this](
                const std::pair<uint64_t, uint64_t>& valid_range,
                size_t& valid_sequence_count, size_t& delayed_request_count,
                std::vector<uint64_t>* latencies, size_t& response_count,
                std::vector<RequestRecord>& valid_requests) -> void {
              this->InferenceProfiler::ValidLatencyMeasurement(
                  valid_range, valid_sequence_count, delayed_request_count,
                  latencies, response_count, valid_requests);
            });
    ON_CALL(*this, SummarizeLatency(testing::_, testing::_))
        .WillByDefault(
            [this](
                const std::vector<uint64_t>& latencies,
                PerfStatus& summary) -> cb::Error {
              return this->InferenceProfiler::SummarizeLatency(
                  latencies, summary);
            });
    ON_CALL(*this, MergePerfStatusReports(testing::_, testing::_))
        .WillByDefault(
            [this](
                std::deque<PerfStatus>& perf_status,
                PerfStatus& summary_status) -> cb::Error {
              return this->InferenceProfiler::MergePerfStatusReports(
                  perf_status, summary_status);
            });
    ON_CALL(*this, MergeServerSideStats(testing::_, testing::_))
        .WillByDefault(
            [this](
                std::vector<ServerSideStats>& server_side_stats,
                ServerSideStats& server_side_summary) -> cb::Error {
              return this->InferenceProfiler::MergeServerSideStats(
                  server_side_stats, server_side_summary);
            });
    ON_CALL(
        *this, SummarizeClientStat(
                   testing::_, testing::_, testing::_, testing::_, testing::_,
                   testing::_, testing::_, testing::_))
        .WillByDefault(
            [this](
                const cb::InferStat& start_stat, const cb::InferStat& end_stat,
                const uint64_t duration_ns, const size_t valid_request_count,
                const size_t delayed_request_count,
                const size_t valid_sequence_count, const size_t response_count,
                PerfStatus& summary) -> cb::Error {
              return this->InferenceProfiler::SummarizeClientStat(
                  start_stat, end_stat, duration_ns, valid_request_count,
                  delayed_request_count, valid_sequence_count, response_count,
                  summary);
            });
  };

  MOCK_METHOD0(IncludeServerStats, bool());
  MOCK_METHOD(
      void, ValidLatencyMeasurement,
      ((const std::pair<uint64_t, uint64_t>&), size_t&, size_t&,
       std::vector<uint64_t>*, size_t&, std::vector<RequestRecord>&),
      (override));
  MOCK_METHOD(
      cb::Error, SummarizeLatency, (const std::vector<uint64_t>&, PerfStatus&),
      (override));
  MOCK_METHOD(
      cb::Error, MergePerfStatusReports, (std::deque<PerfStatus>&, PerfStatus&),
      (override));
  MOCK_METHOD(
      cb::Error, MergeServerSideStats,
      (std::vector<ServerSideStats>&, ServerSideStats&), (override));
  MOCK_METHOD(
      cb::Error, SummarizeClientStat,
      (const cb::InferStat&, const cb::InferStat&, const uint64_t, const size_t,
       const size_t, const size_t, const size_t, PerfStatus&),
      (override));

  std::shared_ptr<ModelParser>& parser_{InferenceProfiler::parser_};
  std::unique_ptr<LoadManager>& manager_{InferenceProfiler::manager_};
  bool& include_lib_stats_{InferenceProfiler::include_lib_stats_};
  std::vector<RequestRecord>& all_request_records_{
      InferenceProfiler::all_request_records_};
};

using MockInferenceProfiler = testing::NiceMock<NaggyMockInferenceProfiler>;

}}  // namespace triton::perfanalyzer
