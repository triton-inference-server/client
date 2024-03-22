// Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "doctest.h"
#include "mock_profile_data_collector.h"
#include "profile_data_collector.h"

namespace triton { namespace perfanalyzer {

TEST_CASE("profile_data_collector: FindExperiment")
{
  MockProfileDataCollector collector{};
  InferenceLoadMode infer_mode1{10, 20.0};

  std::vector<Experiment>::iterator it;
  it = collector.FindExperiment(infer_mode1);
  CHECK(it == collector.experiments_.end());

  std::vector<RequestRecord> request_records{RequestRecord{}};
  collector.AddData(infer_mode1, std::move(request_records));

  it = collector.FindExperiment(infer_mode1);
  CHECK(it != collector.experiments_.end());
  CHECK((*it).mode == infer_mode1);

  InferenceLoadMode infer_mode2{123, 0.0};
  it = collector.FindExperiment(infer_mode2);
  CHECK(it == collector.experiments_.end());
}

TEST_CASE("profile_data_collector: AddData")
{
  MockProfileDataCollector collector{};
  InferenceLoadMode infer_mode{10, 20.0};

  // Add RequestRecords
  auto clock_epoch{std::chrono::time_point<std::chrono::system_clock>()};

  uint64_t sequence_id1{123};
  auto request1_timestamp{clock_epoch + std::chrono::nanoseconds(1)};
  auto request1_response1_timestamp{clock_epoch + std::chrono::nanoseconds(2)};
  auto request1_response2_timestamp{clock_epoch + std::chrono::nanoseconds(3)};
  uint8_t fake_data[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  RequestRecord::ResponseOutput request1_response1_output{
      {"key1", RecordData(fake_data, 1)}, {"key2", RecordData(fake_data, 2)}};
  RequestRecord::ResponseOutput request1_response2_output{
      {"key3", RecordData(fake_data, 3)}, {"key4", RecordData(fake_data, 4)}};

  RequestRecord request_record1{
      request1_timestamp,
      std::vector<std::chrono::time_point<std::chrono::system_clock>>{
          request1_response1_timestamp, request1_response2_timestamp},
      {},
      {request1_response1_output, request1_response2_output},
      0,
      false,
      sequence_id1,
      false};

  uint64_t sequence_id2{456};
  auto request2_timestamp{clock_epoch + std::chrono::nanoseconds(4)};
  auto request2_response1_timestamp{clock_epoch + std::chrono::nanoseconds(5)};
  auto request2_response2_timestamp{clock_epoch + std::chrono::nanoseconds(6)};
  RequestRecord::ResponseOutput request2_response1_output{
      {"key5", RecordData(fake_data, 5)}, {"key6", RecordData(fake_data, 6)}};
  RequestRecord::ResponseOutput request2_response2_output{
      {"key7", RecordData(fake_data, 7)}, {"key8", RecordData(fake_data, 8)}};

  RequestRecord request_record2{
      request2_timestamp,
      std::vector<std::chrono::time_point<std::chrono::system_clock>>{
          request2_response1_timestamp, request2_response2_timestamp},
      {},
      {request2_response1_output, request2_response2_output},
      0,
      false,
      sequence_id2,
      false};

  std::vector<RequestRecord> request_records{request_record1, request_record2};
  collector.AddData(infer_mode, std::move(request_records));

  CHECK(!collector.experiments_.empty());

  std::vector<RequestRecord> rr{collector.experiments_[0].requests};
  CHECK(rr[0].sequence_id_ == sequence_id1);
  CHECK(rr[0].start_time_ == request1_timestamp);
  CHECK(rr[0].response_timestamps_[0] == request1_response1_timestamp);
  CHECK(rr[0].response_timestamps_[1] == request1_response2_timestamp);
  CHECK(rr[0].response_outputs_[0] == request1_response1_output);
  CHECK(rr[0].response_outputs_[1] == request1_response2_output);
  CHECK(rr[1].sequence_id_ == sequence_id2);
  CHECK(rr[1].start_time_ == request2_timestamp);
  CHECK(rr[1].response_timestamps_[0] == request2_response1_timestamp);
  CHECK(rr[1].response_timestamps_[1] == request2_response2_timestamp);
  CHECK(rr[1].response_outputs_[0] == request2_response1_output);
  CHECK(rr[1].response_outputs_[1] == request2_response2_output);
}

TEST_CASE("profile_data_collector: AddWindow")
{
  MockProfileDataCollector collector{};
  InferenceLoadMode infer_mode{10, 20.0};

  uint64_t window_start1{123};
  uint64_t window_end1{456};
  collector.AddWindow(infer_mode, window_start1, window_end1);

  CHECK(!collector.experiments_.empty());
  CHECK(collector.experiments_[0].window_boundaries[0] == window_start1);
  CHECK(collector.experiments_[0].window_boundaries[1] == window_end1);

  uint64_t window_start2{678};
  uint64_t window_end2{912};
  collector.AddWindow(infer_mode, window_start2, window_end2);

  CHECK(collector.experiments_[0].window_boundaries[2] == window_start2);
  CHECK(collector.experiments_[0].window_boundaries[3] == window_end2);
}

}}  // namespace triton::perfanalyzer
