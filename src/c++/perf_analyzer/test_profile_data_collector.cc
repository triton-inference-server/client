// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST_CASE("profile_data_collector")
{
  MockProfileDataCollector collector{};
  InferenceLoadMode infer_mode{10, 20.0};

  std::vector<Experiment>::iterator it;
  it = collector.FindExperiment(infer_mode);
  CHECK(it == collector.experiments_.end());


  SUBCASE("FindExperiment")
  {
    std::vector<RequestRecord> request_records{RequestRecord{}};
    collector.AddData(infer_mode, std::move(request_records));

    it = collector.FindExperiment(infer_mode);
    CHECK(it != collector.experiments_.end());
    CHECK((*it).mode == infer_mode);
  }

  SUBCASE("AddData")
  {
    // Add RequestRecords
    uint64_t sequence_id_0{123};
    RequestRecord request_record_0{
        std::chrono::system_clock::now(),
        std::vector<std::chrono::time_point<std::chrono::system_clock>>{
            std::chrono::system_clock::now()},
        false,
        false,
        sequence_id_0,
        false};

    uint64_t sequence_id_1{456};
    RequestRecord request_record_1{
        std::chrono::system_clock::now(),
        std::vector<std::chrono::time_point<std::chrono::system_clock>>{
            std::chrono::system_clock::now()},
        false,
        false,
        sequence_id_1,
        false};

    std::vector<RequestRecord> request_records{
        request_record_0, request_record_1};
    collector.AddData(infer_mode, std::move(request_records));

    it = collector.FindExperiment(infer_mode);
    CHECK(it != collector.experiments_.end());

    std::vector<RequestRecord> rr{(*it).requests};
    CHECK(rr[0].sequence_id_ == sequence_id_0);
    CHECK(rr[1].sequence_id_ == sequence_id_1);
  }

  SUBCASE("AddWindow")
  {
    uint64_t window_start_ns_0{123};
    uint64_t window_end_ns_0{456};
    collector.AddWindow(infer_mode, window_start_ns_0, window_end_ns_0);

    it = collector.FindExperiment(infer_mode);
    CHECK(it != collector.experiments_.end());
    CHECK((*it).window_boundaries[0] == window_start_ns_0);
    CHECK((*it).window_boundaries[1] == window_end_ns_0);

    uint64_t window_start_ns_1{678};
    uint64_t window_end_ns_1{912};
    collector.AddWindow(infer_mode, window_start_ns_1, window_end_ns_1);

    it = collector.FindExperiment(infer_mode);
    CHECK(it != collector.experiments_.end());
    CHECK((*it).window_boundaries[2] == window_start_ns_1);
    CHECK((*it).window_boundaries[3] == window_end_ns_1);
  }
}

}}  // namespace triton::perfanalyzer
