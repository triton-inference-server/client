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

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "client_backend/client_backend.h"
#include "constants.h"
#include "custom_load_manager.h"
#include "doctest.h"
#include "request_rate_manager.h"
#include "test_load_manager_base.h"

using nanoseconds = std::chrono::nanoseconds;

namespace triton { namespace perfanalyzer {

/// Class to test the CustomLoadManager
///
class TestCustomLoadManager : public TestLoadManagerBase,
                              public CustomLoadManager {
 public:
  TestCustomLoadManager() = default;

  TestCustomLoadManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false,
      bool is_decoupled_model = false)
      : TestLoadManagerBase(params, is_sequence_model, is_decoupled_model),
        CustomLoadManager(
            params.async, params.streaming, "INTERVALS_FILE", params.batch_size,
            params.measurement_window_ms, params.max_trials, params.max_threads,
            params.num_of_sequences, params.sequence_length,
            params.shared_memory_type, params.output_shm_size,
            params.start_sequence_id, params.sequence_id_range, GetParser(),
            GetFactory())
  {
    InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data);
  }

  void TestSchedule(
      std::vector<uint64_t> intervals, PerfAnalyzerParameters params)
  {
    for (auto i : intervals) {
      custom_intervals_.push_back(nanoseconds{i});
    }
    nanoseconds measurement_window_nanoseconds{params.measurement_window_ms *
                                               NANOS_PER_MILLIS};
    nanoseconds max_test_duration{measurement_window_nanoseconds *
                                  params.max_trials};
    nanoseconds expected_current_timestamp{0};
    size_t intervals_index = 0;

    PauseWorkers();
    CreateSchedules();

    // Keep calling GetNextTimestamp for the entire test_duration to make sure
    // the schedule is exactly as expected
    //
    while (expected_current_timestamp < max_test_duration) {
      for (auto worker : workers_) {
        expected_current_timestamp += custom_intervals_[intervals_index];
        auto timestamp = std::dynamic_pointer_cast<RequestRateWorker>(worker)
                             ->GetNextTimestamp();
        REQUIRE(timestamp.count() == expected_current_timestamp.count());
        intervals_index = (intervals_index + 1) % custom_intervals_.size();
      }
    }
  }


  std::string& request_intervals_file_{
      CustomLoadManager::request_intervals_file_};
  RateSchedule& custom_intervals_{CustomLoadManager::custom_intervals_};

  cb::Error ReadTimeIntervalsFile(
      const std::string& path, RateSchedule* contents) override
  {
    return cb::Error::Success;
  }
};

TEST_CASE("custom_load_schedule")
{
  PerfAnalyzerParameters params;
  params.measurement_window_ms = 1000;
  params.max_trials = 10;
  bool is_sequence = false;
  bool is_decoupled = false;
  bool use_mock_infer = false;
  std::vector<uint64_t> intervals;

  const auto& ParameterizeIntervals{[&]() {
    SUBCASE("intervals A") { intervals = {100000000, 110000000, 130000000}; }
    SUBCASE("intervals B") { intervals = {150000000}; }
    SUBCASE("intervals C")
    {
      intervals = {100000000, 110000000, 120000000, 130000000, 140000000};
    }
  }};

  const auto& ParameterizeThreads{[&]() {
    SUBCASE("threads 1")
    {
      ParameterizeIntervals();
      params.max_threads = 1;
    }
    SUBCASE("threads 2")
    {
      ParameterizeIntervals();
      params.max_threads = 2;
    }
    SUBCASE("threads 4")
    {
      ParameterizeIntervals();
      params.max_threads = 4;
    }
    SUBCASE("threads 7")
    {
      ParameterizeIntervals();
      params.max_threads = 7;
    }
  }};

  const auto& ParameterizeTrials{[&]() {
    SUBCASE("trials 3")
    {
      ParameterizeThreads();
      params.max_trials = 3;
    }
    SUBCASE("trials 10")
    {
      ParameterizeThreads();
      params.max_trials = 10;
    }
    SUBCASE("trials 20")
    {
      ParameterizeThreads();
      params.max_trials = 20;
    }
  }};

  const auto& ParameterizeMeasurementWindow{[&]() {
    SUBCASE("window 1000")
    {
      ParameterizeTrials();
      params.measurement_window_ms = 1000;
    }
    SUBCASE("window 10000")
    {
      ParameterizeTrials();
      params.measurement_window_ms = 10000;
    }
    SUBCASE("window 500")
    {
      ParameterizeTrials();
      params.measurement_window_ms = 500;
    }
  }};

  ParameterizeMeasurementWindow();
  TestCustomLoadManager tclm(params, is_sequence, is_decoupled);
  tclm.TestSchedule(intervals, params);
}

// FIXME
// TEST_CASE("testing the InitCustomIntervals function")
//{
//  TestCustomLoadManager tclm{};
//
//  SUBCASE("no file provided")
//  {
//    cb::Error result{tclm.InitCustomIntervals()};
//
//    CHECK(result.Err() == SUCCESS);
//    CHECK(tclm.schedule_.size() == 1);
//    CHECK(tclm.schedule_[0] == nanoseconds(0));
//  }
//
//  SUBCASE("file provided")
//  {
//    tclm.request_intervals_file_ = "nonexistent_file.txt";
//    tclm.gen_duration_ = std::make_unique<nanoseconds>(350000000);
//    tclm.custom_intervals_.push_back(nanoseconds(100000000));
//    tclm.custom_intervals_.push_back(nanoseconds(110000000));
//    tclm.custom_intervals_.push_back(nanoseconds(130000000));
//
//    cb::Error result{tclm.InitCustomIntervals()};
//
//    CHECK(result.Err() == SUCCESS);
//    CHECK(tclm.schedule_.size() == 5);
//    CHECK(tclm.schedule_[0] == nanoseconds(0));
//    CHECK(tclm.schedule_[1] == nanoseconds(100000000));
//    CHECK(tclm.schedule_[2] == nanoseconds(210000000));
//    CHECK(tclm.schedule_[3] == nanoseconds(340000000));
//    CHECK(tclm.schedule_[4] == nanoseconds(440000000));
//  }
//}

TEST_CASE("testing the GetCustomRequestRate function")
{
  TestCustomLoadManager tclm{};
  double request_rate{0.0};

  SUBCASE("custom_intervals_ empty")
  {
    cb::Error result{tclm.GetCustomRequestRate(&request_rate)};

    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(result.Message() == "The custom intervals vector is empty");
  }

  SUBCASE("custom_intervals_ populated")
  {
    tclm.custom_intervals_.push_back(nanoseconds(100000000));
    tclm.custom_intervals_.push_back(nanoseconds(110000000));
    tclm.custom_intervals_.push_back(nanoseconds(130000000));

    cb::Error result{tclm.GetCustomRequestRate(&request_rate)};

    CHECK(result.Err() == SUCCESS);
    CHECK(request_rate == doctest::Approx(8.0));
  }
}
}}  // namespace triton::perfanalyzer
