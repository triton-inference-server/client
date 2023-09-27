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

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "client_backend/client_backend.h"
#include "constants.h"
#include "custom_load_manager.h"
#include "doctest.h"
#include "mock_request_rate_worker.h"
#include "request_rate_manager.h"
#include "test_load_manager_base.h"

using nanoseconds = std::chrono::nanoseconds;
using milliseconds = std::chrono::milliseconds;

namespace triton { namespace perfanalyzer {

/// Class to test the CustomLoadManager
///
class TestCustomLoadManager : public TestLoadManagerBase,
                              public CustomLoadManager {
 public:
  TestCustomLoadManager() = default;

  TestCustomLoadManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false,
      bool is_decoupled_model = false, bool use_mock_infer = false)
      : use_mock_infer_(use_mock_infer),
        TestLoadManagerBase(params, is_sequence_model, is_decoupled_model),
        CustomLoadManager(
            params.async, params.streaming, "INTERVALS_FILE", params.batch_size,
            params.measurement_window_ms, params.max_trials, params.max_threads,
            params.num_of_sequences, params.shared_memory_type,
            params.output_shm_size, params.serial_sequences, GetParser(),
            GetFactory(), params.request_parameters)
  {
    InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data, params.start_sequence_id, params.sequence_id_range,
        params.sequence_length, params.sequence_length_specified,
        params.sequence_length_variation);
  }

  std::shared_ptr<IWorker> MakeWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config) override
  {
    size_t id = workers_.size();
    auto worker = std::make_shared<MockRequestRateWorker>(
        id, thread_stat, thread_config, parser_, data_loader_, factory_,
        on_sequence_model_, async_, max_threads_, using_json_data_, streaming_,
        batch_size_, wake_signal_, wake_mutex_, execute_, start_time_,
        serial_sequences_, infer_data_manager_, sequence_manager_);

    if (use_mock_infer_) {
      EXPECT_CALL(*worker, Infer())
          .WillRepeatedly(testing::Invoke(
              worker.get(), &MockRequestRateWorker::EmptyInfer));
    }
    return worker;
  }

  void TestSchedule(
      std::vector<uint64_t> intervals, PerfAnalyzerParameters params)
  {
    for (auto i : intervals) {
      custom_intervals_.push_back(nanoseconds{i});
    }
    nanoseconds measurement_window_nanoseconds{
        params.measurement_window_ms * NANOS_PER_MILLIS};
    nanoseconds max_test_duration{
        measurement_window_nanoseconds * params.max_trials};
    nanoseconds expected_current_timestamp{0};
    size_t intervals_index = 0;

    PauseWorkers();
    ConfigureThreads();
    GenerateSchedule();

    std::vector<nanoseconds> expected_timestamps;
    std::vector<nanoseconds> observed_timestamps;

    // Determine what the observed schedule was by getting each worker's
    // schedule and then sorting them together
    //
    for (auto worker : workers_) {
      nanoseconds observed_timestamp =
          std::dynamic_pointer_cast<RequestRateWorker>(worker)
              ->GetNextTimestamp();
      while (observed_timestamp <= max_test_duration) {
        observed_timestamps.push_back(observed_timestamp);
        observed_timestamp =
            std::dynamic_pointer_cast<RequestRateWorker>(worker)
                ->GetNextTimestamp();
      }
    }
    sort(observed_timestamps.begin(), observed_timestamps.end());

    // Determine what the schedule "should" be
    //
    while (expected_current_timestamp < observed_timestamps.back()) {
      expected_current_timestamp += custom_intervals_[intervals_index];
      expected_timestamps.push_back(expected_current_timestamp);
      intervals_index = (intervals_index + 1) % custom_intervals_.size();
    }

    // Confirm that the expected and observed schedules were the same
    //
    REQUIRE_MESSAGE(
        observed_timestamps.size() == expected_timestamps.size(),
        "Mismatch in size of schedules");

    for (size_t i = 0; i < observed_timestamps.size(); i++) {
      CHECK(observed_timestamps[i] == expected_timestamps[i]);
    }
  }

  void TestSequences(
      std::vector<uint64_t> intervals, bool check_sequences_balanced)
  {
    auto sleep_time = milliseconds(20);
    for (auto i : intervals) {
      custom_intervals_.push_back(nanoseconds{i});
    }

    PauseWorkers();
    ConfigureThreads();
    GenerateSchedule();
    ResumeWorkers();
    std::this_thread::sleep_for(sleep_time);
    if (check_sequences_balanced) {
      CheckSequenceBalance();
    }
    StopWorkerThreads();
  }

  std::shared_ptr<ModelParser>& parser_{LoadManager::parser_};
  std::shared_ptr<cb::ClientBackendFactory>& factory_{
      TestLoadManagerBase::factory_};

  std::string& request_intervals_file_{
      CustomLoadManager::request_intervals_file_};
  NanoIntervals& custom_intervals_{CustomLoadManager::custom_intervals_};

  cb::Error ReadTimeIntervalsFile(
      const std::string& path, NanoIntervals* contents) override
  {
    return cb::Error::Success;
  }

 private:
  bool use_mock_infer_;
};

TEST_CASE("custom_load_schedule")
{
  PerfAnalyzerParameters params;
  params.measurement_window_ms = 1000;
  params.max_trials = 10;
  bool is_sequence = false;
  bool is_decoupled = false;
  bool use_mock_infer = true;
  std::vector<uint64_t> intervals;

  const auto& ParameterizeIntervals{[&]() {
    SUBCASE("intervals A")
    {
      intervals = {100000000, 110000000, 130000000};
    }
    SUBCASE("intervals B")
    {
      intervals = {150000000};
    }
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

  const auto& ParameterizeSequences{[&]() {
    SUBCASE("sequences off")
    {
      ParameterizeMeasurementWindow();
      is_sequence = false;
    }
    SUBCASE("3 sequences")
    {
      ParameterizeMeasurementWindow();
      is_sequence = true;
      params.num_of_sequences = 3;
    }
    SUBCASE("6 sequences")
    {
      ParameterizeMeasurementWindow();
      is_sequence = true;
      params.num_of_sequences = 6;
    }
    SUBCASE("9 sequences")
    {
      ParameterizeMeasurementWindow();
      is_sequence = true;
      params.num_of_sequences = 9;
    }
  }};

  ParameterizeSequences();
  TestCustomLoadManager tclm(params, is_sequence, is_decoupled, use_mock_infer);
  tclm.TestSchedule(intervals, params);
}

TEST_CASE("custom_load_sequences")
{
  PerfAnalyzerParameters params;

  // This is needed so we can confirm that all sequences are being requested
  // equally when serial_sequences is on. Otherwise we would keep creating new
  // sequences and wouldn't be able to track it properly.
  //
  params.sequence_length = 1000;
  bool is_sequence_model = true;
  bool check_sequences_balanced = false;
  std::vector<uint64_t> intervals;

  const auto& ParameterizeIntervals{[&]() {
    SUBCASE("intervals A")
    {
      intervals = {100000, 110000, 130000};
    }
    SUBCASE("intervals B")
    {
      intervals = {150000};
    }
    SUBCASE("intervals C")
    {
      intervals = {100000, 110000, 120000, 130000, 140000};
    }
  }};

  const auto& ParameterizeSerialSequences{[&]() {
    SUBCASE("serial_sequences")
    {
      ParameterizeIntervals();
      params.serial_sequences = true;
      check_sequences_balanced = true;
    }
    SUBCASE("not serial_sequences")
    {
      ParameterizeIntervals();
      params.serial_sequences = false;
      check_sequences_balanced = false;
    }
  }};

  const auto& ParameterizeNumSequences{[&]() {
    SUBCASE("2 sequences")
    {
      ParameterizeSerialSequences();
      params.num_of_sequences = 2;
    }
    SUBCASE("3 sequences")
    {
      ParameterizeSerialSequences();
      params.num_of_sequences = 3;
    }
    SUBCASE("5 sequences")
    {
      ParameterizeSerialSequences();
      params.num_of_sequences = 5;
    }
    SUBCASE("6 sequences")
    {
      ParameterizeSerialSequences();
      params.num_of_sequences = 6;
    }
    SUBCASE("9 sequences")
    {
      ParameterizeSerialSequences();
      params.num_of_sequences = 9;
    }
  }};


  const auto& ParameterizeThreads{[&]() {
    SUBCASE("threads 1")
    {
      ParameterizeNumSequences();
      params.max_threads = 1;
    }
    SUBCASE("threads 2")
    {
      ParameterizeNumSequences();
      params.max_threads = 2;
    }
    SUBCASE("threads 4")
    {
      ParameterizeNumSequences();
      params.max_threads = 4;
    }
    SUBCASE("threads 7")
    {
      ParameterizeNumSequences();
      params.max_threads = 7;
    }
  }};

  ParameterizeThreads();

  TestCustomLoadManager tclm(params, is_sequence_model);
  tclm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  tclm.TestSequences(intervals, check_sequences_balanced);
}


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
