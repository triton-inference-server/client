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

namespace triton { namespace perfanalyzer {

// FIXME duplicate
class RequestRateWorkerMockedInferInput : public RequestRateWorker {
 public:
  RequestRateWorkerMockedInferInput(
      uint32_t id, std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config,
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<DataLoader> data_loader, cb::BackendKind backend_kind,
      const std::shared_ptr<cb::ClientBackendFactory> factory,
      const size_t sequence_length, const uint64_t start_sequence_id,
      const uint64_t sequence_id_range, const bool on_sequence_model,
      const bool async, const size_t max_threads, const bool using_json_data,
      const bool streaming, const SharedMemoryType shared_memory_type,
      const int32_t batch_size,
      std::vector<std::shared_ptr<SequenceStat>>& sequence_stat,
      std::unordered_map<std::string, SharedMemoryData>& shared_memory_regions,
      std::condition_variable& wake_signal, std::mutex& wake_mutex,
      bool& execute, std::atomic<uint64_t>& curr_seq_id,
      std::chrono::steady_clock::time_point& start_time,
      std::vector<std::chrono::nanoseconds>& schedule,
      std::shared_ptr<std::chrono::nanoseconds> gen_duration,
      std::uniform_int_distribution<uint64_t>& distribution)
      : RequestRateWorker(
            id, thread_stat, thread_config, parser, data_loader, backend_kind,
            factory, sequence_length, start_sequence_id, sequence_id_range,
            on_sequence_model, async, max_threads, using_json_data, streaming,
            shared_memory_type, batch_size, sequence_stat,
            shared_memory_regions, wake_signal, wake_mutex, execute,
            curr_seq_id, start_time, schedule, gen_duration, distribution)
  {
  }

  std::chrono::nanoseconds GetNextTimestamp() override
  {
    return RequestRateWorker::GetNextTimestamp();
  }

  std::shared_ptr<InferContext> CreateInferContext() override
  {
    return std::make_shared<InferContextMockedInferInput>(
        ctxs_.size(), async_, streaming_, on_sequence_model_, using_json_data_,
        batch_size_, backend_kind_, shared_memory_type_, shared_memory_regions_,
        start_sequence_id_, sequence_id_range_, sequence_length_, curr_seq_id_,
        distribution_, thread_stat_, sequence_stat_, data_loader_, parser_,
        factory_);
  }
};

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


  std::shared_ptr<IWorker> MakeWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config) override
  {
    uint32_t id = workers_.size();
    return std::make_shared<RequestRateWorkerMockedInferInput>(
        id, thread_stat, thread_config, LoadManager::parser_, data_loader_,
        backend_->Kind(), RequestRateManager::factory_, sequence_length_,
        start_sequence_id_, sequence_id_range_, on_sequence_model_, async_,
        max_threads_, using_json_data_, streaming_, shared_memory_type_,
        batch_size_, sequence_stat_, shared_memory_regions_, wake_signal_,
        wake_mutex_, execute_, curr_seq_id_, start_time_, schedule_,
        gen_duration_, distribution_);
  }

  void TestSchedule(
      std::vector<uint64_t> intervals, PerfAnalyzerParameters params)
  {
    for (auto i : intervals) {
      custom_intervals_.push_back(std::chrono::nanoseconds{i});
    }
    std::chrono::nanoseconds max_test_duration{params.measurement_window_ms *
                                               1000000 * params.max_trials};
    std::chrono::nanoseconds expected_current_timestamp{0};
    size_t intervals_index = 0;

    PauseWorkers();
    InitCustomIntervals();
    while (expected_current_timestamp.count() < max_test_duration.count()) {
      for (auto worker : workers_) {
        auto timestamp =
            std::dynamic_pointer_cast<RequestRateWorkerMockedInferInput>(worker)
                ->GetNextTimestamp();
        REQUIRE(timestamp.count() == expected_current_timestamp.count());
        expected_current_timestamp += custom_intervals_[intervals_index];
        intervals_index = (intervals_index + 1) % custom_intervals_.size();
      }
    }
  }


  std::shared_ptr<std::chrono::nanoseconds>& gen_duration_{
      RequestRateManager::gen_duration_};
  std::vector<std::chrono::nanoseconds>& schedule_{
      RequestRateManager::schedule_};
  std::string& request_intervals_file_{
      CustomLoadManager::request_intervals_file_};
  std::vector<std::chrono::nanoseconds>& custom_intervals_{
      CustomLoadManager::custom_intervals_};

  cb::Error ReadTimeIntervalsFile(
      const std::string& path,
      std::vector<std::chrono::nanoseconds>* contents) override
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

TEST_CASE("testing the InitCustomIntervals function")
{
  TestCustomLoadManager tclm{};

  SUBCASE("no file provided")
  {
    cb::Error result{tclm.InitCustomIntervals()};

    CHECK(result.Err() == SUCCESS);
    CHECK(tclm.schedule_.size() == 1);
    CHECK(tclm.schedule_[0] == std::chrono::nanoseconds(0));
  }

  SUBCASE("file provided")
  {
    tclm.request_intervals_file_ = "nonexistent_file.txt";
    tclm.gen_duration_ = std::make_unique<std::chrono::nanoseconds>(350000000);
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(100000000));
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(110000000));
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(130000000));

    cb::Error result{tclm.InitCustomIntervals()};

    CHECK(result.Err() == SUCCESS);
    CHECK(tclm.schedule_.size() == 5);
    CHECK(tclm.schedule_[0] == std::chrono::nanoseconds(0));
    CHECK(tclm.schedule_[1] == std::chrono::nanoseconds(100000000));
    CHECK(tclm.schedule_[2] == std::chrono::nanoseconds(210000000));
    CHECK(tclm.schedule_[3] == std::chrono::nanoseconds(340000000));
    CHECK(tclm.schedule_[4] == std::chrono::nanoseconds(440000000));
  }
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
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(100000000));
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(110000000));
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(130000000));

    cb::Error result{tclm.GetCustomRequestRate(&request_rate)};

    CHECK(result.Err() == SUCCESS);
    CHECK(request_rate == doctest::Approx(8.0));
  }
}
}}  // namespace triton::perfanalyzer
