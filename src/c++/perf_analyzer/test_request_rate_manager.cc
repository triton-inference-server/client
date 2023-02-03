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

#include <future>
#include "command_line_parser.h"
#include "common.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_data_loader.h"
#include "mock_memory_manager.h"
#include "mock_model_parser.h"
#include "request_rate_manager.h"
#include "test_load_manager_base.h"
#include "test_utils.h"

namespace cb = triton::perfanalyzer::clientbackend;
using milliseconds = std::chrono::milliseconds;
using nanoseconds = std::chrono::nanoseconds;

namespace triton {
namespace perfanalyzer {

class MockRequestRateWorker : public IWorker {
 public:
  MockRequestRateWorker(
      std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config)
      : thread_config_(thread_config)
  {
  }

  void Infer() override { thread_config_->is_paused_ = true; }

 private:
  std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config_;
};

/// Class to test the RequestRateManager
///
class TestRequestRateManager : public TestLoadManagerBase,
                               public RequestRateManager {
 public:
  TestRequestRateManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false,
      bool is_decoupled_model = false, bool use_mock_infer = false)
      : use_mock_infer_(use_mock_infer),
        TestLoadManagerBase(params, is_sequence_model, is_decoupled_model),
        RequestRateManager(
            params.async, params.streaming, params.request_distribution,
            params.batch_size, params.measurement_window_ms, params.max_trials,
            params.max_threads, params.num_of_sequences, params.sequence_length,
            params.shared_memory_type, params.output_shm_size,
            params.start_sequence_id, params.sequence_id_range, GetParser(),
            GetFactory())
  {
  }

  std::shared_ptr<IWorker> MakeWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config) override
  {
    uint32_t id = workers_.size();
    if (use_mock_infer_) {
      return std::make_shared<MockRequestRateWorker>(thread_config);
    } else {
      return RequestRateManager::MakeWorker(thread_stat, thread_config);
    }
  }

  void TestSchedule(double rate, PerfAnalyzerParameters params)
  {
    PauseWorkers();
    GenerateSchedule(rate);

    nanoseconds measurement_window_nanoseconds{params.measurement_window_ms *
                                               NANOS_PER_MILLIS};
    nanoseconds max_test_duration{measurement_window_nanoseconds *
                                  params.max_trials};

    nanoseconds expected_time_between_requests{int(NANOS_PER_SECOND / rate)};
    nanoseconds expected_current_timestamp{0};

    // Keep calling GetNextTimestamp for the entire test_duration to make sure
    // the schedule is exactly as expected
    //
    while (expected_current_timestamp < max_test_duration) {
      for (auto worker : workers_) {
        expected_current_timestamp += expected_time_between_requests;
        auto timestamp = std::dynamic_pointer_cast<RequestRateWorker>(worker)
                             ->GetNextTimestamp();
        REQUIRE(timestamp.count() == expected_current_timestamp.count());
      }
    }
    early_exit = true;
  }

  /// Test the public function ResetWorkers()
  ///
  /// ResetWorkers pauses and restarts the workers, but the most important and
  /// observable effects are the following:
  ///   - if threads_ is empty, it will populate threads_, threads_stat_, and
  ///   threads_config_ based on max_threads_
  ///   - start_time_ is updated with a new timestamp
  ///
  void TestResetWorkers()
  {
    // Capture the existing start time so we can confirm it changes
    //
    start_time_ = std::chrono::steady_clock::now();
    auto old_time = start_time_;

    SUBCASE("max threads 0")
    {
      // If max threads is 0, nothing happens other than updating
      // the start time
      //
      max_threads_ = 0;
      CHECK(start_time_ == old_time);
      ResetWorkers();
      CHECK(start_time_ != old_time);
      CHECK(threads_config_.size() == 0);
      CHECK(threads_stat_.size() == 0);
      CHECK(threads_.size() == 0);
    }
    SUBCASE("max threads 3, multiple calls")
    {
      max_threads_ = 3;

      // First call will populate threads/config/stat
      //
      CHECK(start_time_ == old_time);
      ResetWorkers();
      CHECK(start_time_ != old_time);
      CHECK(threads_config_.size() == 3);
      CHECK(threads_stat_.size() == 3);
      CHECK(threads_.size() == 3);

      // Second call will only update start_time
      //
      old_time = start_time_;
      ResetWorkers();
      CHECK(start_time_ != old_time);
      CHECK(threads_config_.size() == 3);
      CHECK(threads_stat_.size() == 3);
      CHECK(threads_.size() == 3);
    }
  }

  /// Test that the correct Infer function is called in the backend
  ///
  void TestInferType()
  {
    double request_rate = 50;
    auto sleep_time = milliseconds(100);

    ChangeRequestRate(request_rate);
    std::this_thread::sleep_for(sleep_time);
    StopWorkerThreads();

    CheckInferType();
  }

  /// Test that the inference distribution is as expected
  ///
  void TestDistribution(uint request_rate, uint duration_ms)
  {
    ChangeRequestRate(request_rate);
    std::this_thread::sleep_for(milliseconds(duration_ms));
    StopWorkerThreads();

    CheckCallDistribution(request_rate);
  }

  /// Test that the schedule is properly update after calling ChangeRequestRate
  ///
  void TestMultipleRequestRate()
  {
    std::vector<double> request_rates = {50, 200};
    auto sleep_time = milliseconds(500);

    for (auto request_rate : request_rates) {
      ChangeRequestRate(request_rate);
      ResetStats();
      std::this_thread::sleep_for(sleep_time);
      CheckCallDistribution(request_rate);
    }
  }

  /// Test sequence handling
  ///
  void TestSequences()
  {
    stats_->SetDelays({10});
    double request_rate1 = 100;
    double request_rate2 = 200;

    // A single sequence can't maintain the above rates
    //
    if (params_.num_of_sequences == 1) {
      request_rate1 = 50;
      request_rate2 = 100;
    }

    auto stats = cb::InferStat();
    int sleep_ms = 500;
    double num_seconds = double(sleep_ms) / 1000;

    auto sleep_time = milliseconds(sleep_ms);
    size_t expected_count1 = num_seconds * request_rate1;
    size_t expected_count2 = num_seconds * request_rate2 + expected_count1;

    // Run and check request rate 1
    //
    ChangeRequestRate(request_rate1);
    std::this_thread::sleep_for(sleep_time);

    stats = cb::InferStat();
    GetAccumulatedClientStat(&stats);
    CHECK(
        stats.completed_request_count ==
        doctest::Approx(expected_count1).epsilon(0.10));

    PauseWorkers();
    CheckSequences(params_.num_of_sequences);

    // Make sure that the client and the manager are in agreement on the request
    // count in between rates
    //
    stats = cb::InferStat();
    GetAccumulatedClientStat(&stats);
    int client_total_requests = stats_->num_async_infer_calls +
                                stats_->num_async_stream_infer_calls +
                                stats_->num_infer_calls;
    CHECK(stats.completed_request_count == client_total_requests);

    ResetStats();

    // Run and check request rate 2
    //
    ChangeRequestRate(request_rate2);
    std::this_thread::sleep_for(sleep_time);

    stats = cb::InferStat();
    GetAccumulatedClientStat(&stats);
    CHECK(
        stats.completed_request_count ==
        doctest::Approx(expected_count2).epsilon(0.10));

    // Stop all threads and make sure everything is as expected
    //
    StopWorkerThreads();

    CheckSequences(params_.num_of_sequences);
  }

  /// Test that the shared memory methods are called correctly
  ///
  void TestSharedMemory(uint request_rate, uint duration_ms)
  {
    ChangeRequestRate(request_rate);
    std::this_thread::sleep_for(milliseconds(duration_ms));
    StopWorkerThreads();
  }

  /// Test that tries to find deadlocks and livelocks
  ///
  void TestTimeouts()
  {
    TestWatchDog watchdog(1000);
    ChangeRequestRate(100);
    std::this_thread::sleep_for(milliseconds(100));
    StopWorkerThreads();
    watchdog.stop();
  }

  std::shared_ptr<ModelParser>& parser_{LoadManager::parser_};
  std::shared_ptr<DataLoader>& data_loader_{LoadManager::data_loader_};
  bool& using_json_data_{LoadManager::using_json_data_};
  bool& execute_{RequestRateManager::execute_};
  size_t& batch_size_{LoadManager::batch_size_};
  std::chrono::steady_clock::time_point& start_time_{
      RequestRateManager::start_time_};
  size_t& max_threads_{LoadManager::max_threads_};
  bool& async_{LoadManager::async_};
  bool& streaming_{LoadManager::streaming_};
  std::uniform_int_distribution<uint64_t>& distribution_{
      LoadManager::distribution_};
  std::shared_ptr<cb::ClientBackendFactory> factory_{
      TestLoadManagerBase::factory_};
  std::shared_ptr<MemoryManager>& memory_manager_{LoadManager::memory_manager_};

 private:
  bool use_mock_infer_;

  void CheckCallDistribution(int request_rate)
  {
    auto request_distribution = params_.request_distribution;

    auto timestamps = GetStats()->request_timestamps;
    std::vector<int64_t> time_delays = GatherTimeBetweenRequests(timestamps);

    double delay_average = CalculateAverage(time_delays);
    double delay_variance = CalculateVariance(time_delays, delay_average);

    double expected_delay_average =
        NANOS_PER_SECOND / static_cast<double>(request_rate);

    if (request_distribution == POISSON) {
      // By definition, variance == average for Poisson.
      //
      // With such a small sample size for a poisson distribution, there will be
      // noise. Allow 5% slop
      //
      CHECK(
          delay_average ==
          doctest::Approx(expected_delay_average).epsilon(0.05));
      CHECK(delay_variance == doctest::Approx(delay_average).epsilon(0.05));
    } else if (request_distribution == CONSTANT) {
      // constant should in theory have 0 variance, but with thread timing
      // there is obviously some noise.
      //
      // Allow it to be at most 5% of average
      //
      auto max_allowed_delay_variance = 0.05 * delay_average;

      // Constant should be pretty tight. Allowing 1% slop there is noise in the
      // thread scheduling
      //
      CHECK(
          delay_average ==
          doctest::Approx(expected_delay_average).epsilon(0.01));
      CHECK_LT(delay_variance, max_allowed_delay_variance);
    } else {
      throw std::invalid_argument("Unexpected distribution type");
    }
  }

  std::vector<int64_t> GatherTimeBetweenRequests(
      const std::vector<std::chrono::time_point<std::chrono::system_clock>>&
          timestamps)
  {
    std::vector<int64_t> time_between_requests;

    for (size_t i = 1; i < timestamps.size(); i++) {
      auto diff = timestamps[i] - timestamps[i - 1];
      nanoseconds diff_ns = std::chrono::duration_cast<nanoseconds>(diff);
      time_between_requests.push_back(diff_ns.count());
    }
    return time_between_requests;
  }
};

TEST_CASE("request_rate_schedule")
{
  PerfAnalyzerParameters params;
  params.measurement_window_ms = 1000;
  params.max_trials = 10;
  bool is_sequence = false;
  bool is_decoupled = false;
  bool use_mock_infer = false;
  double rate;

  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

  MockInputPipeline mip = TestLoadManagerBase::ProcessCustomJsonData(json_str);


  const auto& ParameterizeRate{[&]() {
    SUBCASE("rate 10") { rate = 10; }
    SUBCASE("rate 30") { rate = 30; }
    SUBCASE("rate 100") { rate = 100; }
  }};

  const auto& ParameterizeThreads{[&]() {
    SUBCASE("threads 1")
    {
      ParameterizeRate();
      params.max_threads = 1;
    }
    SUBCASE("threads 2")
    {
      ParameterizeRate();
      params.max_threads = 2;
    }
    SUBCASE("threads 4")
    {
      ParameterizeRate();
      params.max_threads = 4;
    }
    SUBCASE("threads 7")
    {
      ParameterizeRate();
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

  TestRequestRateManager trrm(
      params, is_sequence, is_decoupled, use_mock_infer);
  std::shared_ptr<MockMemoryManager> mock_memory_manager{
      std::make_shared<MockMemoryManager>(
          params.batch_size, params.shared_memory_type, params.output_shm_size,
          mip.mock_model_parser_, trrm.factory_, mip.mock_data_loader_)};
  trrm.memory_manager_ = mock_memory_manager;
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data);
  trrm.TestSchedule(rate, params);
}

TEST_CASE("request_rate_reset_workers: Test the public function ResetWorkers()")
{
  PerfAnalyzerParameters params;
  bool is_sequence = false;
  bool is_decoupled = false;
  bool use_mock_infer = true;
  TestRequestRateManager trrm(
      params, is_sequence, is_decoupled, use_mock_infer);
  trrm.TestResetWorkers();
}

/// Check that the correct inference function calls
/// are used given different param values for async and stream
///
TEST_CASE("request_rate_infer_type")
{
  bool async;
  bool stream;

  SUBCASE("async_stream")
  {
    async = true;
    stream = true;
  }
  SUBCASE("async_no_stream")
  {
    async = true;
    stream = false;
  }
  SUBCASE("no_async_stream")
  {
    async = false;
    stream = true;
  }
  SUBCASE("no_async_no_stream")
  {
    async = false;
    stream = false;
  }

  PerfAnalyzerParameters params;
  params.async = async;
  params.streaming = stream;
  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

  MockInputPipeline mip = TestLoadManagerBase::ProcessCustomJsonData(json_str);
  TestRequestRateManager trrm(params, false);
  std::shared_ptr<MockMemoryManager> mock_memory_manager{
      std::make_shared<MockMemoryManager>(
          params.batch_size, params.shared_memory_type, params.output_shm_size,
          mip.mock_model_parser_, trrm.factory_, mip.mock_data_loader_)};
  trrm.memory_manager_ = mock_memory_manager;
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data);
  trrm.TestInferType();
}

/// Check that the request distribution is correct for
/// different Distribution types
///
TEST_CASE("request_rate_distribution")
{
  PerfAnalyzerParameters params;
  uint request_rate = 500;
  uint duration_ms = 1000;

  SUBCASE("constant") { params.request_distribution = CONSTANT; }
  SUBCASE("poisson") { params.request_distribution = POISSON; }

  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

  MockInputPipeline mip = TestLoadManagerBase::ProcessCustomJsonData(json_str);
  TestRequestRateManager trrm(params);
  std::shared_ptr<MockMemoryManager> mock_memory_manager{
      std::make_shared<MockMemoryManager>(
          params.batch_size, params.shared_memory_type, params.output_shm_size,
          mip.mock_model_parser_, trrm.factory_, mip.mock_data_loader_)};
  trrm.memory_manager_ = mock_memory_manager;
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data);
  trrm.TestDistribution(request_rate, duration_ms);
}

/// Check that the request distribution is correct
/// for the case where the measurement window is tiny.
///
TEST_CASE("request_rate_tiny_window")
{
  PerfAnalyzerParameters params;
  params.request_distribution = CONSTANT;
  params.measurement_window_ms = 10;
  params.max_trials = 100;
  uint request_rate = 500;
  uint duration_ms = 1000;


  SUBCASE("one_thread") { params.max_threads = 1; }
  SUBCASE("odd_threads") { params.max_threads = 9; }

  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

  MockInputPipeline mip = TestLoadManagerBase::ProcessCustomJsonData(json_str);

  TestRequestRateManager trrm(params);
  std::shared_ptr<MockMemoryManager> mock_memory_manager{
      std::make_shared<MockMemoryManager>(
          params.batch_size, params.shared_memory_type, params.output_shm_size,
          mip.mock_model_parser_, trrm.factory_, mip.mock_data_loader_)};
  trrm.memory_manager_ = mock_memory_manager;
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data);
  trrm.TestDistribution(request_rate, duration_ms);
}

/// Check that the schedule properly handles mid-test
/// update to the request rate
///
TEST_CASE("request_rate_multiple")
{
  PerfAnalyzerParameters params{};
  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

  MockInputPipeline mip = TestLoadManagerBase::ProcessCustomJsonData(json_str);
  TestRequestRateManager trrm(PerfAnalyzerParameters{});
  std::shared_ptr<MockMemoryManager> mock_memory_manager{
      std::make_shared<MockMemoryManager>(
          params.batch_size, params.shared_memory_type, params.output_shm_size,
          mip.mock_model_parser_, trrm.factory_, mip.mock_data_loader_)};
  trrm.memory_manager_ = mock_memory_manager;
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data);
  trrm.TestMultipleRequestRate();
}

/// Check that the inference requests for sequences
/// follow all rules and parameters
///
TEST_CASE("request_rate_sequence")
{
  PerfAnalyzerParameters params = TestLoadManagerBase::GetSequenceTestParams();
  bool is_sequence_model = true;
  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

  MockInputPipeline mip = TestLoadManagerBase::ProcessCustomJsonData(json_str);
  TestRequestRateManager trrm(params, is_sequence_model);
  std::shared_ptr<MockMemoryManager> mock_memory_manager{
      std::make_shared<MockMemoryManager>(
          params.batch_size, params.shared_memory_type, params.output_shm_size,
          mip.mock_model_parser_, trrm.factory_, mip.mock_data_loader_)};
  trrm.memory_manager_ = mock_memory_manager;
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data);
  trrm.TestSequences();
}

TEST_CASE("request_rate_streaming: test that streaming-specific logic works")
{
  bool is_sequence = false;
  bool is_decoupled;
  bool expected_enable_stats_value;

  SUBCASE("enable_stats true")
  {
    is_decoupled = false;
    expected_enable_stats_value = true;
  }
  SUBCASE("enable_stats false")
  {
    is_decoupled = true;
    expected_enable_stats_value = false;
  }

  PerfAnalyzerParameters params{};
  params.streaming = true;

  RateSchedulePtr_t schedule = std::make_shared<RateSchedule>();
  schedule->intervals = NanoIntervals{nanoseconds(1)};
  schedule->duration = nanoseconds{1};

  std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
  std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config{
      std::make_shared<RequestRateWorker::ThreadConfig>(0, 0)};

  TestRequestRateManager trrm(params, is_sequence, is_decoupled);
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data);

  auto worker = trrm.MakeWorker(thread_stat, thread_config);
  std::dynamic_pointer_cast<IScheduler>(worker)->SetSchedule(schedule);
  std::future<void> infer_future{std::async(&IWorker::Infer, worker)};

  early_exit = true;
  infer_future.get();

  CHECK(
      trrm.stats_->start_stream_enable_stats_value ==
      expected_enable_stats_value);
}

TEST_CASE(
    "custom_json_data: Check custom json data to ensure that it is processed "
    "correctly")
{
  PerfAnalyzerParameters params{};
  bool is_sequence_model{false};
  const auto& ParameterizeAsyncAndStreaming
  {
    [](bool& async, bool& streaming) {
      SUBCASE("non-sequence")
      {
        is_sequence_model = false;
        ParameterizeAsyncAndStreaming(params.async, params.streaming);
      }
      SUBCASE("sequence")
      {
        is_sequence_model = true;
        params.num_of_sequences = 1;
        ParameterizeAsyncAndStreaming(params.async, params.streaming);
      }

      TestRequestRateManager trrm(params, is_sequence_model);

      std::shared_ptr<MockModelParser> mmp{
          std::make_shared<MockModelParser>(false, false)};
      ModelTensor model_tensor{};
      model_tensor.datatype_ = "INT32";
      model_tensor.is_optional_ = false;
      model_tensor.is_shape_tensor_ = false;
      model_tensor.name_ = "INPUT0";
      model_tensor.shape_ = {1};
      mmp->inputs_ = std::make_shared<ModelTensorMap>();
      (*mmp->inputs_)[model_tensor.name_] = model_tensor;

      std::shared_ptr<MockDataLoader> mdl{std::make_shared<MockDataLoader>()};
      const std::string json_str{R"(
{
  "data": [
    {
      "INPUT0": [2000000000]
    },
    {
      "INPUT0": [2000000001]
    }
  ]
}
    )"};
      mdl->ReadDataFromStr(json_str, mmp->Inputs(), mmp->Outputs());

      std::shared_ptr<MockMemoryManager> mock_memory_manager{
          std::make_shared<MockMemoryManager>(
              params.batch_size, params.shared_memory_type,
              params.output_shm_size, mmp, trrm.factory_, mdl)};


      std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
      std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config{
          std::make_shared<RequestRateWorker::ThreadConfig>(0, 1)};

      trrm.memory_manager_ = mock_memory_manager;
      trrm.parser_ = mmp;
      trrm.data_loader_ = mdl;
      trrm.using_json_data_ = true;
      trrm.execute_ = true;
      trrm.batch_size_ = 1;
      trrm.max_threads_ = 1;
      RateSchedulePtr_t schedule = std::make_shared<RateSchedule>();
      schedule->intervals = NanoIntervals{milliseconds(4), milliseconds(8),
                                          milliseconds(12), milliseconds(16)};
      schedule->duration = nanoseconds{16000000};

      trrm.distribution_ = std::uniform_int_distribution<uint64_t>(
          0, mdl->GetDataStreamsCount() - 1);
      trrm.start_time_ = std::chrono::steady_clock::now();

      trrm.InitManager(
          params.string_length, params.string_data, params.zero_input,
          params.user_data);
      std::shared_ptr<IWorker> worker{
          trrm.MakeWorker(thread_stat, thread_config)};
      std::dynamic_pointer_cast<IScheduler>(worker)->SetSchedule(schedule);
      std::future<void> infer_future{std::async(&IWorker::Infer, worker)};

      std::this_thread::sleep_for(milliseconds(18));

      early_exit = true;
      infer_future.get();

      const auto& recorded_inputs{trrm.stats_->recorded_inputs};

      REQUIRE(trrm.stats_->recorded_inputs.size() >= 4);
      CHECK(
          *reinterpret_cast<const int32_t*>(recorded_inputs[0][0].first) ==
          2000000000);
      CHECK(recorded_inputs[0][0].second == 4);
      CHECK(
          *reinterpret_cast<const int32_t*>(recorded_inputs[1][0].first) ==
          2000000001);
      CHECK(recorded_inputs[1][0].second == 4);
      CHECK(
          *reinterpret_cast<const int32_t*>(recorded_inputs[2][0].first) ==
          2000000000);
      CHECK(recorded_inputs[2][0].second == 4);
      CHECK(
          *reinterpret_cast<const int32_t*>(recorded_inputs[3][0].first) ==
          2000000001);
      CHECK(recorded_inputs[3][0].second == 4);
    }

    /// Check that the using_shared_memory_ is being set correctly
    ///
    TEST_CASE("Request rate - Check setting of InitSharedMemory")
    {
      PerfAnalyzerParameters params;
      bool is_sequence = false;
      bool is_decoupled = false;
      bool use_mock_infer = true;
      const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

      MockInputPipeline mip =
          TestLoadManagerBase::ProcessCustomJsonData(json_str);

      SUBCASE("No shared memory")
      {
        params.shared_memory_type = NO_SHARED_MEMORY;
        TestRequestRateManager trrm(
            params, is_sequence, is_decoupled, use_mock_infer);
        std::shared_ptr<MockMemoryManager> mock_memory_manager{
            std::make_shared<MockMemoryManager>(
                params.batch_size, params.shared_memory_type,
                params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
                mip.mock_data_loader_)};
        trrm.memory_manager_ = mock_memory_manager;
        trrm.InitManager(
            params.string_length, params.string_data, params.zero_input,
            params.user_data);
        CHECK(false == mock_memory_manager->using_shared_memory_);
      }

      SUBCASE("System shared memory")
      {
        params.shared_memory_type = SYSTEM_SHARED_MEMORY;
        TestRequestRateManager trrm(
            params, is_sequence, is_decoupled, use_mock_infer);
        std::shared_ptr<MockMemoryManager> mock_memory_manager{
            std::make_shared<MockMemoryManager>(
                params.batch_size, params.shared_memory_type,
                params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
                mip.mock_data_loader_)};
        trrm.memory_manager_ = mock_memory_manager;
        trrm.InitManager(
            params.string_length, params.string_data, params.zero_input,
            params.user_data);
        CHECK(true == mock_memory_manager->using_shared_memory_);
      }
    }

    /// Verify Shared Memory api calls
    ///
    TEST_CASE("Request rate - Shared memory methods")
    {
      PerfAnalyzerParameters params;
      bool is_sequence = false;
      bool is_decoupled = false;
      bool use_mock_infer = true;

      const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2123456789]
      }
    ]
  }
      )"};

      MockInputPipeline mip =
          TestLoadManagerBase::ProcessCustomJsonData(json_str);

      cb::MockClientStats::SharedMemoryStats expected_stats;

      SUBCASE("System shared memory usage")
      {
        params.shared_memory_type = SYSTEM_SHARED_MEMORY;
        TestRequestRateManager trrm(
            params, is_sequence, is_decoupled, use_mock_infer);
        std::shared_ptr<MockMemoryManager> mock_memory_manager{
            std::make_shared<MockMemoryManager>(
                params.batch_size, params.shared_memory_type,
                params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
                mip.mock_data_loader_)};
        trrm.memory_manager_ = mock_memory_manager;
        trrm.parser_ = mip.mock_model_parser_;
        trrm.data_loader_ = mip.mock_data_loader_;
        trrm.InitManager(
            params.string_length, params.string_data, params.zero_input,
            params.user_data);

        expected_stats.num_unregister_all_shared_memory_calls = 1;
        expected_stats.num_register_system_shared_memory_calls = 1;
        expected_stats.num_create_shared_memory_region_calls = 1;
        expected_stats.num_map_shared_memory_calls = 1;
        trrm.CheckSharedMemory(expected_stats);
      }

      SUBCASE("Cuda shared memory usage")
      {
        params.shared_memory_type = CUDA_SHARED_MEMORY;
        TestRequestRateManager trrm(
            params, is_sequence, is_decoupled, use_mock_infer);
        std::shared_ptr<MockMemoryManager> mock_memory_manager{
            std::make_shared<MockMemoryManager>(
                params.batch_size, params.shared_memory_type,
                params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
                mip.mock_data_loader_)};
        trrm.memory_manager_ = mock_memory_manager;
        trrm.parser_ = mip.mock_model_parser_;
        trrm.data_loader_ = mip.mock_data_loader_;
        trrm.InitManager(
            params.string_length, params.string_data, params.zero_input,
            params.user_data);

        expected_stats.num_unregister_all_shared_memory_calls = 1;
        expected_stats.num_register_cuda_shared_memory_calls = 1;
        trrm.CheckSharedMemory(expected_stats);
      }

      SUBCASE("No shared memory usage")
      {
        params.shared_memory_type = NO_SHARED_MEMORY;
        TestRequestRateManager trrm(
            params, is_sequence, is_decoupled, use_mock_infer);
        std::shared_ptr<MockMemoryManager> mock_memory_manager{
            std::make_shared<MockMemoryManager>(
                params.batch_size, params.shared_memory_type,
                params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
                mip.mock_data_loader_)};
        trrm.memory_manager_ = mock_memory_manager;
        trrm.parser_ = mip.mock_model_parser_;
        trrm.data_loader_ = mip.mock_data_loader_;
        trrm.InitManager(
            params.string_length, params.string_data, params.zero_input,
            params.user_data);

        trrm.CheckSharedMemory(expected_stats);
      }
    }

    TEST_CASE("Request rate - Shared memory infer input calls")
    {
      PerfAnalyzerParameters params{};
      bool is_sequence_model{false};

      const auto& ParameterizeAsyncAndStreaming{[&]() {
        SUBCASE("sync non-streaming")
        {
          params.async = false;
          params.streaming = false;
        }
        SUBCASE("async non-streaming")
        {
          params.async = true;
          params.streaming = false;
        }
        SUBCASE("async streaming")
        {
          params.async = true;
          params.streaming = true;
        }
      }};

      const auto& ParameterizeSequence{[&]() {
        SUBCASE("non-sequence")
        {
          is_sequence_model = false;
          ParameterizeAsyncAndStreaming();
        }
        SUBCASE("sequence")
        {
          is_sequence_model = true;
          params.num_of_sequences = 1;
          ParameterizeAsyncAndStreaming();
        }
      }};

      const auto& ParameterizeMemory{[&]() {
        SUBCASE("No shared memory")
        {
          params.shared_memory_type = NO_SHARED_MEMORY;
          ParameterizeSequence();
        }
        SUBCASE("system shared memory")
        {
          params.shared_memory_type = SYSTEM_SHARED_MEMORY;
          ParameterizeSequence();
        }
        SUBCASE("cuda shared memory")
        {
          params.shared_memory_type = CUDA_SHARED_MEMORY;
          ParameterizeSequence();
        }
      }};

      ParameterizeMemory();
      TestRequestRateManager trrm(params, is_sequence_model);

      const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};
      MockInputPipeline mip =
          TestLoadManagerBase::ProcessCustomJsonData(json_str);
      std::shared_ptr<MockMemoryManager> mock_memory_manager{
          std::make_shared<MockMemoryManager>(
              params.batch_size, params.shared_memory_type,
              params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
              mip.mock_data_loader_)};

      std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
      std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config{
          std::make_shared<RequestRateWorker::ThreadConfig>(0, 1)};

      trrm.memory_manager_ = mock_memory_manager;
      trrm.parser_ = mip.mock_model_parser_;
      trrm.data_loader_ = mip.mock_data_loader_;
      trrm.using_json_data_ = true;
      trrm.execute_ = true;
      trrm.batch_size_ = 1;
      trrm.max_threads_ = 1;

      RateSchedulePtr_t schedule = std::make_shared<RateSchedule>();
      schedule->intervals = NanoIntervals{milliseconds(4), milliseconds(8),
                                          milliseconds(12), milliseconds(16)};
      schedule->duration = nanoseconds{16000000};

      trrm.distribution_ = std::uniform_int_distribution<uint64_t>(
          0, mip.mock_data_loader_->GetDataStreamsCount() - 1);
      trrm.start_time_ = std::chrono::steady_clock::now();
      trrm.InitManager(
          params.string_length, params.string_data, params.zero_input,
          params.user_data);

      std::shared_ptr<IWorker> worker{
          trrm.MakeWorker(thread_stat, thread_config)};
      std::dynamic_pointer_cast<IScheduler>(worker)->SetSchedule(schedule);
      std::future<void> infer_future{std::async(&IWorker::Infer, worker)};

      std::this_thread::sleep_for(milliseconds(18));

      early_exit = true;
      infer_future.get();

      const auto& actual_append_raw_calls{trrm.stats_->num_append_raw_calls};
      const auto& actual_set_shared_memory_calls{
          trrm.stats_->num_set_shared_memory_calls};

      if (params.shared_memory_type == NO_SHARED_MEMORY) {
        CHECK(actual_append_raw_calls > 0);
        CHECK(actual_set_shared_memory_calls == 0);
      } else {
        CHECK(actual_append_raw_calls == 0);
        CHECK(actual_set_shared_memory_calls > 0);
      }
    }

    TEST_CASE("request_rate_deadlock")
    {
      PerfAnalyzerParameters params{};
      params.max_concurrency = 6;
      bool is_sequence_model{true};
      bool some_infer_failures{false};

      const auto& ParameterizeSync{[&]() {
        SUBCASE("sync")
        {
          params.async = false;
          params.streaming = false;
        }
        SUBCASE("aync no streaming")
        {
          params.async = true;
          params.streaming = false;
        }
        SUBCASE("async streaming")
        {
          params.async = true;
          params.streaming = true;
        }
      }};

      const auto& ParameterizeThreads{[&]() {
        SUBCASE("2 thread")
        {
          ParameterizeSync();
          params.max_threads = 2;
        }
        SUBCASE("10 thread")
        {
          ParameterizeSync();
          params.max_threads = 10;
        }
      }};

      const auto& ParameterizeSequence{[&]() {
        SUBCASE("non-sequence")
        {
          ParameterizeThreads();
          is_sequence_model = false;
        }
        SUBCASE("sequence")
        {
          ParameterizeThreads();
          is_sequence_model = true;
          params.num_of_sequences = 3;
        }
      }};

      const auto& ParameterizeFailures{[&]() {
        SUBCASE("yes_failures")
        {
          some_infer_failures = true;
          ParameterizeSequence();
        }
        SUBCASE("no_failures")
        {
          some_infer_failures = false;
          ParameterizeSequence();
        }
      }};

      std::vector<uint64_t> delays;

      const auto& ParameterizeDelays{[&]() {
        SUBCASE("no_delay")
        {
          delays = {0};
          ParameterizeFailures();
        }
        SUBCASE("random_delay")
        {
          delays = {1, 5, 20, 4, 3};
          ParameterizeFailures();
        }
      }};

      ParameterizeDelays();

      const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

      MockInputPipeline mip =
          TestLoadManagerBase::ProcessCustomJsonData(json_str);
      TestRequestRateManager trrm(params, is_sequence_model);
      std::shared_ptr<MockMemoryManager> mock_memory_manager{
          std::make_shared<MockMemoryManager>(
              params.batch_size, params.shared_memory_type,
              params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
              mip.mock_data_loader_)};
      trrm.memory_manager_ = mock_memory_manager;
      trrm.stats_->SetDelays(delays);
      trrm.InitManager(
          params.string_length, params.string_data, params.zero_input,
          params.user_data);

      // Sometimes have a request fail
      if (some_infer_failures) {
        trrm.stats_->SetReturnStatuses({true, true, true, false});
      }

      trrm.TestTimeouts();
    }
  }
}  // namespace triton::perfanalyzer
