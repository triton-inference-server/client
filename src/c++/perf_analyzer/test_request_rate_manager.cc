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
#include <memory>

#include "command_line_parser.h"
#include "common.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_data_loader.h"
#include "mock_infer_data_manager.h"
#include "mock_model_parser.h"
#include "mock_request_rate_worker.h"
#include "mock_sequence_manager.h"
#include "request_rate_manager.h"
#include "test_load_manager_base.h"
#include "test_utils.h"

namespace cb = triton::perfanalyzer::clientbackend;
using milliseconds = std::chrono::milliseconds;
using nanoseconds = std::chrono::nanoseconds;

namespace triton { namespace perfanalyzer {

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
            params.max_threads, params.num_of_sequences,
            params.shared_memory_type, params.output_shm_size,
            params.serial_sequences, GetParser(), GetFactory())
  {
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

  void TestConfigureThreads(
      std::vector<RequestRateWorker::ThreadConfig>& expected_configs)
  {
    RequestRateManager::ConfigureThreads();

    auto expected_size = expected_configs.size();

    // Check that the correct number of threads are created
    //
    CHECK(threads_.size() == expected_size);

    // Check that threads_config has correct number of sequences and
    // seq stat index offset
    for (auto i = 0; i < expected_configs.size(); i++) {
      CHECK(
          threads_config_[i]->num_sequences_ ==
          expected_configs[i].num_sequences_);
      CHECK(
          threads_config_[i]->seq_stat_index_offset_ ==
          expected_configs[i].seq_stat_index_offset_);
    }
  }

  void TestCalculateThreadIds(std::vector<size_t>& expected_thread_ids)
  {
    std::vector<size_t> actual_thread_ids =
        RequestRateManager::CalculateThreadIds();
    CHECK(actual_thread_ids.size() == expected_thread_ids.size());

    for (auto i = 0; i < actual_thread_ids.size(); i++) {
      CHECK(actual_thread_ids[i] == expected_thread_ids[i]);
    }
  }

  void StopWorkerThreads() { LoadManager::StopWorkerThreads(); }

  void TestSchedule(double rate, PerfAnalyzerParameters params)
  {
    PauseWorkers();
    ConfigureThreads();
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

  void TestCreateSchedule(
      double rate, PerfAnalyzerParameters params,
      std::vector<uint32_t>& expected_worker_ratio)
  {
    PauseWorkers();
    ConfigureThreads();
    GenerateSchedule(rate);

    std::vector<uint32_t> worker_schedule_sizes;
    uint32_t total_num_seqs{0};

    for (auto worker : workers_) {
      auto w = std::dynamic_pointer_cast<RequestRateWorker>(worker);
      total_num_seqs += w->thread_config_->num_sequences_;
      worker_schedule_sizes.push_back(w->schedule_->intervals.size());
    }
    early_exit = true;

    CHECK(num_of_sequences_ == total_num_seqs);
    for (int i = 0; i < worker_schedule_sizes.size() - 1; i++) {
      CHECK(
          worker_schedule_sizes[i] / expected_worker_ratio[i] ==
          worker_schedule_sizes[i + 1] / expected_worker_ratio[i + 1]);
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
  void TestSequences(bool verify_seq_balance, bool check_expected_count)
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
    if (check_expected_count) {
      CHECK(
          stats.completed_request_count ==
          doctest::Approx(expected_count1).epsilon(0.10));
    }

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

    if (verify_seq_balance) {
      CheckSequenceBalance();
    }

    ResetStats();

    // Run and check request rate 2
    //
    ChangeRequestRate(request_rate2);
    std::this_thread::sleep_for(sleep_time);

    stats = cb::InferStat();
    GetAccumulatedClientStat(&stats);
    if (check_expected_count) {
      CHECK(
          stats.completed_request_count ==
          doctest::Approx(expected_count2).epsilon(0.10));
    }

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

  /// Test that idle time is tracked correctly
  void TestOverhead(uint request_rate)
  {
    stats_->SetDelays({1});
    ChangeRequestRate(request_rate);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // During a run of 100 ms (100,000,000 ns), make sure that the idle time is
    // at least 95% of that
    //
    auto idle_time_ns = GetIdleTime();
    CHECK(idle_time_ns > 95000000);
    StopWorkerThreads();
  }

  /// Helper function that will setup and run a case to verify custom data
  /// behavior
  /// \param num_requests Integer number of requests to send during the test
  /// \param tensors Vector of input ModelTensors
  /// \param json_str The custom data json text
  /// \param expected_values Vector of expected input values for each inference
  /// \param expect_init_failure True if InitManager is expected to throw an
  /// error
  /// \param expect_thread_failure True if the thread is expected to have
  /// an error
  void TestCustomData(
      size_t num_requests, std::vector<ModelTensor>& tensors,
      const std::string json_str,
      std::vector<std::vector<int32_t>>& expected_values,
      bool expect_init_failure, bool expect_thread_failure)
  {
    CustomDataTestSetup(tensors, json_str, expect_init_failure);
    if (expect_init_failure) {
      // The rest of the test is invalid if init failed
      return;
    }
    auto thread_status = CustomDataTestSendRequests(num_requests);
    CustomDataTestCheckResults(
        thread_status, expect_thread_failure, expected_values);
  }

  void CustomDataTestSetup(
      std::vector<ModelTensor>& tensors, const std::string json_str,
      bool expect_init_failure)
  {
    params_.user_data = {json_str};

    std::shared_ptr<MockDataLoader> mdl{
        std::make_shared<MockDataLoader>(params_.batch_size)};

    std::shared_ptr<MockModelParser> mmp{
        std::make_shared<MockModelParser>(on_sequence_model_, false)};
    mmp->inputs_ = std::make_shared<ModelTensorMap>();
    for (auto t : tensors) {
      (*mmp->inputs_)[t.name_] = t;
    }

    infer_data_manager_ =
        MockInferDataManagerFactory::CreateMockInferDataManager(
            params_.max_threads, params_.batch_size, params_.shared_memory_type,
            params_.output_shm_size, mmp, factory_, mdl);

    parser_ = mmp;
    data_loader_ = mdl;
    using_json_data_ = true;
    execute_ = true;
    max_threads_ = 1;

    if (expect_init_failure) {
      REQUIRE_THROWS_AS(
          InitManager(
              params_.string_length, params_.string_data, params_.zero_input,
              params_.user_data, params_.start_sequence_id,
              params_.sequence_id_range, params_.sequence_length,
              params_.sequence_length_specified,
              params_.sequence_length_variation),
          PerfAnalyzerException);
      return;
    } else {
      REQUIRE_NOTHROW(InitManager(
          params_.string_length, params_.string_data, params_.zero_input,
          params_.user_data, params_.start_sequence_id,
          params_.sequence_id_range, params_.sequence_length,
          params_.sequence_length_specified,
          params_.sequence_length_variation));
    }
  }

  cb::Error CustomDataTestSendRequests(size_t num_requests)
  {
    std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
    std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config{
        std::make_shared<RequestRateWorker::ThreadConfig>(0)};
    std::shared_ptr<IWorker> worker{MakeWorker(thread_stat, thread_config)};

    auto mock_worker = std::dynamic_pointer_cast<MockRequestRateWorker>(worker);

    mock_worker->CreateContext();

    for (size_t i = 0; i < num_requests; i++) {
      mock_worker->SendInferRequest();
    }

    return thread_stat->status_;
  }

  void CustomDataTestCheckResults(
      cb::Error& thread_status, bool expect_thread_failure,
      std::vector<std::vector<int32_t>>& expected_values)
  {
    if (expect_thread_failure) {
      REQUIRE(!thread_status.IsOk());
    } else {
      REQUIRE_MESSAGE(thread_status.IsOk(), thread_status.Message());
    }

    auto recorded_values = GetRecordedInputValues();

    // Check that results are exactly as expected
    REQUIRE(recorded_values.size() == expected_values.size());
    for (size_t i = 0; i < expected_values.size(); i++) {
      REQUIRE(recorded_values[i].size() == expected_values[i].size());
      for (size_t j = 0; j < expected_values[i].size(); j++) {
        CHECK(recorded_values[i][j] == expected_values[i][j]);
      }
    }
  }

  std::shared_ptr<ModelParser>& parser_{LoadManager::parser_};
  std::shared_ptr<DataLoader>& data_loader_{LoadManager::data_loader_};
  std::shared_ptr<SequenceManager>& sequence_manager_{
      LoadManager::sequence_manager_};
  bool& using_json_data_{LoadManager::using_json_data_};
  bool& execute_{RequestRateManager::execute_};
  size_t& batch_size_{LoadManager::batch_size_};
  std::chrono::steady_clock::time_point& start_time_{
      RequestRateManager::start_time_};
  size_t& max_threads_{LoadManager::max_threads_};
  bool& async_{LoadManager::async_};
  bool& streaming_{LoadManager::streaming_};
  std::shared_ptr<cb::ClientBackendFactory>& factory_{
      TestLoadManagerBase::factory_};
  std::shared_ptr<IInferDataManager>& infer_data_manager_{
      LoadManager::infer_data_manager_};

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

  // Gets the inputs recorded in the mock backend
  // Returns a vector of vector of int32_t. Each entry in the parent vector is a
  // list of all input values for a single inference request
  //
  std::vector<std::vector<int32_t>> GetRecordedInputValues()
  {
    auto recorded_inputs{stats_->recorded_inputs};
    std::vector<std::vector<int32_t>> recorded_values;
    // Convert the recorded inputs into values, for both shared memory and non
    // shared memory cases
    //
    if (params_.shared_memory_type != SharedMemoryType::NO_SHARED_MEMORY) {
      auto recorded_memory_regions =
          std::dynamic_pointer_cast<MockInferDataManagerShm>(
              infer_data_manager_)
              ->mocked_shared_memory_regions;
      for (auto recorded_input : recorded_inputs) {
        std::vector<int32_t> recorded_value;
        for (auto memory_label : recorded_input) {
          auto itr =
              recorded_memory_regions.find(memory_label.shared_memory_label);
          if (itr == recorded_memory_regions.end()) {
            std::string err_str = "Test error: Could not find label " +
                                  memory_label.shared_memory_label +
                                  " in recorded shared memory";
            REQUIRE_MESSAGE(false, err_str);
          } else {
            for (auto val : itr->second) {
              recorded_value.push_back(val);
            }
          }
        }
        recorded_values.push_back(recorded_value);
      }
    } else {
      for (auto recorded_input : recorded_inputs) {
        std::vector<int32_t> recorded_value;
        for (auto val : recorded_input) {
          recorded_value.push_back(val.data);
        }
        recorded_values.push_back(recorded_value);
      }
    }
    return recorded_values;
  }

  std::shared_ptr<SequenceManager> MakeSequenceManager(
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      const size_t sequence_length, const bool sequence_length_specified,
      const double sequence_length_variation, const bool using_json_data,
      std::shared_ptr<DataLoader> data_loader) override
  {
    return std::make_shared<MockSequenceManager>(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
  }
};

TEST_CASE("request_rate_schedule")
{
  PerfAnalyzerParameters params;
  params.measurement_window_ms = 1000;
  params.max_trials = 10;
  bool is_sequence = false;
  bool is_decoupled = false;
  bool use_mock_infer = true;
  double rate;


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

  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
  trrm.TestSchedule(rate, params);
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

  TestRequestRateManager trrm(params, false);

  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
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

  TestRequestRateManager trrm(params);

  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
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


  TestRequestRateManager trrm(params);
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
  trrm.TestDistribution(request_rate, duration_ms);
}

/// Check that the schedule properly handles mid-test
/// update to the request rate
///
TEST_CASE("request_rate_multiple")
{
  PerfAnalyzerParameters params{};
  TestRequestRateManager trrm(PerfAnalyzerParameters{});

  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
  trrm.TestMultipleRequestRate();
}

/// Check that the inference requests for sequences
/// follow all rules and parameters
///
TEST_CASE("request_rate_sequence")
{
  PerfAnalyzerParameters params = TestLoadManagerBase::GetSequenceTestParams();
  bool verify_seq_balance = false;
  bool check_expected_count = true;
  bool is_sequence_model = true;

  TestRequestRateManager trrm(params, is_sequence_model);
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
  trrm.TestSequences(verify_seq_balance, check_expected_count);
}

TEST_CASE("request_rate_serial_sequences")
{
  PerfAnalyzerParameters params;
  params.serial_sequences = true;
  bool verify_seq_balance = false;
  bool check_expected_count = true;
  bool is_sequence_model = true;

  const auto& ParameterizeDistribution{[&]() {
    SUBCASE("Constant") { params.request_distribution = CONSTANT; }
    SUBCASE("Poisson")
    {
      params.request_distribution = POISSON;
      check_expected_count = false;
    }
  }};

  SUBCASE("num seqs 7, threads 4")
  {
    verify_seq_balance = true;
    params.sequence_length = 100;
    params.num_of_sequences = 7;
    params.max_threads = 4;
    ParameterizeDistribution();
  }
  SUBCASE("num seqs 13, threads 5")
  {
    verify_seq_balance = true;
    params.sequence_length = 100;
    params.num_of_sequences = 13;
    params.max_threads = 5;
    ParameterizeDistribution();
  }

  TestRequestRateManager trrm(params, is_sequence_model);
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
  trrm.TestSequences(verify_seq_balance, check_expected_count);
}

TEST_CASE("request_rate max inflight per seq")
{
  // Confirm that we can have multiple inferences in-flight for a given sequence
  // unless in serial-sequence mode
  PerfAnalyzerParameters params;
  bool is_sequence_model = true;
  params.num_of_sequences = 2;
  size_t rate = 1000;
  size_t time_ms = 10;

  bool expect_multiple_in_flight_sequences = false;

  SUBCASE("sync will never have multiple in flight")
  {
    params.async = false;
    expect_multiple_in_flight_sequences = false;

    SUBCASE("serial_sequences on") { params.serial_sequences = true; }
    SUBCASE("serial_sequences off") { params.serial_sequences = false; }
  }
  SUBCASE("async may have multiple in flight depending on serial sequences")
  {
    params.async = true;

    SUBCASE("serial_sequences on")
    {
      params.serial_sequences = true;
      expect_multiple_in_flight_sequences = false;
    }
    SUBCASE("serial_sequences off")
    {
      params.serial_sequences = false;
      expect_multiple_in_flight_sequences = true;
    }
  }

  TestRequestRateManager trrm(params, is_sequence_model);
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  trrm.stats_->SetDelays({100});

  trrm.ChangeRequestRate(rate);
  std::this_thread::sleep_for(std::chrono::milliseconds(time_ms));

  auto max_observed_inflight =
      trrm.stats_->sequence_status.max_inflight_seq_count;

  if (expect_multiple_in_flight_sequences) {
    CHECK(max_observed_inflight > 1);
  } else {
    CHECK(max_observed_inflight == 1);
  }

  trrm.StopWorkerThreads();
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
      std::make_shared<RequestRateWorker::ThreadConfig>(0)};

  TestRequestRateManager trrm(params, is_sequence, is_decoupled);
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

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
  params.user_data = {"fake_file.json"};
  bool is_sequence_model{false};

  std::vector<std::vector<int32_t>> expected_results;
  std::vector<ModelTensor> tensors;
  bool expect_init_failure = false;
  bool expect_thread_failure = false;

  ModelTensor model_tensor1{};
  model_tensor1.datatype_ = "INT32";
  model_tensor1.is_optional_ = false;
  model_tensor1.is_shape_tensor_ = false;
  model_tensor1.name_ = "INPUT1";
  model_tensor1.shape_ = {1};

  ModelTensor model_tensor2 = model_tensor1;
  model_tensor2.name_ = "INPUT2";

  std::string json_str{R"({
   "data": [
     { "INPUT1": [1], "INPUT2": [21] },
     { "INPUT1": [2], "INPUT2": [22] },
     { "INPUT1": [3], "INPUT2": [23] }
   ]})"};

  size_t num_requests = 4;

  const auto& ParameterizeTensors{[&]() {
    SUBCASE("one tensor")
    {
      tensors.push_back(model_tensor1);

      switch (params.batch_size) {
        case 1:
          expected_results = {{1}, {2}, {3}, {1}};
          break;
        case 2:
          expected_results = {{1, 2}, {3, 1}, {2, 3}, {1, 2}};
          break;
        case 4:
          expected_results = {
              {1, 2, 3, 1}, {2, 3, 1, 2}, {3, 1, 2, 3}, {1, 2, 3, 1}};
          break;
        default:
          REQUIRE(false);
      }
    }
    SUBCASE("two tensors")
    {
      tensors.push_back(model_tensor1);
      tensors.push_back(model_tensor2);

      switch (params.batch_size) {
        case 1:
          expected_results = {{1, 21}, {2, 22}, {3, 23}, {1, 21}};
          break;
        case 2:
          expected_results = {
              {1, 2, 21, 22}, {3, 1, 23, 21}, {2, 3, 22, 23}, {1, 2, 21, 22}};
          break;
        case 4:
          expected_results = {{1, 2, 3, 1, 21, 22, 23, 21},
                              {2, 3, 1, 2, 22, 23, 21, 22},
                              {3, 1, 2, 3, 23, 21, 22, 23},
                              {1, 2, 3, 1, 21, 22, 23, 21}};
          break;
        default:
          REQUIRE(false);
      }
    }
  }};

  const auto& ParameterizeBatchSize{[&]() {
    SUBCASE("batchsize = 1")
    {
      params.batch_size = 1;
      ParameterizeTensors();
    }
    SUBCASE("batchsize = 2")
    {
      params.batch_size = 2;
      ParameterizeTensors();
    }
    SUBCASE("batchsize = 4")
    {
      params.batch_size = 4;
      ParameterizeTensors();
    }
  }};

  const auto& ParameterizeSharedMemory{[&]() {
    SUBCASE("no_shared_memory")
    {
      params.shared_memory_type = SharedMemoryType::NO_SHARED_MEMORY;
      ParameterizeBatchSize();
    }
    SUBCASE("system_shared_memory")
    {
      params.shared_memory_type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
      ParameterizeBatchSize();
    }
    SUBCASE("cuda_shared_memory")
    {
      params.shared_memory_type = SharedMemoryType::CUDA_SHARED_MEMORY;
      ParameterizeBatchSize();
    }
  }};

  ParameterizeSharedMemory();

  TestRequestRateManager trrm(params, is_sequence_model);

  trrm.TestCustomData(
      num_requests, tensors, json_str, expected_results, expect_init_failure,
      expect_thread_failure);
}

TEST_CASE("custom_json_data: handling is_shape_tensor")
{
  // Test the case where is_shape_tensor is true and is the same
  // across a batch: it only ends up in each batch once
  PerfAnalyzerParameters params{};
  params.user_data = {"fake_file.json"};
  bool is_sequence_model{false};

  std::vector<std::vector<int32_t>> expected_results;
  std::vector<ModelTensor> tensors;
  bool expect_init_failure = false;
  bool expect_thread_failure = false;

  ModelTensor model_tensor1{};
  model_tensor1.datatype_ = "INT32";
  model_tensor1.is_optional_ = false;
  model_tensor1.is_shape_tensor_ = false;
  model_tensor1.name_ = "INPUT1";
  model_tensor1.shape_ = {1};

  ModelTensor model_tensor2 = model_tensor1;
  model_tensor2.name_ = "INPUT2";

  std::string json_str{R"({
   "data": [
     { "INPUT1": [1], "INPUT2": [21] },
     { "INPUT1": [1], "INPUT2": [22] },
     { "INPUT1": [1], "INPUT2": [23] }
   ]})"};

  model_tensor1.is_shape_tensor_ = true;
  model_tensor2.is_optional_ = true;

  size_t num_requests = 4;


  const auto& ParameterizeBatch{[&]() {
    SUBCASE("batch 1")
    {
      params.batch_size = 1;
      expected_results = {{1, 21}, {1, 22}, {1, 23}, {1, 21}};
    }
    SUBCASE("batch 2")
    {
      params.batch_size = 2;
      expected_results = {{1, 21, 22}, {1, 23, 21}, {1, 22, 23}, {1, 21, 22}};
    }
    SUBCASE("batch 4")
    {
      params.batch_size = 4;
      expected_results = {{1, 21, 22, 23, 21},
                          {1, 22, 23, 21, 22},
                          {1, 23, 21, 22, 23},
                          {1, 21, 22, 23, 21}};
    }
  }};

  // Being optional should have no impact
  SUBCASE("optional = 0,0")
  {
    model_tensor1.is_optional_ = false;
    model_tensor2.is_optional_ = false;
    ParameterizeBatch();
  }
  SUBCASE("optional = 0,1")
  {
    model_tensor1.is_optional_ = false;
    model_tensor2.is_optional_ = true;
    ParameterizeBatch();
  }
  SUBCASE("optional = 1,0")
  {
    model_tensor1.is_optional_ = true;
    model_tensor2.is_optional_ = false;
    ParameterizeBatch();
  }
  SUBCASE("optional = 1,1")
  {
    model_tensor1.is_optional_ = true;
    model_tensor2.is_optional_ = true;
    ParameterizeBatch();
  }


  TestRequestRateManager trrm(params, is_sequence_model);

  tensors.push_back(model_tensor1);
  tensors.push_back(model_tensor2);

  trrm.TestCustomData(
      num_requests, tensors, json_str, expected_results, expect_init_failure,
      expect_thread_failure);
}

TEST_CASE("custom_json_data: handling missing optional is_shape_tensor")
{
  // Test the case where is_shape_tensor is true and is_optional_ is true
  // and data for that input is completely omitted
  PerfAnalyzerParameters params{};
  params.user_data = {"fake_file.json"};
  bool is_sequence_model{false};

  std::vector<std::vector<int32_t>> expected_results;
  std::vector<ModelTensor> tensors;
  bool expect_init_failure = false;
  bool expect_thread_failure = false;

  ModelTensor model_tensor1{};
  model_tensor1.datatype_ = "INT32";
  model_tensor1.is_optional_ = true;
  model_tensor1.is_shape_tensor_ = true;
  model_tensor1.name_ = "INPUT1";
  model_tensor1.shape_ = {1};

  ModelTensor model_tensor2 = model_tensor1;
  model_tensor2.is_shape_tensor_ = false;
  model_tensor2.is_optional_ = false;
  model_tensor2.name_ = "INPUT2";

  std::string json_str{R"({
   "data": [
     { "INPUT2": [21] },
     { "INPUT2": [22] },
     { "INPUT2": [23] }
   ]})"};


  size_t num_requests = 4;

  const auto& ParameterizeBatch{[&]() {
    SUBCASE("batch 1")
    {
      params.batch_size = 1;
      expected_results = {{21}, {22}, {23}, {21}};
    }
    SUBCASE("batch 2")
    {
      params.batch_size = 2;
      expected_results = {{21, 22}, {23, 21}, {22, 23}, {21, 22}};
    }
    SUBCASE("batch 4")
    {
      params.batch_size = 4;
      expected_results = {{21, 22, 23, 21},
                          {22, 23, 21, 22},
                          {23, 21, 22, 23},
                          {21, 22, 23, 21}};
    }
  }};

  SUBCASE("no shm")
  {
    params.shared_memory_type = SharedMemoryType::NO_SHARED_MEMORY;
    ParameterizeBatch();
  }
  SUBCASE("system shm")
  {
    params.shared_memory_type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
    ParameterizeBatch();
    expect_init_failure = true;
  }
  SUBCASE("cuda shm")
  {
    params.shared_memory_type = SharedMemoryType::CUDA_SHARED_MEMORY;
    ParameterizeBatch();
    expect_init_failure = true;
  }

  TestRequestRateManager trrm(params, is_sequence_model);

  tensors.push_back(model_tensor1);
  tensors.push_back(model_tensor2);

  trrm.TestCustomData(
      num_requests, tensors, json_str, expected_results, expect_init_failure,
      expect_thread_failure);
}

TEST_CASE("custom_json_data: handling invalid is_shape_tensor")
{
  PerfAnalyzerParameters params{};
  params.user_data = {"fake_file.json"};
  bool is_sequence_model{false};

  std::vector<std::vector<int32_t>> expected_results;
  std::vector<ModelTensor> tensors;
  bool expect_init_failure = false;
  bool expect_thread_failure = false;

  ModelTensor model_tensor1{};
  model_tensor1.datatype_ = "INT32";
  model_tensor1.is_optional_ = true;
  model_tensor1.is_shape_tensor_ = true;
  model_tensor1.name_ = "INPUT1";
  model_tensor1.shape_ = {1};

  ModelTensor model_tensor2 = model_tensor1;
  model_tensor2.name_ = "INPUT2";

  size_t num_requests = 4;

  std::string json_str;


  const auto& ParameterizeJson{[&]() {
    SUBCASE("different data")
    {
      json_str = R"({
   "data": [
     { "INPUT1": [1], "INPUT2": [21] },
     { "INPUT1": [2], "INPUT2": [22] },
     { "INPUT1": [3], "INPUT2": [23] }
   ]})";
      expected_results = {{1, 21}, {2, 22}, {3, 23}, {1, 21}};
    }
    SUBCASE("missing data")
    {
      json_str = R"({
   "data": [
     { "INPUT2": [21] },
     { "INPUT2": [22] }
   ]})";
      expected_results = {{21}, {22}, {21}, {22}};
    }
  }};


  SUBCASE("no batching is ok")
  {
    params.batch_size = 1;
    ParameterizeJson();
  }
  SUBCASE("batching - no shm")
  {
    params.batch_size = 2;
    params.shared_memory_type = SharedMemoryType::NO_SHARED_MEMORY;
    expect_init_failure = true;
    ParameterizeJson();
  }
  SUBCASE("batching - shm")
  {
    params.batch_size = 2;
    params.shared_memory_type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
    expect_init_failure = true;
    ParameterizeJson();
  }

  TestRequestRateManager trrm(params, is_sequence_model);

  tensors.push_back(model_tensor1);
  tensors.push_back(model_tensor2);

  trrm.TestCustomData(
      num_requests, tensors, json_str, expected_results, expect_init_failure,
      expect_thread_failure);
}


TEST_CASE("custom_json_data: handling of optional tensors")
{
  PerfAnalyzerParameters params{};
  params.user_data = {"fake_file.json"};
  bool is_sequence_model{false};

  std::vector<std::vector<int32_t>> expected_results;
  std::vector<ModelTensor> tensors;
  bool expect_init_failure = false;
  bool expect_thread_failure = false;

  ModelTensor model_tensor1{};
  model_tensor1.datatype_ = "INT32";
  model_tensor1.is_optional_ = false;
  model_tensor1.is_shape_tensor_ = false;
  model_tensor1.name_ = "INPUT1";
  model_tensor1.shape_ = {1};

  ModelTensor model_tensor2 = model_tensor1;
  model_tensor2.name_ = "INPUT2";

  std::string json_str{R"({
  "data": [
    { "INPUT1": [1] },
    { "INPUT1": [2], "INPUT2": [22] },
    { "INPUT1": [3] }
  ]})"};

  size_t num_requests = 4;

  SUBCASE("normal")
  {
    model_tensor2.is_optional_ = true;
    params.batch_size = 1;
    expected_results = {{1}, {2, 22}, {3}, {1}};
  }
  SUBCASE("tensor not optional -- expect parsing fail")
  {
    model_tensor2.is_optional_ = false;
    expect_init_failure = true;
  }
  SUBCASE("shared memory not supported")
  {
    model_tensor2.is_optional_ = true;
    params.shared_memory_type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
    // FIXME: TMA-765 - Shared memory mode does not support optional inputs,
    // currently, and will be implemented in the associated story.
    expect_init_failure = true;
  }
  SUBCASE("batching with mismatching data")
  {
    model_tensor2.is_optional_ = true;
    params.batch_size = 2;
    // For batch sizes larger than 1, the same set of inputs
    // must be specified for each batch. You cannot use different
    // set of optional inputs for each individual batch.
    expect_init_failure = true;
  }

  TestRequestRateManager trrm(params, is_sequence_model);

  tensors.push_back(model_tensor1);
  tensors.push_back(model_tensor2);

  trrm.TestCustomData(
      num_requests, tensors, json_str, expected_results, expect_init_failure,
      expect_thread_failure);
}

TEST_CASE("custom_json_data: multiple streams")
{
  PerfAnalyzerParameters params{};
  params.user_data = {"fake_file.json"};
  params.num_of_sequences = 1;
  bool is_sequence_model{false};

  std::vector<std::vector<int32_t>> expected_results;
  std::vector<ModelTensor> tensors;
  bool expect_init_failure = false;
  bool expect_thread_failure = false;

  ModelTensor model_tensor1{};
  model_tensor1.datatype_ = "INT32";
  model_tensor1.is_optional_ = false;
  model_tensor1.is_shape_tensor_ = false;
  model_tensor1.name_ = "INPUT1";
  model_tensor1.shape_ = {1};

  ModelTensor model_tensor2 = model_tensor1;
  model_tensor2.name_ = "INPUT2";

  std::string json_str{R"({
  "data": [[
    { "INPUT1": [1], "INPUT2": [21] },
    { "INPUT1": [2], "INPUT2": [22] },
    { "INPUT1": [3], "INPUT2": [23] }
  ],[
    { "INPUT1": [201], "INPUT2": [221] },
    { "INPUT1": [202], "INPUT2": [222] }
  ]]})"};

  size_t num_requests = 10;

  const auto& ParameterizeMemory{[&]() {
    SUBCASE("No shared memory")
    {
      params.shared_memory_type = NO_SHARED_MEMORY;
    }
    SUBCASE("system shared memory")
    {
      params.shared_memory_type = SYSTEM_SHARED_MEMORY;
    }
    SUBCASE("cuda shared memory")
    {
      params.shared_memory_type = CUDA_SHARED_MEMORY;
    }
  }};

  SUBCASE("yes sequence")
  {
    // Sequences will randomly pick among all streams
    // (Although this test is hardcoded to pick ID 1 twice, and then ID 0
    // forever after)
    is_sequence_model = true;
    expected_results = {{201, 221}, {202, 222}, {201, 221}, {202, 222},
                        {1, 21},    {2, 22},    {3, 23},    {1, 21},
                        {2, 22},    {3, 23}};
    ParameterizeMemory();
  }
  SUBCASE("no sequence")
  {
    // For the case of no sequences, only a single data stream is supported. The
    // rest will be ignored
    is_sequence_model = false;
    expected_results = {{1, 21}, {2, 22}, {3, 23}, {1, 21}, {2, 22},
                        {3, 23}, {1, 21}, {2, 22}, {3, 23}, {1, 21}};
    ParameterizeMemory();
  }

  TestRequestRateManager trrm(params, is_sequence_model);

  tensors.push_back(model_tensor1);
  tensors.push_back(model_tensor2);

  trrm.CustomDataTestSetup(tensors, json_str, expect_init_failure);

  if (is_sequence_model) {
    // Force GetNewDataStreamId to return 1 twice and 0 every time after
    EXPECT_CALL(
        *std::dynamic_pointer_cast<MockSequenceManager>(trrm.sequence_manager_),
        GetNewDataStreamId())
        .WillOnce(testing::Return(1))
        .WillOnce(testing::Return(1))
        .WillRepeatedly(testing::Return(0));
  } else {
    // Expect that GetNewDataStreamId will never be called
    EXPECT_CALL(
        *std::dynamic_pointer_cast<MockSequenceManager>(trrm.sequence_manager_),
        GetNewDataStreamId())
        .Times(0);
  }
  auto thread_status = trrm.CustomDataTestSendRequests(num_requests);
  trrm.CustomDataTestCheckResults(
      thread_status, expect_thread_failure, expected_results);
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


  MockInputPipeline mip = TestLoadManagerBase::ProcessCustomJsonData(json_str);

  cb::MockClientStats::SharedMemoryStats expected_stats;
  SUBCASE("System shared memory usage")
  {
    params.shared_memory_type = SYSTEM_SHARED_MEMORY;
    TestRequestRateManager trrm(
        params, is_sequence, is_decoupled, use_mock_infer);

    trrm.infer_data_manager_ =
        MockInferDataManagerFactory::CreateMockInferDataManager(
            params.max_threads, params.batch_size, params.shared_memory_type,
            params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
            mip.mock_data_loader_);

    trrm.parser_ = mip.mock_model_parser_;
    trrm.data_loader_ = mip.mock_data_loader_;
    trrm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data, params.start_sequence_id, params.sequence_id_range,
        params.sequence_length, params.sequence_length_specified,
        params.sequence_length_variation);

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

    trrm.infer_data_manager_ =
        MockInferDataManagerFactory::CreateMockInferDataManager(
            params.max_threads, params.batch_size, params.shared_memory_type,
            params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
            mip.mock_data_loader_);

    trrm.parser_ = mip.mock_model_parser_;
    trrm.data_loader_ = mip.mock_data_loader_;
    trrm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data, params.start_sequence_id, params.sequence_id_range,
        params.sequence_length, params.sequence_length_specified,
        params.sequence_length_variation);

    expected_stats.num_unregister_all_shared_memory_calls = 1;
    expected_stats.num_register_cuda_shared_memory_calls = 1;
    trrm.CheckSharedMemory(expected_stats);
  }

  SUBCASE("No shared memory usage")
  {
    params.shared_memory_type = NO_SHARED_MEMORY;
    TestRequestRateManager trrm(
        params, is_sequence, is_decoupled, use_mock_infer);

    trrm.infer_data_manager_ =
        MockInferDataManagerFactory::CreateMockInferDataManager(
            params.max_threads, params.batch_size, params.shared_memory_type,
            params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
            mip.mock_data_loader_);

    trrm.parser_ = mip.mock_model_parser_;
    trrm.data_loader_ = mip.mock_data_loader_;
    trrm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data, params.start_sequence_id, params.sequence_id_range,
        params.sequence_length, params.sequence_length_specified,
        params.sequence_length_variation);

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
      TestLoadManagerBase::ProcessCustomJsonData(json_str, is_sequence_model);

  trrm.infer_data_manager_ =
      MockInferDataManagerFactory::CreateMockInferDataManager(
          params.max_threads, params.batch_size, params.shared_memory_type,
          params.output_shm_size, mip.mock_model_parser_, trrm.factory_,
          mip.mock_data_loader_);

  std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
  std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config{
      std::make_shared<RequestRateWorker::ThreadConfig>(0)};

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

  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  trrm.start_time_ = std::chrono::steady_clock::now();

  std::shared_ptr<IWorker> worker{trrm.MakeWorker(thread_stat, thread_config)};
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

  TestRequestRateManager trrm(params, is_sequence_model);
  trrm.stats_->SetDelays(delays);
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  // Sometimes have a request fail
  if (some_infer_failures) {
    trrm.stats_->SetReturnStatuses({true, true, true, false});
  }

  trrm.TestTimeouts();
}

TEST_CASE("request_rate_overhead")
{
  uint rate;
  PerfAnalyzerParameters params{};
  SUBCASE("sync, rate 10")
  {
    params.async = false;
    rate = 10;
  }
  SUBCASE("sync, rate 100")
  {
    params.async = false;
    rate = 100;
  }
  SUBCASE("async, rate 10")
  {
    params.async = true;
    rate = 10;
  }
  SUBCASE("async, rate 100")
  {
    params.async = true;
    rate = 100;
  }
  TestRequestRateManager trrm(params, false);
  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  trrm.TestOverhead(rate);
}

std::chrono::steady_clock::time_point mk_start{};

TEST_CASE(
    "send_request_rate_request_rate_manager: testing logic around detecting "
    "send request count")
{
  PerfAnalyzerParameters params{};

  std::vector<uint64_t> delays;
  bool is_sequence_model = false;
  size_t rate = 1000;
  size_t time_ms = 50;
  size_t expected_count = time_ms;

  SUBCASE("sync")
  {
    params.async = false;
    delays = {0};
  }
  SUBCASE("async - fast response")
  {
    params.async = true;
    delays = {0};
  }
  SUBCASE(
      "async - slow response with sequences off should not slow down our send "
      "rate")
  {
    params.async = true;
    delays = {100};
  }
  SUBCASE("async - slow response with sequences on")
  {
    is_sequence_model = true;
    params.async = true;
    params.num_of_sequences = 5;
    delays = {100};

    SUBCASE("send rate can be limited if serial sequences is on")
    {
      params.serial_sequences = true;
      expected_count = params.num_of_sequences;
    }
    SUBCASE(
        "send rate will not be affected by response time if serial sequences "
        "is off")
    {
      params.serial_sequences = false;
    }
  }

  TestRequestRateManager trrm(params, is_sequence_model);

  trrm.stats_->SetDelays(delays);

  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  trrm.ChangeRequestRate(rate);
  std::this_thread::sleep_for(std::chrono::milliseconds(time_ms));
  const size_t num_sent_requests{trrm.GetAndResetNumSentRequests()};
  CHECK(num_sent_requests == doctest::Approx(expected_count).epsilon(0.1));

  trrm.StopWorkerThreads();
}

TEST_CASE("request rate manager - Configure threads")
{
  PerfAnalyzerParameters params{};
  std::vector<RequestRateWorker::ThreadConfig> expected_config_values;
  std::vector<size_t> expected_number_of_sequences_owned_by_thread;
  std::vector<size_t> expected_seq_stat_index_offsets;
  bool is_sequence_model = true;
  bool is_decoupled_model = false;
  bool use_mock_infer = true;

  SUBCASE("normal")
  {
    params.max_threads = 4;
    params.num_of_sequences = 4;

    expected_number_of_sequences_owned_by_thread = {1, 1, 1, 1};
    expected_seq_stat_index_offsets = {0, 1, 2, 3};
  }

  SUBCASE("max_threads > num_seqs")
  {
    params.max_threads = 10;
    params.num_of_sequences = 4;

    expected_number_of_sequences_owned_by_thread = {1, 1, 1, 1};
    expected_seq_stat_index_offsets = {0, 1, 2, 3};
  }

  SUBCASE("num_seqs > max_threads")
  {
    params.max_threads = 4;
    params.num_of_sequences = 10;

    expected_number_of_sequences_owned_by_thread = {3, 3, 2, 2};
    expected_seq_stat_index_offsets = {0, 3, 6, 8};
  }

  SUBCASE("not divisible")
  {
    params.max_threads = 4;
    params.num_of_sequences = 7;

    expected_number_of_sequences_owned_by_thread = {2, 2, 2, 1};
    expected_seq_stat_index_offsets = {0, 2, 4, 6};
  }

  for (auto i = 0; i < expected_number_of_sequences_owned_by_thread.size();
       i++) {
    RequestRateWorker::ThreadConfig tc(i);
    tc.num_sequences_ = expected_number_of_sequences_owned_by_thread[i];
    tc.seq_stat_index_offset_ = expected_seq_stat_index_offsets[i];
    expected_config_values.push_back(tc);
  }
  TestRequestRateManager trrm(
      params, is_sequence_model, is_decoupled_model, use_mock_infer);
  trrm.TestConfigureThreads(expected_config_values);
}

TEST_CASE("request rate manager - Calculate thread ids")
{
  PerfAnalyzerParameters params{};
  bool is_sequence_model;
  bool is_decoupled_model = false;
  bool use_mock_infer = true;
  std::vector<size_t> expected_thread_ids;

  SUBCASE("normal, on sequence model")
  {
    is_sequence_model = true;
    params.max_threads = 4;
    params.num_of_sequences = 4;
    expected_thread_ids = {0, 1, 2, 3};
  }
  SUBCASE("normal, not sequence model")
  {
    is_sequence_model = false;
    params.max_threads = 4;
    params.num_of_sequences = 4;
    expected_thread_ids = {0, 1, 2, 3};
  }
  SUBCASE("num_seq > max_threads, on sequence model")
  {
    is_sequence_model = true;
    params.max_threads = 4;
    params.num_of_sequences = 5;
    expected_thread_ids = {0, 1, 2, 3, 0};
  }
  SUBCASE("num_seq > max_threads, not sequence model")
  {
    is_sequence_model = false;
    params.max_threads = 4;
    params.num_of_sequences = 5;
    expected_thread_ids = {0, 1, 2, 3};
  }
  SUBCASE("max_threads > num_seq, on sequence model")
  {
    is_sequence_model = true;
    params.max_threads = 5;
    params.num_of_sequences = 4;
    expected_thread_ids = {0, 1, 2, 3};
  }
  SUBCASE("max_threads > num_seq, not sequence model")
  {
    is_sequence_model = false;
    params.max_threads = 5;
    params.num_of_sequences = 4;
    expected_thread_ids = {0, 1, 2, 3, 4};
  }
  SUBCASE("large example")
  {
    is_sequence_model = true;
    params.max_threads = 4;
    params.num_of_sequences = 7;
    expected_thread_ids = {0, 1, 2, 3, 0, 1, 2};
  }

  TestRequestRateManager trrm(
      params, is_sequence_model, is_decoupled_model, use_mock_infer);
  trrm.TestCalculateThreadIds(expected_thread_ids);
}

TEST_CASE("request rate create schedule")
{
  PerfAnalyzerParameters params;
  params.measurement_window_ms = 1000;
  params.max_trials = 10;
  bool is_sequence_model = false;
  bool is_decoupled = false;
  bool use_mock_infer = false;
  double rate = 10;
  std::vector<uint32_t> expected_worker_ratio;

  SUBCASE("num_seq > max_threads, on sequence model, CONSTANT")
  {
    is_sequence_model = true;
    params.max_threads = 4;
    params.num_of_sequences = 5;
    expected_worker_ratio = {2, 1, 1, 1};
  }

  SUBCASE("num_seq = 7, max_threads = 4, on sequence model, CONSTANT")
  {
    is_sequence_model = true;
    params.max_threads = 4;
    params.num_of_sequences = 7;
    expected_worker_ratio = {2, 2, 2, 1};
  }

  SUBCASE("num_seq = 4, max_threads = 2, on sequence model, CONSTANT")
  {
    is_sequence_model = true;
    params.max_threads = 2;
    params.num_of_sequences = 4;
    expected_worker_ratio = {1, 1};
  }

  SUBCASE("num_seq > max_threads, on sequence model, POISSON")
  {
    is_sequence_model = true;
    params.max_threads = 4;
    params.num_of_sequences = 5;
    expected_worker_ratio = {2, 1, 1, 1};
    params.request_distribution = POISSON;
  }

  TestRequestRateManager trrm(
      params, is_sequence_model, is_decoupled, use_mock_infer);

  trrm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
  trrm.TestCreateSchedule(rate, params, expected_worker_ratio);
}
}}  // namespace triton::perfanalyzer
