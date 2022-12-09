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

#include <future>
#include "command_line_parser.h"
#include "common.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_data_loader.h"
#include "mock_model_parser.h"
#include "request_rate_manager.h"
#include "test_load_manager_base.h"
#include "test_utils.h"

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

class TestRequestRateWorker : public RequestRateWorker {
 public:
  TestRequestRateWorker(
      std::shared_ptr<ThreadStat> thread_stat,
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
            thread_stat, thread_config, parser, data_loader, backend_kind,
            factory, sequence_length, start_sequence_id, sequence_id_range,
            on_sequence_model, async, max_threads, using_json_data, streaming,
            shared_memory_type, batch_size, sequence_stat,
            shared_memory_regions, wake_signal, wake_mutex, execute,
            curr_seq_id, start_time, schedule, gen_duration, distribution)
  {
  }

  /// Mock out the InferInput object
  ///
  cb::Error CreateInferInput(
      cb::InferInput** infer_input, const cb::BackendKind kind,
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype) override
  {
    *infer_input = new cb::MockInferInput(kind, name, dims, datatype);
    return cb::Error::Success;
  }
};

class MockRequestRateWorker : public IRequestRateWorker {
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
            params.batch_size, params.measurement_window_ms, params.max_threads,
            params.num_of_sequences, params.sequence_length,
            params.shared_memory_type, params.output_shm_size,
            params.start_sequence_id, params.sequence_id_range,
            params.string_length, params.string_data, params.zero_input,
            params.user_data, GetParser(), GetFactory())
  {
  }

  std::shared_ptr<IRequestRateWorker> MakeWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config) override
  {
    if (use_mock_infer_) {
      return std::make_shared<MockRequestRateWorker>(thread_config);
    } else {
      return std::make_shared<TestRequestRateWorker>(
          thread_stat, thread_config, parser_, data_loader_, backend_->Kind(),
          RequestRateManager::factory_, sequence_length_, start_sequence_id_,
          sequence_id_range_, on_sequence_model_, async_, max_threads_,
          using_json_data_, streaming_, shared_memory_type_, batch_size_,
          sequence_stat_, shared_memory_regions_, wake_signal_, wake_mutex_,
          execute_, curr_seq_id_, start_time_, schedule_, gen_duration_,
          distribution_);
    }
  }

  /// Test the public function ResetWorkers()
  ///
  /// ResetWorkers pauses and restarts the workers, but the most important and
  /// observable effects are the following:
  ///   - if threads_ is empty, it will populate threads_, threads_stat_, and
  ///   threads_config_ based on max_threads_
  ///   - each thread config has its index reset to its ID
  ///   - each thread config has its rounds set to 0
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
      // First call will populate threads/config/stat
      //
      max_threads_ = 3;
      CHECK(start_time_ == old_time);
      ResetWorkers();
      CHECK(start_time_ != old_time);
      CHECK(threads_config_.size() == 3);
      CHECK(threads_stat_.size() == 3);
      CHECK(threads_.size() == 3);

      // Second call will reset thread_config values
      //
      for (auto& thread_config : threads_config_) {
        thread_config->index_ = 99;
        thread_config->rounds_ = 17;
      }
      old_time = start_time_;
      ResetWorkers();
      CHECK(start_time_ != old_time);
      CHECK(threads_config_.size() == 3);
      CHECK(threads_stat_.size() == 3);
      CHECK(threads_.size() == 3);
      for (auto& thread_config : threads_config_) {
        CHECK(thread_config->index_ == thread_config->id_);
        CHECK(thread_config->rounds_ == 0);
      }
    }
  }

  /// Test that the correct Infer function is called in the backend
  ///
  void TestInferType()
  {
    double request_rate = 50;
    auto sleep_time = std::chrono::milliseconds(100);

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
    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    StopWorkerThreads();

    CheckCallDistribution(request_rate);
  }

  /// Test that the schedule is properly update after calling ChangeRequestRate
  ///
  void TestMultipleRequestRate()
  {
    std::vector<double> request_rates = {50, 200};
    auto sleep_time = std::chrono::milliseconds(500);

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
    stats_->response_delay = std::chrono::milliseconds(10);
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

    auto sleep_time = std::chrono::milliseconds(sleep_ms);
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

  std::shared_ptr<ModelParser>& parser_{LoadManager::parser_};
  std::shared_ptr<DataLoader>& data_loader_{LoadManager::data_loader_};
  bool& using_json_data_{LoadManager::using_json_data_};
  bool& execute_{RequestRateManager::execute_};
  size_t& batch_size_{LoadManager::batch_size_};
  std::vector<std::chrono::nanoseconds>& schedule_{
      RequestRateManager::schedule_};
  std::shared_ptr<std::chrono::nanoseconds>& gen_duration_{
      RequestRateManager::gen_duration_};
  std::chrono::steady_clock::time_point& start_time_{
      RequestRateManager::start_time_};
  size_t& max_threads_{LoadManager::max_threads_};
  bool& async_{LoadManager::async_};
  bool& streaming_{LoadManager::streaming_};

 private:
  bool use_mock_infer_;

  void CheckCallDistribution(int request_rate)
  {
    auto request_distribution = params_.request_distribution;

    auto timestamps = GetStats()->request_timestamps;
    std::vector<int64_t> time_delays = GatherTimeBetweenRequests(timestamps);

    double delay_average = CalculateAverage(time_delays);
    double delay_variance = CalculateVariance(time_delays, delay_average);

    std::chrono::nanoseconds ns_in_one_second = std::chrono::seconds(1);
    double expected_delay_average =
        ns_in_one_second.count() / static_cast<double>(request_rate);

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
      std::chrono::nanoseconds diff_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
      time_between_requests.push_back(diff_ns.count());
    }
    return time_between_requests;
  }
};

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
  TestRequestRateManager trrm(params, false);
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
  trrm.TestDistribution(request_rate, duration_ms);
}

/// Check that the request distribution is correct
/// for the case where the measurement window is tiny
/// and thus the code will loop through the schedule
/// many times
///
TEST_CASE("request_rate_tiny_window")
{
  PerfAnalyzerParameters params;
  params.request_distribution = CONSTANT;
  params.measurement_window_ms = 10;
  uint request_rate = 500;
  uint duration_ms = 1000;


  SUBCASE("one_thread") { params.max_threads = 1; }
  SUBCASE("odd_threads") { params.max_threads = 9; }

  TestRequestRateManager trrm(params);
  trrm.TestDistribution(request_rate, duration_ms);
}

/// Check that the schedule properly handles mid-test
/// update to the request rate
///
TEST_CASE("request_rate_multiple")
{
  TestRequestRateManager trrm(PerfAnalyzerParameters{});
  trrm.TestMultipleRequestRate();
}

/// Check that the inference requests for sequences
/// follow all rules and parameters
///
TEST_CASE("request_rate_sequence")
{
  PerfAnalyzerParameters params = TestLoadManagerBase::GetSequenceTestParams();
  bool is_sequence_model = true;
  TestRequestRateManager trrm(params, is_sequence_model);
  trrm.TestSequences();
}

TEST_CASE("request_rate_streaming: test that streaming-specific logic works")
{
  PerfAnalyzerParameters params{};
  params.streaming = true;

  std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
  std::shared_ptr<RequestRateWorker::ThreadConfig> thread_config{
      std::make_shared<RequestRateWorker::ThreadConfig>(0, 0)};

  SUBCASE("enable_stats true")
  {
    TestRequestRateManager trrm(params);
    trrm.schedule_.push_back(std::chrono::nanoseconds(1));

    auto worker = trrm.MakeWorker(thread_stat, thread_config);
    std::future<void> infer_future{
        std::async(&IRequestRateWorker::Infer, worker)};

    early_exit = true;
    infer_future.get();

    CHECK(trrm.stats_->start_stream_enable_stats_value == true);
  }

  SUBCASE("enable_stats false")
  {
    TestRequestRateManager trrm(
        params, false /* is_sequence */, true /* is_decoupled */);
    trrm.schedule_.push_back(std::chrono::nanoseconds(1));

    auto worker = trrm.MakeWorker(thread_stat, thread_config);
    std::future<void> infer_future{
        std::async(&IRequestRateWorker::Infer, worker)};

    early_exit = true;
    infer_future.get();

    CHECK(trrm.stats_->start_stream_enable_stats_value == false);
  }
}

TEST_CASE(
    "custom_json_data: Check custom json data to ensure that it is processed "
    "correctly")
{
  PerfAnalyzerParameters params{};
  TestRequestRateManager trrm{params};

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

  std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
  std::shared_ptr<IRequestRateWorker::ThreadConfig> thread_config{
      std::make_shared<IRequestRateWorker::ThreadConfig>(0, 1)};

  trrm.parser_ = mmp;
  trrm.data_loader_ = mdl;
  trrm.using_json_data_ = true;
  trrm.execute_ = true;
  trrm.batch_size_ = 1;
  trrm.max_threads_ = 1;
  trrm.schedule_.push_back(std::chrono::milliseconds(4));
  trrm.schedule_.push_back(std::chrono::milliseconds(8));
  trrm.schedule_.push_back(std::chrono::milliseconds(12));
  trrm.schedule_.push_back(std::chrono::milliseconds(16));
  trrm.gen_duration_ = std::make_unique<std::chrono::nanoseconds>(16000000);
  trrm.start_time_ = std::chrono::steady_clock::now();

  SUBCASE("sync non-streaming")
  {
    trrm.async_ = false;
    trrm.streaming_ = false;
  }
  SUBCASE("async non-streaming")
  {
    trrm.async_ = true;
    trrm.streaming_ = false;
  }
  SUBCASE("async streaming")
  {
    trrm.async_ = true;
    trrm.streaming_ = true;
  }

  auto worker = trrm.MakeWorker(thread_stat, thread_config);
  std::future<void> infer_future{
      std::async(&IRequestRateWorker::Infer, worker)};

  std::this_thread::sleep_for(std::chrono::milliseconds(14));

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

}}  // namespace triton::perfanalyzer
