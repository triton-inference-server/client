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

#include "command_line_parser.h"
#include "common.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_model_parser.h"
#include "request_rate_manager.h"
#include "test_utils.h"

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

/// Class to test the RequestRateManager
///
class TestRequestRateManager : public RequestRateManager {
 public:
  TestRequestRateManager()
  {
    // Must reset this global flag every unit test.
    // It gets set to true when we deconstruct any load manager
    // (aka every unit test)
    //
    early_exit = false;

    stats_ = std::make_shared<cb::MockClientStats>();
  }

  // Mock out most of the complicated Infer code
  //
  void Infer(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config) override
  {
    if (!execute_) {
      thread_config->is_paused_ = true;
    }
  }

  /// Test the public function CheckHealth
  ///
  /// It will return a bad result if any of the thread stats
  /// have a bad status or cb_status
  ///
  void TestCheckHealth()
  {
    auto good = std::make_shared<ThreadStat>();
    good->status_ = cb::Error::Success;
    good->cb_status_ = cb::Error::Success;

    auto bad_status = std::make_shared<ThreadStat>();
    bad_status->status_ = cb::Error::Failure;
    bad_status->cb_status_ = cb::Error::Success;

    auto bad_cb_status = std::make_shared<ThreadStat>();
    bad_cb_status->status_ = cb::Error::Success;
    bad_cb_status->cb_status_ = cb::Error::Failure;

    threads_stat_.clear();
    bool expect_ok = true;

    SUBCASE("Empty") { expect_ok = true; }
    SUBCASE("Good")
    {
      // Good entries: expect OK
      threads_stat_.push_back(good);
      threads_stat_.push_back(good);
      expect_ok = true;
    }
    SUBCASE("BadStatus")
    {
      // Bad Status: expect not OK
      threads_stat_.push_back(good);
      threads_stat_.push_back(bad_status);
      expect_ok = false;
    }
    SUBCASE("BadCbStatus")
    {
      // Bad cb_Status: expect not OK
      threads_stat_.push_back(bad_cb_status);
      threads_stat_.push_back(good);
      expect_ok = false;
    }
    SUBCASE("BadBothStatus")
    {
      threads_stat_.push_back(bad_status);
      threads_stat_.push_back(good);
      threads_stat_.push_back(bad_cb_status);
      expect_ok = false;
    }

    CHECK(CheckHealth().IsOk() == expect_ok);
  }

  /// Test the public function SwapTimestamps
  ///
  /// It will gather all timestamps from the thread_stats
  /// and return them, and clear the thread_stats timestamps
  ///
  void TestSwapTimeStamps()
  {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;
    using ns = std::chrono::nanoseconds;
    auto timestamp1 =
        std::make_tuple(time_point(ns(1)), time_point(ns(2)), 0, false);
    auto timestamp2 =
        std::make_tuple(time_point(ns(3)), time_point(ns(4)), 0, false);
    auto timestamp3 =
        std::make_tuple(time_point(ns(5)), time_point(ns(6)), 0, false);

    TimestampVector source_timestamps;

    SUBCASE("No threads")
    {
      auto ret = SwapTimestamps(source_timestamps);
      CHECK(source_timestamps.size() == 0);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("Source has timestamps")
    {
      // Any timestamps in the vector passed in to Swaptimestamps will
      // be dropped on the floor
      //
      source_timestamps.push_back(timestamp1);
      auto ret = SwapTimestamps(source_timestamps);
      CHECK(source_timestamps.size() == 0);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("One thread")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->request_timestamps_.push_back(timestamp1);
      stat1->request_timestamps_.push_back(timestamp2);
      stat1->request_timestamps_.push_back(timestamp3);
      threads_stat_.push_back(stat1);

      CHECK(stat1->request_timestamps_.size() == 3);
      auto ret = SwapTimestamps(source_timestamps);
      CHECK(stat1->request_timestamps_.size() == 0);

      REQUIRE(source_timestamps.size() == 3);
      CHECK(source_timestamps[0] == timestamp1);
      CHECK(source_timestamps[1] == timestamp2);
      CHECK(source_timestamps[2] == timestamp3);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("Multiple threads")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->request_timestamps_.push_back(timestamp2);

      auto stat2 = std::make_shared<ThreadStat>();
      stat2->request_timestamps_.push_back(timestamp1);
      stat2->request_timestamps_.push_back(timestamp3);

      threads_stat_.push_back(stat1);
      threads_stat_.push_back(stat2);

      CHECK(stat1->request_timestamps_.size() == 1);
      CHECK(stat2->request_timestamps_.size() == 2);
      auto ret = SwapTimestamps(source_timestamps);
      CHECK(stat1->request_timestamps_.size() == 0);
      CHECK(stat2->request_timestamps_.size() == 0);

      REQUIRE(source_timestamps.size() == 3);
      CHECK(source_timestamps[0] == timestamp2);
      CHECK(source_timestamps[1] == timestamp1);
      CHECK(source_timestamps[2] == timestamp3);
      CHECK(ret.IsOk() == true);
    }
  }

  /// Test the public function GetAccumulatedClientStat
  ///
  /// It will accumulate all contexts_stat data from all threads_stat
  ///
  void TestGetAccumulatedClientStat()
  {
    cb::InferStat result_stat;

    SUBCASE("No threads")
    {
      auto ret = GetAccumulatedClientStat(&result_stat);
      CHECK(result_stat.completed_request_count == 0);
      CHECK(result_stat.cumulative_total_request_time_ns == 0);
      CHECK(result_stat.cumulative_send_time_ns == 0);
      CHECK(result_stat.cumulative_receive_time_ns == 0);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("One thread one context stat")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->contexts_stat_.push_back(cb::InferStat());
      stat1->contexts_stat_[0].completed_request_count = 2;
      stat1->contexts_stat_[0].cumulative_total_request_time_ns = 3;
      stat1->contexts_stat_[0].cumulative_send_time_ns = 4;
      stat1->contexts_stat_[0].cumulative_receive_time_ns = 5;
      threads_stat_.push_back(stat1);

      auto ret = GetAccumulatedClientStat(&result_stat);
      CHECK(result_stat.completed_request_count == 2);
      CHECK(result_stat.cumulative_total_request_time_ns == 3);
      CHECK(result_stat.cumulative_send_time_ns == 4);
      CHECK(result_stat.cumulative_receive_time_ns == 5);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("Existing data in function arg")
    {
      // If the input InferStat already has data in it,
      // it will be included in the output result
      //
      result_stat.completed_request_count = 10;
      result_stat.cumulative_total_request_time_ns = 10;
      result_stat.cumulative_send_time_ns = 10;
      result_stat.cumulative_receive_time_ns = 10;

      auto stat1 = std::make_shared<ThreadStat>();
      stat1->contexts_stat_.push_back(cb::InferStat());
      stat1->contexts_stat_[0].completed_request_count = 2;
      stat1->contexts_stat_[0].cumulative_total_request_time_ns = 3;
      stat1->contexts_stat_[0].cumulative_send_time_ns = 4;
      stat1->contexts_stat_[0].cumulative_receive_time_ns = 5;
      threads_stat_.push_back(stat1);

      auto ret = GetAccumulatedClientStat(&result_stat);
      CHECK(result_stat.completed_request_count == 12);
      CHECK(result_stat.cumulative_total_request_time_ns == 13);
      CHECK(result_stat.cumulative_send_time_ns == 14);
      CHECK(result_stat.cumulative_receive_time_ns == 15);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("Multiple thread multiple contexts")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->contexts_stat_.push_back(cb::InferStat());
      stat1->contexts_stat_.push_back(cb::InferStat());
      stat1->contexts_stat_[0].completed_request_count = 2;
      stat1->contexts_stat_[0].cumulative_total_request_time_ns = 3;
      stat1->contexts_stat_[0].cumulative_send_time_ns = 4;
      stat1->contexts_stat_[0].cumulative_receive_time_ns = 5;
      stat1->contexts_stat_[1].completed_request_count = 3;
      stat1->contexts_stat_[1].cumulative_total_request_time_ns = 4;
      stat1->contexts_stat_[1].cumulative_send_time_ns = 5;
      stat1->contexts_stat_[1].cumulative_receive_time_ns = 6;
      threads_stat_.push_back(stat1);

      auto stat2 = std::make_shared<ThreadStat>();
      stat2->contexts_stat_.push_back(cb::InferStat());
      stat2->contexts_stat_.push_back(cb::InferStat());
      stat2->contexts_stat_[0].completed_request_count = 7;
      stat2->contexts_stat_[0].cumulative_total_request_time_ns = 8;
      stat2->contexts_stat_[0].cumulative_send_time_ns = 9;
      stat2->contexts_stat_[0].cumulative_receive_time_ns = 10;
      stat2->contexts_stat_[1].completed_request_count = 11;
      stat2->contexts_stat_[1].cumulative_total_request_time_ns = 12;
      stat2->contexts_stat_[1].cumulative_send_time_ns = 13;
      stat2->contexts_stat_[1].cumulative_receive_time_ns = 14;
      threads_stat_.push_back(stat2);

      auto ret = GetAccumulatedClientStat(&result_stat);
      // 2 + 3 + 7 + 11
      //
      CHECK(result_stat.completed_request_count == 23);
      // 3 + 4 + 8 + 12
      //
      CHECK(result_stat.cumulative_total_request_time_ns == 27);
      // 4 + 5 + 9 + 13
      //
      CHECK(result_stat.cumulative_send_time_ns == 31);
      // 5 + 6 + 10 + 14
      //
      CHECK(result_stat.cumulative_receive_time_ns == 35);

      CHECK(ret.IsOk() == true);
    }
  }

  /// Test the public function CountCollectedRequests
  ///
  /// It will count all timestamps in the thread_stats (and not modify
  /// the thread_stats in any way)
  ///
  void TestCountCollectedRequests()
  {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;
    using ns = std::chrono::nanoseconds;
    auto timestamp1 =
        std::make_tuple(time_point(ns(1)), time_point(ns(2)), 0, false);
    auto timestamp2 =
        std::make_tuple(time_point(ns(3)), time_point(ns(4)), 0, false);
    auto timestamp3 =
        std::make_tuple(time_point(ns(5)), time_point(ns(6)), 0, false);

    SUBCASE("No threads") { CHECK(CountCollectedRequests() == 0); }
    SUBCASE("One thread")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->request_timestamps_.push_back(timestamp1);
      stat1->request_timestamps_.push_back(timestamp2);
      stat1->request_timestamps_.push_back(timestamp3);
      threads_stat_.push_back(stat1);

      CHECK(stat1->request_timestamps_.size() == 3);
      CHECK(CountCollectedRequests() == 3);
      CHECK(stat1->request_timestamps_.size() == 3);
    }
    SUBCASE("Multiple threads")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->request_timestamps_.push_back(timestamp2);

      auto stat2 = std::make_shared<ThreadStat>();
      stat2->request_timestamps_.push_back(timestamp1);
      stat2->request_timestamps_.push_back(timestamp3);

      threads_stat_.push_back(stat1);
      threads_stat_.push_back(stat2);

      CHECK(stat1->request_timestamps_.size() == 1);
      CHECK(stat2->request_timestamps_.size() == 2);
      CHECK(CountCollectedRequests() == 3);
      CHECK(stat1->request_timestamps_.size() == 1);
      CHECK(stat2->request_timestamps_.size() == 2);
    }
  }

  /// Test the public function BatchSize
  ///
  /// It will just return the value passed in from the constructor
  ///
  void TestBatchSize()
  {
    PerfAnalyzerParameters params;
    int expected_value;

    SUBCASE("batch size 0")
    {
      params.batch_size = 0;
      expected_value = 0;
    }
    SUBCASE("batch size 1")
    {
      params.batch_size = 1;
      expected_value = 1;
    }
    SUBCASE("batch size 4")
    {
      params.batch_size = 4;
      expected_value = 4;
    }
    std::unique_ptr<LoadManager> manager = CreateManager(params);

    CHECK(manager->BatchSize() == expected_value);
  }

  /// Test the public function ResetWorkers()
  ///
  /// It pauses and restarts the workers, but the most important and observable
  /// effects are the following:
  ///   - if threads_ is empty, it will populate threads_, threads_stat_, and
  ///   threads_config_ based on max_threads_
  ///   - each thread config has its index reset to its ID
  ///   - each thread config has its rounds set to 0
  ///   - start_time_ is updated with a new timestamp
  ///
  void TestResetWorkers()
  {
    // Set up the schedule, factory, and parser to avoid seg faults
    //
    factory_ = std::make_shared<cb::MockClientBackendFactory>(stats_);
    parser_ = std::make_shared<MockModelParser>(false);

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
  void TestInferType(bool is_async, bool is_streaming)
  {
    PerfAnalyzerParameters params;
    params.async = is_async;
    params.streaming = is_streaming;

    double request_rate = 50;
    auto sleep_time = std::chrono::milliseconds(100);

    std::unique_ptr<LoadManager> manager = CreateManager(params);
    dynamic_cast<RequestRateManager*>(manager.get())
        ->ChangeRequestRate(request_rate);
    std::this_thread::sleep_for(sleep_time);

    // Kill the manager to stop any more requests
    //
    manager.reset();

    CheckInferType(params);
  }

  /// Test that the inference distribution is as expected
  ///
  void TestDistribution(
      PerfAnalyzerParameters params, uint request_rate, uint duration_ms)
  {
    std::unique_ptr<LoadManager> manager = CreateManager(params);
    dynamic_cast<RequestRateManager*>(manager.get())
        ->ChangeRequestRate(request_rate);
    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

    // Kill the manager to stop any more requests
    //
    manager.reset();

    CheckCallDistribution(params.request_distribution, request_rate);
  }

  /// Test that the schedule is properly update after calling ChangeRequestRate
  ///
  void TestMultipleRequestRate(PerfAnalyzerParameters params)
  {
    std::vector<double> request_rates = {50, 200};
    auto sleep_time = std::chrono::milliseconds(500);

    std::unique_ptr<LoadManager> manager = CreateManager(params);
    for (auto request_rate : request_rates) {
      dynamic_cast<RequestRateManager*>(manager.get())
          ->ChangeRequestRate(request_rate);
      ResetStats();
      std::this_thread::sleep_for(sleep_time);
      CheckCallDistribution(params.request_distribution, request_rate);
    }
  }

  /// Test sequence handling
  ///
  void TestSequences(PerfAnalyzerParameters params)
  {
    bool is_sequence_model = true;
    std::unique_ptr<LoadManager> manager =
        CreateManager(params, is_sequence_model);

    double request_rate = 200;
    auto sleep_time = std::chrono::milliseconds(500);

    dynamic_cast<RequestRateManager*>(manager.get())
        ->ChangeRequestRate(request_rate);
    std::this_thread::sleep_for(sleep_time);

    // FIXME - it would be nice to call manager.reset() here
    // before checking the results to explicitly stop the load manager from
    // sending any more requests. However, the result is that all partially
    // completed sequences are immediately finished, which results in a number
    // of sequences being shorter than 'expected'.
    //
    CheckSequences(params);
  }

 private:
  std::shared_ptr<cb::MockClientStats> stats_;

  std::shared_ptr<cb::MockClientStats> GetStats() { return stats_; }

  void ResetStats() { stats_->Reset(); }

  void CheckInferType(PerfAnalyzerParameters params)
  {
    auto stats = GetStats();

    if (params.async) {
      if (params.streaming) {
        CHECK(stats->num_infer_calls == 0);
        CHECK(stats->num_async_infer_calls == 0);
        CHECK(stats->num_async_stream_infer_calls > 0);
        CHECK(stats->num_start_stream_calls > 0);
      } else {
        CHECK(stats->num_infer_calls == 0);
        CHECK(stats->num_async_infer_calls > 0);
        CHECK(stats->num_async_stream_infer_calls == 0);
        CHECK(stats->num_start_stream_calls == 0);
      }
    } else {
      if (params.streaming) {
        CHECK(stats->num_infer_calls > 0);
        CHECK(stats->num_async_infer_calls == 0);
        CHECK(stats->num_async_stream_infer_calls == 0);
        CHECK(stats->num_start_stream_calls > 0);
      } else {
        CHECK(stats->num_infer_calls > 0);
        CHECK(stats->num_async_infer_calls == 0);
        CHECK(stats->num_async_stream_infer_calls == 0);
        CHECK(stats->num_start_stream_calls == 0);
      }
    }
  }

  void CheckCallDistribution(
      Distribution request_distribution, int request_rate)
  {
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

  void CheckSequences(PerfAnalyzerParameters params)
  {
    auto stats = GetStats();

    // Make sure all seq IDs are within range
    //
    for (auto seq_id : stats->sequence_status.used_seq_ids) {
      CHECK(seq_id >= params.start_sequence_id);
      CHECK(seq_id <= params.start_sequence_id + params.sequence_id_range);
    }

    // Make sure that we had the correct number of concurrently live sequences
    //
    // If the sequence length is only 1 then there is nothing to check because
    // there are never any overlapping requests -- they always immediately exit
    //
    if (params.sequence_length != 1) {
      CHECK(
          params.num_of_sequences == stats->sequence_status.max_live_seq_count);
    }

    // Make sure that the length of each sequence is as expected
    // (The code explicitly has a 20% slop, so that is what we are checking)
    //
    for (auto len : stats->sequence_status.seq_lengths) {
      CHECK(len == doctest::Approx(params.sequence_length).epsilon(0.20));
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

  double CalculateAverage(std::vector<int64_t> values)
  {
    double avg =
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return avg;
  }

  double CalculateVariance(std::vector<int64_t> values, double average)
  {
    double tmp = 0;
    for (auto value : values) {
      tmp += (value - average) * (value - average) / values.size();
    }
    double variance = sqrt(tmp);
    return variance;
  }

  std::unique_ptr<LoadManager> CreateManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false)
  {
    std::unique_ptr<LoadManager> manager;
    std::shared_ptr<cb::ClientBackendFactory> factory =
        std::make_shared<cb::MockClientBackendFactory>(stats_);
    std::shared_ptr<ModelParser> parser =
        std::make_shared<MockModelParser>(is_sequence_model);

    RequestRateManager::Create(
        params.async, params.streaming, params.measurement_window_ms,
        params.request_distribution, params.batch_size, params.max_threads,
        params.num_of_sequences, params.sequence_length, params.string_length,
        params.string_data, params.zero_input, params.user_data,
        params.shared_memory_type, params.output_shm_size,
        params.start_sequence_id, params.sequence_id_range, parser, factory,
        &manager);

    return manager;
  }
};

TEST_CASE("request_rate_check_health: Test the public function CheckHealth()")
{
  TestRequestRateManager trrm{};
  trrm.TestCheckHealth();
}

TEST_CASE(
    "request_rate_swap_timestamps: Test the public function SwapTimeStamps()")
{
  TestRequestRateManager trrm{};
  trrm.TestSwapTimeStamps();
}

TEST_CASE(
    "request_rate_get_accumulated_client_stat: Test the public function "
    "GetAccumulatedClientStat()")
{
  TestRequestRateManager trrm{};
  trrm.TestGetAccumulatedClientStat();
}

TEST_CASE(
    "request_rate_count_collected_requests: Test the public function "
    "CountCollectedRequests()")
{
  TestRequestRateManager trrm{};
  trrm.TestCountCollectedRequests();
}

TEST_CASE("request_rate_batch_size: Test the public function BatchSize()")
{
  TestRequestRateManager trrm{};
  trrm.TestBatchSize();
}

TEST_CASE("request_rate_reset_workers: Test the public function ResetWorkers()")
{
  TestRequestRateManager trrm{};
  trrm.TestResetWorkers();
}

/// Check that the correct inference function calls
/// are used given different param values for async and stream
///
TEST_CASE("request_rate_infer_type")
{
  TestRequestRateManager trrm{};
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

  trrm.TestInferType(async, stream);
}

/// Check that the request distribution is correct for
/// different Distribution types
///
TEST_CASE("request_rate_distribution")
{
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  uint request_rate = 500;
  uint duration_ms = 1000;

  SUBCASE("constant") { params.request_distribution = CONSTANT; }
  SUBCASE("poisson") { params.request_distribution = POISSON; }
  trrm.TestDistribution(params, request_rate, duration_ms);
}

/// Check that the request distribution is correct
/// for the case where the measurement window is tiny
/// and thus the code will loop through the schedule
/// many times
///
TEST_CASE("request_rate_tiny_window")
{
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.request_distribution = CONSTANT;
  params.measurement_window_ms = 10;
  uint request_rate = 500;
  uint duration_ms = 1000;


  SUBCASE("one_thread") { params.max_threads = 1; }
  SUBCASE("odd_threads") { params.max_threads = 9; }
  trrm.TestDistribution(params, request_rate, duration_ms);
}

/// Check that the schedule properly handles mid-test
/// update to the request rate
///
TEST_CASE("request_rate_multiple")
{
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  trrm.TestMultipleRequestRate(params);
}

/// Check that the inference requests for sequences
/// follow all rules and parameters
///
TEST_CASE("request_rate_sequence")
{
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;

  // Generally we want short sequences for testing
  // so we can hit the corner cases more often
  //
  params.sequence_length = 3;

  SUBCASE("Normal") {}
  SUBCASE("sequence IDs 1")
  {
    params.start_sequence_id = 1;
    params.sequence_id_range = 5;
  }
  SUBCASE("sequence IDs 2")
  {
    params.start_sequence_id = 17;
    params.sequence_id_range = 8;
  }
  SUBCASE("num_of_sequences 1") { params.num_of_sequences = 1; }
  SUBCASE("num_of_sequences 10")
  {
    params.num_of_sequences = 10;
    // Make sequences longer so we actually get 10 in flight at a time
    params.sequence_length = 8;
  }
  SUBCASE("sequence_length 1") { params.sequence_length = 1; }
  SUBCASE("sequence_length 10") { params.sequence_length = 10; }

  trrm.TestSequences(params);
}

}}  // namespace triton::perfanalyzer
