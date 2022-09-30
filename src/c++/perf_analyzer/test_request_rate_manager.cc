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

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

class TestRequestRateManager {
  public:
    TestRequestRateManager() {
      // Must reset this global flag every unit test. 
      // It gets set to true when we deconstruct any load manager 
      // (aka every unit test)
      //
      early_exit = false;

      stats_ = std::make_shared<cb::MockClientStats>();
    }

    /// Test that the correct Infer function is called in the backend
    ///
    void TestInferType(bool is_async, bool is_streaming) {
      PerfAnalyzerParameters params;
      params.async = is_async;
      params.streaming = is_streaming;

      double request_rate = 50;
      auto sleep_time = std::chrono::milliseconds(100);

      std::unique_ptr<LoadManager> manager = CreateManager(params);
      dynamic_cast<RequestRateManager*>(manager.get())->ChangeRequestRate(request_rate);
      std::this_thread::sleep_for(sleep_time);

      CheckInferType(params);
    }

    /// Test that the inference distribution is as expected
    ///
    void TestDistribution(PerfAnalyzerParameters params) {

      double request_rate = 500;
      auto sleep_time = std::chrono::milliseconds(2000);

      std::unique_ptr<LoadManager> manager = CreateManager(params);
      dynamic_cast<RequestRateManager*>(manager.get())->ChangeRequestRate(request_rate);
      std::this_thread::sleep_for(sleep_time);

      CheckCallDistribution(params.request_distribution, request_rate);
    }    


    /// Test that the schedule is properly update after calling ChangeRequestRate
    ///
    void TestMultipleRequestRate(PerfAnalyzerParameters params) {
      std::vector<double> request_rates = {50, 200};
      auto sleep_time = std::chrono::milliseconds(500);

      std::unique_ptr<LoadManager> manager = CreateManager(params);
      for (auto request_rate : request_rates) {
        dynamic_cast<RequestRateManager*>(manager.get())->ChangeRequestRate(request_rate);
        ResetStats();
        std::this_thread::sleep_for(sleep_time);
        CheckCallDistribution(params.request_distribution, request_rate);
      }
    }

    void Run(PerfAnalyzerParameters params, std::vector<int> request_rates = std::vector<int>{20, 100}, uint32_t duration_ms=500, bool is_sequence_model = false) {

      std::unique_ptr<LoadManager> manager = CreateManager(params, is_sequence_model);
      std::chrono::milliseconds duration  = std::chrono::milliseconds(duration_ms);

      bool first = true;
      for (auto request_rate : request_rates) {
        dynamic_cast<RequestRateManager*>(manager.get())->ChangeRequestRate(request_rate);
        ResetStats();
        std::this_thread::sleep_for(duration);
        Check(params, request_rate, duration, first);
        first = false;
      }
    }


  private:
    std::shared_ptr<cb::MockClientStats> stats_;

    std::shared_ptr<cb::MockClientStats> GetStats() {
      return stats_;
    }

    void ResetStats() {
      stats_->Reset();
    }

    std::unique_ptr<LoadManager> CreateManager(PerfAnalyzerParameters params, bool is_sequence_model=false) {

      std::unique_ptr<LoadManager> manager;
      std::shared_ptr<cb::ClientBackendFactory> factory = std::make_shared<cb::MockClientBackendFactory>(stats_);
      std::shared_ptr<ModelParser> parser = std::make_shared<MockModelParser>(is_sequence_model);

      RequestRateManager::Create(
        params.async, params.streaming, params.measurement_window_ms,
        params.request_distribution, params.batch_size,
        params.max_threads, params.num_of_sequences,
        params.sequence_length, params.string_length,
        params.string_data, params.zero_input, params.user_data,
        params.shared_memory_type, params.output_shm_size,
        params.start_sequence_id, params.sequence_id_range, parser,
        factory, &manager);

      return manager;
    }        

    void Check(PerfAnalyzerParameters params, int request_rate, std::chrono::milliseconds duration, bool first_call) {
      CheckCallCounts(params, request_rate, duration, first_call);
      CheckCallDistribution(params.request_distribution, request_rate);
    }

    void CheckCallCounts(PerfAnalyzerParameters params, int request_rate, std::chrono::milliseconds duration, bool first_call) {

      std::shared_ptr<cb::MockClientStats> stats = GetStats();
      std::shared_ptr<cb::MockClientStats> expected_stats = GetExpectedStats(params, request_rate, duration, first_call);

      // Allow 20% slop in the infer call numbers, as we can't guarentee the exact amount of 
      // time we allow the threads to run before capturing the stats
      //
      CHECK(stats->num_start_stream_calls == expected_stats->num_start_stream_calls);
      CHECK(stats->num_infer_calls == doctest::Approx(expected_stats->num_infer_calls).epsilon(0.20));
      CHECK(stats->num_async_infer_calls == doctest::Approx(expected_stats->num_async_infer_calls).epsilon(0.20));
      CHECK(stats->num_async_stream_infer_calls == doctest::Approx(expected_stats->num_async_stream_infer_calls).epsilon(0.20));
    }

    void CheckCallDistribution(Distribution request_distribution, int request_rate) {
      auto timestamps = GetStats()->request_timestamps;
      std::vector<int64_t> time_delays = GatherTimeBetweenRequests(timestamps);

      double delay_average = CalculateAverage(time_delays);
      double delay_variance = CalculateVariance(time_delays, delay_average);

      std::chrono::nanoseconds ns_in_one_second = std::chrono::seconds(1);
      double expected_delay_average = ns_in_one_second.count() / static_cast<double>(request_rate);

      if (request_distribution == POISSON) {
        // By definition, variance == average for Poisson.
        //
        // With such a small sample size for a poisson distribution, there will be noise.
        // Allow 5% slop
        //
        CHECK(delay_average == doctest::Approx(expected_delay_average).epsilon(0.05));
        CHECK(delay_variance == doctest::Approx(delay_average).epsilon(0.05));        
      }
      else if (request_distribution == CONSTANT) {
        // constant should in theory have 0 variance, but with thread timing
        // there is obviously some noise. 
        //
        // Allow it to be at most 5% of average
        //
        auto max_allowed_delay_variance = 0.05 * delay_average;

        // Constant should be pretty tight. Allowing 1% slop there is noise in the thread scheduling
        //
        CHECK(delay_average == doctest::Approx(expected_delay_average).epsilon(0.01));        
        CHECK_LT(delay_variance, max_allowed_delay_variance);
      }
      else {
        CHECK(true == false);
      }
    }

    std::shared_ptr<cb::MockClientStats> GetExpectedStats(PerfAnalyzerParameters params, int request_rate, std::chrono::milliseconds duration, bool first_call) {
      std::shared_ptr<cb::MockClientStats> expected_stats = std::make_shared<cb::MockClientStats>();

      double time_in_seconds = duration.count() / static_cast<double>(std::chrono::milliseconds(1000).count());
      auto num_expected_requests = request_rate * time_in_seconds;

      if (params.async) {
        if (params.streaming) {
          expected_stats->num_start_stream_calls = params.max_threads;
          expected_stats->num_async_stream_infer_calls = num_expected_requests;
        }
        else {
          expected_stats->num_async_infer_calls = num_expected_requests;
        }
      }
      else {
        expected_stats->num_infer_calls =  num_expected_requests;
        if (params.streaming) {
          expected_stats->num_start_stream_calls = params.max_threads;
        }
      }

      // Only the first pass is expected to call StartStream(). After that
      // the threads are reused
      //
      if (!first_call) {
        expected_stats->num_start_stream_calls = 0;
      }

      return expected_stats;
    }

    std::vector<int64_t> GatherTimeBetweenRequests(std::vector<std::chrono::time_point<std::chrono::system_clock>> timestamps) {
      std::vector<int64_t> time_between_requests;

      sort(timestamps.begin(), timestamps.end());

      for (size_t i = 1; i < timestamps.size(); i++) {
        auto diff = timestamps[i] - timestamps[i-1];
        std::chrono::nanoseconds diff_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
        time_between_requests.push_back(diff_ns.count());
      }
      return time_between_requests;
    }

    double CalculateAverage(std::vector<int64_t> values) {
      double avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
      return avg;
    }

    double CalculateVariance(std::vector<int64_t> values, double average) {
      double tmp = 0;
      for (auto value : values) {
        tmp += (value - average) * (value - average) / values.size();
      }
      double variance = sqrt(tmp);
      return variance;
    }

    void CheckInferType(PerfAnalyzerParameters params) {
      auto stats = GetStats();

      if (params.async) {
        if (params.streaming) {
          CHECK(stats->num_infer_calls == 0);
          CHECK(stats->num_async_infer_calls == 0);
          CHECK(stats->num_async_stream_infer_calls > 0);
          CHECK(stats->num_start_stream_calls > 0);
        }
        else {
          CHECK(stats->num_infer_calls == 0);
          CHECK(stats->num_async_infer_calls > 0);
          CHECK(stats->num_async_stream_infer_calls == 0);
          CHECK(stats->num_start_stream_calls == 0);          
        }
      }
      else {
        if (params.streaming) {
          CHECK(stats->num_infer_calls > 0);
          CHECK(stats->num_async_infer_calls == 0);
          CHECK(stats->num_async_stream_infer_calls == 0);
          CHECK(stats->num_start_stream_calls > 0);
        }
        else {
          CHECK(stats->num_infer_calls > 0);
          CHECK(stats->num_async_infer_calls == 0);
          CHECK(stats->num_async_stream_infer_calls == 0);
          CHECK(stats->num_start_stream_calls == 0);
        }
      }
    }    

};


/// Check that the correct inference function calls
/// are used given different param values for async and stream
///
TEST_CASE("request_rate_infer_type") {
  TestRequestRateManager trrm{};
  bool async;
  bool stream;

  SUBCASE("async_stream") {
    async=true;
    stream=true;
  }
  SUBCASE("async_no_stream") {
    async=true;
    stream=false;
  }
  SUBCASE("no_async_stream") {
    async=false;
    stream=true;
  }
  SUBCASE("no_async_no_stream") {
    async=false;
    stream=false;
  }

  trrm.TestInferType(async, stream);
}

/// Check that the request distribution is correct for
/// different Distribution types
/// 
TEST_CASE("request_rate_distribution") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;

  SUBCASE("constant") {
    params.request_distribution = CONSTANT;
  }
  SUBCASE("poisson") {
    params.request_distribution = POISSON;
  }
  trrm.TestDistribution(params);
}

/// Check that the request distribution is correct
/// for the case where the measurement window is tiny
/// and thus the code will loop through the schedule
/// many times
///
TEST_CASE("request_rate_tiny_window") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.request_distribution = CONSTANT;
  params.measurement_window_ms = 10;

  trrm.TestDistribution(params);
}

/// Check that the schedule properly handles mid-test
/// update to the request rate
///
TEST_CASE("request_rate_multiple") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  trrm.TestMultipleRequestRate(params);
}


TEST_CASE("request_rate_max_threads_1") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.streaming = false;
  params.async = false;
  params.max_threads = 1;
  trrm.Run(params);
}

TEST_CASE("request_rate_max_threads_10") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.streaming = false;
  params.async = false;
  params.max_threads = 10;
  trrm.Run(params);
}

// FIXME -- check sequence IDs?
TEST_CASE("request_rate_sequence") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.streaming = false;
  params.async = false;
  trrm.Run(params, {100}, 1000, true);
}


}}  // namespace triton::perfanalyzer
