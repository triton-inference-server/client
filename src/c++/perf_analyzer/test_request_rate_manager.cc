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
    }

    void Run(PerfAnalyzerParameters params, std::vector<int> request_rates = std::vector<int>{20, 100}, uint32_t duration_ms=500, bool is_sequence_model = false) {

      std::shared_ptr<cb::ClientBackendFactory> factory = std::make_shared<cb::MockClientBackendFactory>(&stats_);
      std::shared_ptr<ModelParser> parser = std::make_shared<MockModelParser>(is_sequence_model);

      std::unique_ptr<LoadManager> manager;
      CreateManager(params, factory, parser, &manager);

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
    void Check(PerfAnalyzerParameters params, int request_rate, std::chrono::milliseconds duration, bool first_call) {
      CheckCallCounts(params, request_rate, duration, first_call);
      CheckCallDistribution(params.request_distribution, request_rate);
    }

    void CheckCallCounts(PerfAnalyzerParameters params, int request_rate, std::chrono::milliseconds duration, bool first_call) {

      cb::MockClientStats stats = GetStats();
      cb::MockClientStats expected_stats = GetExpectedStats(params, request_rate, duration, first_call);

      // Allow 20% slop in the infer call numbers, as we can't guarentee the exact amount of 
      // time we allow the threads to run before capturing the stats
      //
      CHECK(stats.num_start_stream_calls == expected_stats.num_start_stream_calls);
      CHECK(stats.num_infer_calls == doctest::Approx(expected_stats.num_infer_calls).epsilon(0.20));
      CHECK(stats.num_async_infer_calls == doctest::Approx(expected_stats.num_async_infer_calls).epsilon(0.20));
      CHECK(stats.num_async_stream_infer_calls == doctest::Approx(expected_stats.num_async_stream_infer_calls).epsilon(0.20));
    }

    void CheckCallDistribution(Distribution request_distribution, int request_rate) {
      std::vector<int64_t> time_delays = GatherTimeBetweenRequests(GetStats().request_timestamps);

      double delay_average = CalculateAverage(time_delays);
      double delay_variance = CalculateVariance(time_delays, delay_average);

      double expected_delay_average = std::chrono::milliseconds(1000).count() / static_cast<double>(request_rate);

      if (request_distribution == POISSON) {
        // With such a small sample size for a poisson distribution, we might be far off.
        // Allow 25% slop
        //
        CHECK(delay_average == doctest::Approx(expected_delay_average).epsilon(0.25));

        // By definition, variance == average for Poisson.
        // Allow 10% slop
        //
        CHECK(delay_variance == doctest::Approx(delay_average).epsilon(0.10));        
      }
      else if (request_distribution == CONSTANT) {

        // Constant should be pretty tight. Allowing 10% slop since the same size is so small
        //
        CHECK(delay_average == doctest::Approx(expected_delay_average).epsilon(0.10));        

        // constant should in theory have 0 variance, but with thread timing
        // there is obviously some noise. 
        // Make sure it is less than 1 (millisecond)
        //
        CHECK_LT(delay_variance, 1);
      }
      else {
        CHECK(true == false);
      }
    }

    void ResetStats() {
      stats_ = cb::MockClientStats();
    }

    cb::MockClientStats stats_;

    cb::MockClientStats GetStats() {
      return stats_;
    }

    cb::MockClientStats GetExpectedStats(PerfAnalyzerParameters params, int request_rate, std::chrono::milliseconds duration, bool first_call) {
      cb::MockClientStats expected_stats;

      auto time_in_seconds = duration.count() / std::chrono::milliseconds(1000).count();
      auto num_expected_requests = request_rate * time_in_seconds;

      if (params.async) {
        if (params.streaming) {
          expected_stats.num_start_stream_calls = params.max_threads;
          expected_stats.num_async_stream_infer_calls = num_expected_requests;
        }
        else {
          expected_stats.num_async_infer_calls = num_expected_requests;
        }
      }
      else {
        expected_stats.num_infer_calls =  num_expected_requests;
        if (params.streaming) {
          expected_stats.num_start_stream_calls = params.max_threads;
        }
      }

      // Only the first pass is expected to call StartStream(). After that
      // the threads are reused
      //
      if (!first_call) {
        expected_stats.num_start_stream_calls = 0;
      }

      return expected_stats;
    }

    std::vector<int64_t> GatherTimeBetweenRequests(std::vector<std::chrono::time_point<std::chrono::system_clock>> timestamps) {
      std::vector<int64_t> time_between_requests;

      sort(timestamps.begin(), timestamps.end());

      for (size_t i = 1; i < timestamps.size(); i++) {
        auto diff = timestamps[i] - timestamps[i-1];
        std::chrono::milliseconds diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
        time_between_requests.push_back(diff_ms.count());
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

    void CreateManager(PerfAnalyzerParameters params, 
                       std::shared_ptr<cb::ClientBackendFactory> factory,
                       std::shared_ptr<ModelParser> parser,
                       std::unique_ptr<LoadManager>* manager) {
      RequestRateManager::Create(
        params.async, params.streaming, params.measurement_window_ms,
        params.request_distribution, params.batch_size,
        params.max_threads, params.num_of_sequences,
        params.sequence_length, params.string_length,
        params.string_data, params.zero_input, params.user_data,
        params.shared_memory_type, params.output_shm_size,
        params.start_sequence_id, params.sequence_id_range, parser,
        factory, manager);
    }    
};

TEST_CASE("request_rate_no_stream_no_async") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.streaming = false;
  params.async = false;
  trrm.Run(params);
}


TEST_CASE("request_rate_no_stream_no_async_poisson") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.streaming = false;
  params.async = false;
  params.request_distribution = POISSON;
  trrm.Run(params);
}

TEST_CASE("request_rate_stream_no_async") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.streaming = true;
  params.async = false;
  trrm.Run(params);
}

TEST_CASE("request_rate_no_stream_async") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.streaming = false;
  params.async = true;
  trrm.Run(params);
}

TEST_CASE("request_rate_stream_async") {
  TestRequestRateManager trrm{};
  PerfAnalyzerParameters params;
  params.streaming = true;
  params.async = true;
  trrm.Run(params);
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

TEST_CASE("test_poisson_distribution") {
  std::mt19937 schedule_rng;

  std::vector<int64_t> delays;
  double request_rate = 10000;

  std::function<std::chrono::nanoseconds(std::mt19937&)> distribution = ScheduleDistribution<Distribution::POISSON>(request_rate);

  for (int i = 0; i < 100000; i++) {
    auto delay = distribution(schedule_rng);
    delays.push_back(delay.count());
  }

  double avg = std::accumulate(delays.begin(), delays.end(), 0.0) / delays.size();

  double tmp = 0;
  for (auto delay : delays) {
    tmp += (delay - avg) * (delay - avg) / delays.size();
  }
  double variance = sqrt(tmp);

  std::chrono::nanoseconds expected_average_delay_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1) / request_rate);
  auto expected_average_delay = expected_average_delay_ns.count();
  
  // By definition, variance = mean for poisson
  auto expected_variance = expected_average_delay;

  CHECK(avg == doctest::Approx(expected_average_delay).epsilon(0.01));
  CHECK(variance == doctest::Approx(expected_average_delay).epsilon(0.01));
}

TEST_CASE("test_constant_distribution") {
  std::mt19937 schedule_rng;

  std::vector<int64_t> delays;
  double request_rate = 10000;

  std::function<std::chrono::nanoseconds(std::mt19937&)> distribution = ScheduleDistribution<Distribution::CONSTANT>(request_rate);

  for (int i = 0; i < 100000; i++) {
    auto delay = distribution(schedule_rng);
    delays.push_back(delay.count());
  }

  double avg = std::accumulate(delays.begin(), delays.end(), 0.0) / delays.size();

  double tmp = 0;
  for (auto delay : delays) {
    tmp += (delay - avg) * (delay - avg) / delays.size();
  }
  double variance = sqrt(tmp);

  std::chrono::nanoseconds expected_average_delay_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1) / request_rate);
  auto expected_average_delay = expected_average_delay_ns.count();
  auto expected_variance = 0;

  CHECK(avg == expected_average_delay);
  CHECK(variance == expected_variance);
}

}}  // namespace triton::perfanalyzer
