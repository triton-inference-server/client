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

#include "doctest.h"
#include "mock_inference_profiler.h"


namespace triton { namespace perfanalyzer {

class TestInferenceProfiler {
 public:
  static void ValidLatencyMeasurement(
      const std::pair<uint64_t, uint64_t>& valid_range,
      size_t& valid_sequence_count, size_t& delayed_request_count,
      std::vector<uint64_t>* latencies, TimestampVector& all_timestamps)
  {
    InferenceProfiler inference_profiler{};
    inference_profiler.all_timestamps_ = all_timestamps;
    inference_profiler.ValidLatencyMeasurement(
        valid_range, valid_sequence_count, delayed_request_count, latencies);
  }

  static std::tuple<uint64_t, uint64_t> GetMeanAndStdDev(
      const std::vector<uint64_t>& latencies)
  {
    InferenceProfiler inference_profiler{};
    return inference_profiler.GetMeanAndStdDev(latencies);
  }

  static bool TestCheckWithinThreshold(
      LoadStatus& ls, LoadParams& lp, uint64_t latency_threshold_ms)
  {
    InferenceProfiler ip;
    size_t idx = ls.infer_per_sec.size() - lp.stability_window;
    ip.latency_threshold_ms_ = latency_threshold_ms;

    return ip.CheckWithinThreshold(idx, ls);
  }

  static bool TestCheckWindowForStability(LoadStatus& ls, LoadParams& lp)
  {
    size_t idx = ls.infer_per_sec.size() - lp.stability_window;

    InferenceProfiler ip;
    ip.load_parameters_.stability_threshold = lp.stability_threshold;
    ip.load_parameters_.stability_window = lp.stability_window;

    return ip.CheckWindowForStability(idx, ls);
  };

  static bool TestDetermineStability(LoadStatus& ls, LoadParams& lp)
  {
    InferenceProfiler ip;
    ip.load_parameters_.stability_threshold = lp.stability_threshold;
    ip.load_parameters_.stability_window = lp.stability_window;

    return ip.DetermineStability(ls);
  }

  static bool TestIsDoneProfiling(
      LoadStatus& ls, LoadParams& lp, uint64_t latency_threshold_ms)
  {
    InferenceProfiler ip;
    ip.load_parameters_.stability_threshold = lp.stability_threshold;
    ip.load_parameters_.stability_window = lp.stability_window;
    ip.latency_threshold_ms_ = latency_threshold_ms;
    ip.mpi_driver_ = std::make_shared<triton::perfanalyzer::MPIDriver>(false);

    bool is_stable = ip.DetermineStability(ls);
    return ip.IsDoneProfiling(ls, &is_stable);
  };
};

TEST_CASE("testing the ValidLatencyMeasurement function")
{
  size_t valid_sequence_count{};
  size_t delayed_request_count{};
  std::vector<uint64_t> latencies{};

  const std::pair<uint64_t, uint64_t> window{4, 17};
  using time_point = std::chrono::time_point<std::chrono::system_clock>;
  using ns = std::chrono::nanoseconds;
  TimestampVector all_timestamps{
      // request ends before window starts, this should not be possible to exist
      // in the vector of requests, but if it is, we exclude it: not included in
      // current window
      std::make_tuple(time_point(ns(1)), time_point(ns(2)), 0, false),

      // request starts before window starts and ends inside window: included in
      // current window
      std::make_tuple(time_point(ns(3)), time_point(ns(5)), 0, false),

      // requests start and end inside window: included in current window
      std::make_tuple(time_point(ns(6)), time_point(ns(9)), 0, false),
      std::make_tuple(time_point(ns(10)), time_point(ns(14)), 0, false),

      // request starts before window ends and ends after window ends: not
      // included in current window
      std::make_tuple(time_point(ns(15)), time_point(ns(20)), 0, false),

      // request starts after window ends: not included in current window
      std::make_tuple(time_point(ns(21)), time_point(ns(27)), 0, false)};

  TestInferenceProfiler::ValidLatencyMeasurement(
      window, valid_sequence_count, delayed_request_count, &latencies,
      all_timestamps);

  const auto& convert_timestamp_to_latency{
      [](std::tuple<
          std::chrono::time_point<std::chrono::system_clock>,
          std::chrono::time_point<std::chrono::system_clock>, uint32_t, bool>
             t) {
        return CHRONO_TO_NANOS(std::get<1>(t)) -
               CHRONO_TO_NANOS(std::get<0>(t));
      }};

  CHECK(latencies.size() == 3);
  CHECK(latencies[0] == convert_timestamp_to_latency(all_timestamps[1]));
  CHECK(latencies[1] == convert_timestamp_to_latency(all_timestamps[2]));
  CHECK(latencies[2] == convert_timestamp_to_latency(all_timestamps[3]));
}

TEST_CASE("test_check_window_for_stability")
{
  LoadStatus ls;
  LoadParams lp;

  SUBCASE("test throughput not stable")
  {
    ls.infer_per_sec = {1.0, 1000.0, 500.0};
    ls.latencies = {1, 1, 1};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    CHECK(TestInferenceProfiler::TestCheckWindowForStability(ls, lp) == false);
  }
  SUBCASE("test throughput stable")
  {
    ls.infer_per_sec = {500.0, 520.0, 510.0};
    ls.latencies = {1, 1, 1};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    CHECK(TestInferenceProfiler::TestCheckWindowForStability(ls, lp) == true);
  }
  SUBCASE("test latency not stable")
  {
    ls.infer_per_sec = {500.0, 520.0, 510.0};
    ls.latencies = {100, 106, 112};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    CHECK(TestInferenceProfiler::TestCheckWindowForStability(ls, lp) == false);
  }
  SUBCASE("test latency stable")
  {
    ls.infer_per_sec = {500.0, 520.0, 510.0};
    ls.latencies = {100, 104, 108};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    CHECK(TestInferenceProfiler::TestCheckWindowForStability(ls, lp) == true);
  }
  SUBCASE("test throughput stable after many measurements")
  {
    ls.infer_per_sec = {1.0, 1000.0, 500.0, 1500.0, 500.0, 520.0, 510.0};
    ls.latencies = {1, 1, 1, 1, 1, 1, 1};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    CHECK(TestInferenceProfiler::TestCheckWindowForStability(ls, lp) == true);
  }
}

TEST_CASE("test check within threshold")
{
  LoadStatus ls;
  LoadParams lp;

  ls.infer_per_sec = {500.0, 520.0, 510.0};
  lp.stability_window = 3;
  lp.stability_threshold = 0.1;
  uint64_t latency_threshold_ms = 1;

  SUBCASE("test not within threshold")
  {
    ls.latencies = {2000000, 2000000, 2000000};
    CHECK(
        TestInferenceProfiler::TestCheckWithinThreshold(
            ls, lp, latency_threshold_ms) == false);
  }

  SUBCASE("test within threshold")
  {
    ls.latencies = {100000, 100000, 100000};
    CHECK(
        TestInferenceProfiler::TestCheckWithinThreshold(
            ls, lp, latency_threshold_ms) == true);
  }
}

TEST_CASE("test_determine_stability")
{
  LoadStatus ls;
  LoadParams lp;

  SUBCASE("test inference equals zero")
  {
    ls.infer_per_sec = {500.0, 0.0, 510.0};
    ls.latencies = {1, 1, 1};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    uint64_t latency_threshold_ms = 1;
    CHECK(TestInferenceProfiler::TestDetermineStability(ls, lp) == false);

    ls.infer_per_sec = {500.0, 520.0, 510.0};
    CHECK(TestInferenceProfiler::TestDetermineStability(ls, lp) == true);
  }
}

TEST_CASE("test_is_done_profiling")
{
  LoadStatus ls;
  LoadParams lp;


  SUBCASE("test latency_threshold is NO_LIMIT")
  {
    ls.infer_per_sec = {1.0, 1000.0, 500.0};
    ls.latencies = {1, 1, 1};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    uint64_t latency_threshold_ms = NO_LIMIT;

    CHECK(
        TestInferenceProfiler::TestIsDoneProfiling(
            ls, lp, latency_threshold_ms) == false);
  }

  SUBCASE("test not within threshold from done profiling")
  {
    ls.infer_per_sec = {1.0, 1000.0, 500.0};
    ls.latencies = {2000000, 2000000, 2000000};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    uint64_t latency_threshold_ms = 1;
    CHECK(
        TestInferenceProfiler::TestIsDoneProfiling(
            ls, lp, latency_threshold_ms) == true);
  }

  SUBCASE("test stability from is done profiling")
  {
    ls.infer_per_sec = {1.0, 1000.0, 500.0};
    ls.latencies = {1, 1, 1};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    uint64_t latency_threshold_ms = 1;

    CHECK(
        TestInferenceProfiler::TestIsDoneProfiling(
            ls, lp, latency_threshold_ms) == false);
    ls.infer_per_sec = {500.0, 520.0, 510.0};

    CHECK(
        TestInferenceProfiler::TestIsDoneProfiling(
            ls, lp, latency_threshold_ms) == true);
  }

  SUBCASE("test underflow")
  {
    ls.infer_per_sec = {500.0, 510.0};
    ls.latencies = {1, 1};
    lp.stability_window = 3;
    lp.stability_threshold = 0.1;
    uint64_t latency_threshold_ms = 1;
    CHECK(
        TestInferenceProfiler::TestIsDoneProfiling(
            ls, lp, latency_threshold_ms) == false);
  }
}

TEST_CASE("test mocking")
{
  using testing::AtLeast;
  using testing::Return;
  MockInferenceProfiler mip;

  EXPECT_CALL(mip, IncludeServerStats())
      .Times(AtLeast(1))
      .WillOnce(Return(false));

  CHECK(mip.IncludeServerStats() == false);
}

TEST_CASE("testing the GetMeanAndStdDev function")
{
  uint64_t avg_latency_ns{0};
  uint64_t std_dev_latency_us{0};

  SUBCASE("calculation using small latencies")
  {
    std::vector<uint64_t> latencies{100000, 200000, 50000};
    std::tie(avg_latency_ns, std_dev_latency_us) =
        TestInferenceProfiler::GetMeanAndStdDev(latencies);
    CHECK(avg_latency_ns == 116666);
    CHECK(std_dev_latency_us == 76);
  }

  SUBCASE("calculation using big latencies")
  {
    // Squaring these would exceed UINT64_MAX.
    std::vector<uint64_t> latencies{4300000000, 4400000000, 5000000000};
    std::tie(avg_latency_ns, std_dev_latency_us) =
        TestInferenceProfiler::GetMeanAndStdDev(latencies);
    CHECK(avg_latency_ns == 4566666666);
    CHECK(std_dev_latency_us == 378593);
  }

  SUBCASE("calculation using one latency")
  {
    // Edge case should set standard deviation to near infinity
    std::vector<uint64_t> latencies{100};
    std::tie(avg_latency_ns, std_dev_latency_us) =
        TestInferenceProfiler::GetMeanAndStdDev(latencies);
    CHECK(avg_latency_ns == 100);
    CHECK(std_dev_latency_us == UINT64_MAX);
  }
}

}}  // namespace triton::perfanalyzer
