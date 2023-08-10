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

#include "doctest.h"
#include "inference_profiler.h"
#include "mock_inference_profiler.h"
#include "mock_load_manager.h"
#include "mock_model_parser.h"

namespace triton { namespace perfanalyzer {

class TestInferenceProfiler : public InferenceProfiler {
 public:
  static void ValidLatencyMeasurement(
      const std::pair<uint64_t, uint64_t>& valid_range,
      size_t& valid_sequence_count, size_t& delayed_request_count,
      std::vector<uint64_t>* latencies, size_t& response_count,
      std::vector<RequestRecord>& valid_requests,
      std::vector<RequestRecord>& all_request_records)
  {
    InferenceProfiler inference_profiler{};
    inference_profiler.all_request_records_ = all_request_records;
    inference_profiler.ValidLatencyMeasurement(
        valid_range, valid_sequence_count, delayed_request_count, latencies,
        response_count, valid_requests);
  }

  static std::tuple<uint64_t, uint64_t> GetMeanAndStdDev(
      const std::vector<uint64_t>& latencies)
  {
    InferenceProfiler inference_profiler{};
    return inference_profiler.GetMeanAndStdDev(latencies);
  }

  void SummarizeSendRequestRate(
      const double window_duration_s, const size_t num_sent_requests,
      PerfStatus& summary)
  {
    InferenceProfiler::SummarizeSendRequestRate(
        window_duration_s, num_sent_requests, summary);
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

  cb::Error MergeMetrics(
      const std::vector<std::reference_wrapper<const Metrics>>& all_metrics,
      Metrics& merged_metrics)
  {
    return InferenceProfiler::MergeMetrics(all_metrics, merged_metrics);
  }

  template <typename T>
  void GetMetricAveragePerGPU(
      const std::vector<std::reference_wrapper<const std::map<std::string, T>>>&
          input_metric_maps,
      std::map<std::string, T>& output_metric_map)
  {
    InferenceProfiler::GetMetricAveragePerGPU<T>(
        input_metric_maps, output_metric_map);
  }

  template <typename T>
  void GetMetricMaxPerGPU(
      const std::vector<std::reference_wrapper<const std::map<std::string, T>>>&
          input_metric_maps,
      std::map<std::string, T>& output_metric_map)
  {
    InferenceProfiler::GetMetricMaxPerGPU<T>(
        input_metric_maps, output_metric_map);
  }

  template <typename T>
  void GetMetricFirstPerGPU(
      const std::vector<std::reference_wrapper<const std::map<std::string, T>>>&
          input_metric_maps,
      std::map<std::string, T>& output_metric_map)
  {
    InferenceProfiler::GetMetricFirstPerGPU<T>(
        input_metric_maps, output_metric_map);
  }

  void SummarizeOverhead(
      const uint64_t window_duration_ns, const uint64_t idle_ns,
      PerfStatus& summary)
  {
    InferenceProfiler::SummarizeOverhead(window_duration_ns, idle_ns, summary);
  }


  cb::Error DetermineStatsModelVersion(
      const cb::ModelIdentifier& model_identifier,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_stats,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_stats,
      int64_t* model_version)
  {
    return InferenceProfiler::DetermineStatsModelVersion(
        model_identifier, start_stats, end_stats, model_version);
  }
};

TEST_CASE("testing the ValidLatencyMeasurement function")
{
  size_t valid_sequence_count{};
  size_t delayed_request_count{};
  std::vector<uint64_t> latencies{};
  size_t response_count{};
  std::vector<RequestRecord> valid_requests{};

  const std::pair<uint64_t, uint64_t> window{4, 17};
  using time_point = std::chrono::time_point<std::chrono::system_clock>;
  using ns = std::chrono::nanoseconds;
  std::vector<RequestRecord> all_request_records{
      // request ends before window starts, this should not be possible to exist
      // in the vector of requests, but if it is, we exclude it: not included in
      // current window
      RequestRecord(
          time_point(ns(1)), std::vector<time_point>{time_point(ns(2))}, 0,
          false, 0),

      // request starts before window starts and ends inside window: included in
      // current window
      RequestRecord(
          time_point(ns(3)), std::vector<time_point>{time_point(ns(5))}, 0,
          false, 0),

      // requests start and end inside window: included in current window
      RequestRecord(
          time_point(ns(6)), std::vector<time_point>{time_point(ns(9))}, 0,
          false, 0),
      RequestRecord(
          time_point(ns(10)), std::vector<time_point>{time_point(ns(14))}, 0,
          false, 0),

      // request starts before window ends and ends after window ends: not
      // included in current window
      RequestRecord(
          time_point(ns(15)), std::vector<time_point>{time_point(ns(20))}, 0,
          false, 0),

      // request starts after window ends: not included in current window
      RequestRecord(
          time_point(ns(21)), std::vector<time_point>{time_point(ns(27))}, 0,
          false, 0)};

  TestInferenceProfiler::ValidLatencyMeasurement(
      window, valid_sequence_count, delayed_request_count, &latencies,
      response_count, valid_requests, all_request_records);

  const auto& convert_request_record_to_latency{[](RequestRecord t) {
    return CHRONO_TO_NANOS(t.response_times_.back()) -
           CHRONO_TO_NANOS(t.start_time_);
  }};

  CHECK(latencies.size() == 3);
  CHECK(
      latencies[0] ==
      convert_request_record_to_latency(all_request_records[1]));
  CHECK(
      latencies[1] ==
      convert_request_record_to_latency(all_request_records[2]));
  CHECK(
      latencies[2] ==
      convert_request_record_to_latency(all_request_records[3]));
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
  SUBCASE("test stability window of 5")
  {
    ls.infer_per_sec = {500.0, 520.0, 510.0, 505.0, 515.0};
    ls.latencies = {100, 104, 108, 102, 106};
    lp.stability_window = 5;
    lp.stability_threshold = 0.1;
    CHECK(TestInferenceProfiler::TestCheckWindowForStability(ls, lp) == true);
  }
  SUBCASE("test not stable in 5 but stable in 3")
  {
    ls.infer_per_sec = {1.0, 1000.0, 510.0, 505.0, 515.0};
    ls.latencies = {100, 104, 108, 102, 106};
    lp.stability_window = 5;
    lp.stability_threshold = 0.1;
    CHECK(TestInferenceProfiler::TestCheckWindowForStability(ls, lp) == false);
  }
  SUBCASE("test stability window of 2")
  {
    ls.infer_per_sec = {500.0, 1000.0, 1.0, 505.0, 515.0};
    ls.latencies = {100, 104, 108, 102, 106};
    lp.stability_window = 2;
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

TEST_CASE("testing the MergeMetrics function")
{
  TestInferenceProfiler tip{};
  Metrics metrics_1{}, metrics_2{}, merged_metrics{};

  SUBCASE("all metrics present")
  {
    metrics_1.gpu_utilization_per_gpu["gpu0"] = 0.45;
    metrics_2.gpu_utilization_per_gpu["gpu0"] = 0.52;

    metrics_1.gpu_power_usage_per_gpu["gpu0"] = 70.0;
    metrics_2.gpu_power_usage_per_gpu["gpu0"] = 84.5;

    metrics_1.gpu_memory_used_bytes_per_gpu["gpu0"] = 10000;
    metrics_2.gpu_memory_used_bytes_per_gpu["gpu0"] = 12000;

    metrics_1.gpu_memory_total_bytes_per_gpu["gpu0"] = 100000;
    metrics_2.gpu_memory_total_bytes_per_gpu["gpu0"] = 100000;

    const std::vector<std::reference_wrapper<const Metrics>> all_metrics{
        metrics_1, metrics_2};

    tip.MergeMetrics(all_metrics, merged_metrics);
    CHECK(merged_metrics.gpu_utilization_per_gpu.size() == 1);
    CHECK(merged_metrics.gpu_power_usage_per_gpu.size() == 1);
    CHECK(merged_metrics.gpu_memory_used_bytes_per_gpu.size() == 1);
    CHECK(merged_metrics.gpu_memory_total_bytes_per_gpu.size() == 1);
    CHECK(
        merged_metrics.gpu_utilization_per_gpu["gpu0"] ==
        doctest::Approx(0.485));
    CHECK(
        merged_metrics.gpu_power_usage_per_gpu["gpu0"] ==
        doctest::Approx(77.25));
    CHECK(merged_metrics.gpu_memory_used_bytes_per_gpu["gpu0"] == 12000);
    CHECK(merged_metrics.gpu_memory_total_bytes_per_gpu["gpu0"] == 100000);
  }

  SUBCASE("missing multiple metrics")
  {
    metrics_1.gpu_utilization_per_gpu["gpu0"] = 0.45;
    metrics_2.gpu_utilization_per_gpu["gpu0"] = 0.52;

    metrics_1.gpu_memory_used_bytes_per_gpu["gpu0"] = 10000;
    metrics_2.gpu_memory_used_bytes_per_gpu["gpu0"] = 12000;

    const std::vector<std::reference_wrapper<const Metrics>> all_metrics{
        metrics_1, metrics_2};

    tip.MergeMetrics(all_metrics, merged_metrics);
    CHECK(merged_metrics.gpu_utilization_per_gpu.size() == 1);
    CHECK(merged_metrics.gpu_power_usage_per_gpu.size() == 0);
    CHECK(merged_metrics.gpu_memory_used_bytes_per_gpu.size() == 1);
    CHECK(merged_metrics.gpu_memory_total_bytes_per_gpu.size() == 0);
    CHECK(
        merged_metrics.gpu_utilization_per_gpu["gpu0"] ==
        doctest::Approx(0.485));
    CHECK(merged_metrics.gpu_memory_used_bytes_per_gpu["gpu0"] == 12000);
  }
}

TEST_CASE("testing the GetMetricAveragePerGPU function")
{
  TestInferenceProfiler tip{};
  std::map<std::string, double> metric_averages{};

  SUBCASE("all GPUs present")
  {
    const std::map<std::string, double> metric_1{
        {"gpu0", 0.45}, {"gpu1", 0.23}},
        metric_2{{"gpu0", 0.52}, {"gpu1", 0.27}},
        metric_3{{"gpu0", 0.56}, {"gpu1", 0.30}};

    const std::vector<
        std::reference_wrapper<const std::map<std::string, double>>>
        all_metrics{metric_1, metric_2, metric_3};

    tip.GetMetricAveragePerGPU<double>(all_metrics, metric_averages);

    CHECK(metric_averages.size() == 2);
    CHECK(metric_averages["gpu0"] == doctest::Approx(0.51));
    CHECK(metric_averages["gpu1"] == doctest::Approx(0.26666));
  }

  SUBCASE("missing one GPU from one metric")
  {
    const std::map<std::string, double> metric_1{
        {"gpu0", 0.45}, {"gpu1", 0.23}},
        metric_2{{"gpu0", 0.52}}, metric_3{{"gpu0", 0.56}, {"gpu1", 0.30}};

    const std::vector<
        std::reference_wrapper<const std::map<std::string, double>>>
        all_metrics{metric_1, metric_2, metric_3};

    tip.GetMetricAveragePerGPU<double>(all_metrics, metric_averages);

    CHECK(metric_averages.size() == 2);
    CHECK(metric_averages["gpu0"] == doctest::Approx(0.51));
    CHECK(metric_averages["gpu1"] == doctest::Approx(0.265));
  }
}

TEST_CASE("testing the GetMetricMaxPerGPU function")
{
  TestInferenceProfiler tip{};
  std::map<std::string, uint64_t> metric_maxes{};

  SUBCASE("all GPUs present")
  {
    const std::map<std::string, uint64_t> metric_1{{"gpu0", 10}, {"gpu1", 55}},
        metric_2{{"gpu0", 12}, {"gpu1", 84}},
        metric_3{{"gpu0", 15}, {"gpu1", 47}};

    const std::vector<
        std::reference_wrapper<const std::map<std::string, uint64_t>>>
        all_metrics{metric_1, metric_2, metric_3};

    tip.GetMetricMaxPerGPU<uint64_t>(all_metrics, metric_maxes);

    CHECK(metric_maxes.size() == 2);
    CHECK(metric_maxes["gpu0"] == 15);
    CHECK(metric_maxes["gpu1"] == 84);
  }

  SUBCASE("missing one GPU from one metric")
  {
    const std::map<std::string, uint64_t> metric_1{{"gpu0", 10}, {"gpu1", 55}},
        metric_2{{"gpu0", 12}}, metric_3{{"gpu0", 15}, {"gpu1", 47}};

    const std::vector<
        std::reference_wrapper<const std::map<std::string, uint64_t>>>
        all_metrics{metric_1, metric_2, metric_3};

    tip.GetMetricMaxPerGPU<uint64_t>(all_metrics, metric_maxes);

    CHECK(metric_maxes.size() == 2);
    CHECK(metric_maxes["gpu0"] == 15);
    CHECK(metric_maxes["gpu1"] == 55);
  }
}

TEST_CASE("testing the GetMetricFirstPerGPU function")
{
  TestInferenceProfiler tip{};
  std::map<std::string, uint64_t> metric_firsts{};

  SUBCASE("all GPUs present")
  {
    const std::map<std::string, uint64_t> metric_1{{"gpu0", 10}, {"gpu1", 55}},
        metric_2{{"gpu0", 12}, {"gpu1", 84}},
        metric_3{{"gpu0", 15}, {"gpu1", 47}};

    const std::vector<
        std::reference_wrapper<const std::map<std::string, uint64_t>>>
        all_metrics{metric_1, metric_2, metric_3};

    tip.GetMetricFirstPerGPU<uint64_t>(all_metrics, metric_firsts);

    CHECK(metric_firsts.size() == 2);
    CHECK(metric_firsts["gpu0"] == 10);
    CHECK(metric_firsts["gpu1"] == 55);
  }

  SUBCASE("missing one GPU from one metric")
  {
    const std::map<std::string, uint64_t> metric_1{{"gpu0", 10}},
        metric_2{{"gpu0", 12}, {"gpu1", 84}},
        metric_3{{"gpu0", 15}, {"gpu1", 47}};

    const std::vector<
        std::reference_wrapper<const std::map<std::string, uint64_t>>>
        all_metrics{metric_1, metric_2, metric_3};

    tip.GetMetricFirstPerGPU<uint64_t>(all_metrics, metric_firsts);

    CHECK(metric_firsts.size() == 2);
    CHECK(metric_firsts["gpu0"] == 10);
    CHECK(metric_firsts["gpu1"] == 84);
  }
}

TEST_CASE("test the ReportPrometheusMetrics function")
{
  Metrics metrics{};
  std::stringstream captured_cout;
  std::streambuf* old_cout{std::cout.rdbuf(captured_cout.rdbuf())};

  SUBCASE("regular output")
  {
    metrics.gpu_utilization_per_gpu["gpu0"] = 0.45;
    metrics.gpu_utilization_per_gpu["gpu1"] = 0.52;

    metrics.gpu_power_usage_per_gpu["gpu0"] = 70.0;
    metrics.gpu_power_usage_per_gpu["gpu1"] = 84.5;

    metrics.gpu_memory_used_bytes_per_gpu["gpu0"] = 10000;
    metrics.gpu_memory_used_bytes_per_gpu["gpu1"] = 12000;

    metrics.gpu_memory_total_bytes_per_gpu["gpu0"] = 100000;
    metrics.gpu_memory_total_bytes_per_gpu["gpu1"] = 100000;

    cb::Error result{ReportPrometheusMetrics(metrics)};

    std::cout.rdbuf(old_cout);

    CHECK(result.Err() == SUCCESS);
    CHECK(
        captured_cout.str() ==
        "    Avg GPU Utilization:\n"
        "      gpu0 : 45%\n"
        "      gpu1 : 52%\n"
        "    Avg GPU Power Usage:\n"
        "      gpu0 : 70 watts\n"
        "      gpu1 : 84.5 watts\n"
        "    Max GPU Memory Usage:\n"
        "      gpu0 : 10000 bytes\n"
        "      gpu1 : 12000 bytes\n"
        "    Total GPU Memory:\n"
        "      gpu0 : 100000 bytes\n"
        "      gpu1 : 100000 bytes\n");
  }

  SUBCASE("too many GPUs")
  {
    const size_t num_gpus{17};
    for (size_t gpu_idx{0}; gpu_idx < num_gpus; gpu_idx++) {
      const auto& gpu_key{"gpu" + std::to_string(gpu_idx)};
      metrics.gpu_utilization_per_gpu[gpu_key] = 0.5;
      metrics.gpu_power_usage_per_gpu[gpu_key] = 75.5;
      metrics.gpu_memory_used_bytes_per_gpu[gpu_key] = 12500;
      metrics.gpu_memory_total_bytes_per_gpu[gpu_key] = 150000;
    }

    cb::Error result{ReportPrometheusMetrics(metrics)};

    std::cout.rdbuf(old_cout);

    CHECK(result.Err() == SUCCESS);
    CHECK(
        captured_cout.str() ==
        "Too many GPUs on system to print out individual Prometheus metrics, "
        "use the CSV output feature to see metrics.\n");
  }
}

TEST_CASE("InferenceProfiler: Test SummarizeOverhead")
{
  TestInferenceProfiler tip{};
  PerfStatus status;
  SUBCASE("normal")
  {
    tip.SummarizeOverhead(100, 63, status);
    CHECK(status.overhead_pct == doctest::Approx(37));
  }
  SUBCASE("normal 2")
  {
    tip.SummarizeOverhead(234, 56, status);
    CHECK(status.overhead_pct == doctest::Approx(76.068));
  }
  SUBCASE("overflow")
  {
    tip.SummarizeOverhead(100, 101, status);
    CHECK(status.overhead_pct == doctest::Approx(0));
  }
}

TEST_CASE(
    "summarize_send_request_rate: testing the SummarizeSendRequestRate "
    "function")
{
  TestInferenceProfiler tip{};
  PerfStatus perf_status;

  SUBCASE("invalid zero window duration")
  {
    double window_duration_s{0.0};
    size_t num_sent_requests{0};
    CHECK_THROWS_WITH_AS(
        tip.SummarizeSendRequestRate(
            window_duration_s, num_sent_requests, perf_status),
        "window_duration_s must be positive", std::runtime_error);
  }

  SUBCASE("invalid negative window duration")
  {
    double window_duration_s{-1.0};
    size_t num_sent_requests{0};
    CHECK_THROWS_WITH_AS(
        tip.SummarizeSendRequestRate(
            window_duration_s, num_sent_requests, perf_status),
        "window_duration_s must be positive", std::runtime_error);
  }

  SUBCASE("regular case")
  {
    double window_duration_s{2.0};
    size_t num_sent_requests{100};
    tip.SummarizeSendRequestRate(
        window_duration_s, num_sent_requests, perf_status);
    CHECK(perf_status.send_request_rate == doctest::Approx(50));
  }
}

TEST_CASE("determine_stats_model_version: testing DetermineStatsModelVersion()")
{
  TestInferenceProfiler tip{};
  cb::ModelIdentifier model_identifier;
  cb::ModelStatistics old_stats;
  cb::ModelStatistics new_stats;
  old_stats.queue_count_ = 1;
  new_stats.queue_count_ = 2;

  int64_t expected_model_version;
  bool expect_warning = false;
  bool expect_exception = false;

  std::map<cb::ModelIdentifier, cb::ModelStatistics> start_stats_map;
  std::map<cb::ModelIdentifier, cb::ModelStatistics> end_stats_map;

  SUBCASE("One entry - unspecified - valid and in start")
  {
    model_identifier = {"ModelA", ""};
    start_stats_map.insert({{"ModelA", "3"}, old_stats});
    end_stats_map.insert({{"ModelA", "3"}, new_stats});
    expected_model_version = 3;
  }
  SUBCASE("One entry - unspecified - valid and not in start")
  {
    model_identifier = {"ModelA", ""};
    end_stats_map.insert({{"ModelA", "3"}, new_stats});
    expected_model_version = 3;
  }
  SUBCASE("One entry - unspecified - invalid")
  {
    model_identifier = {"ModelA", ""};
    start_stats_map.insert({{"ModelA", "3"}, old_stats});
    end_stats_map.insert({{"ModelA", "3"}, old_stats});
    expect_exception = true;
    expected_model_version = -1;
  }
  SUBCASE("One entry - match")
  {
    model_identifier = {"ModelA", "3"};
    end_stats_map.insert({{"ModelA", "3"}, new_stats});
    expected_model_version = 3;
  }
  SUBCASE("One entry - miss")
  {
    model_identifier = {"ModelA", "2"};
    end_stats_map.insert({{"ModelA", "3"}, new_stats});
    expect_exception = true;
    expected_model_version = -1;
  }
  SUBCASE("Two entries - unspecified case 1")
  {
    model_identifier = {"ModelA", ""};
    start_stats_map.insert({{"ModelA", "3"}, old_stats});
    start_stats_map.insert({{"ModelA", "4"}, old_stats});
    end_stats_map.insert({{"ModelA", "3"}, new_stats});
    end_stats_map.insert({{"ModelA", "4"}, old_stats});
    expected_model_version = 3;
  }
  SUBCASE("Two entries - unspecified case 2")
  {
    model_identifier = {"ModelA", ""};
    start_stats_map.insert({{"ModelA", "3"}, old_stats});
    start_stats_map.insert({{"ModelA", "4"}, old_stats});
    end_stats_map.insert({{"ModelA", "3"}, old_stats});
    end_stats_map.insert({{"ModelA", "4"}, new_stats});
    expected_model_version = 4;
  }
  SUBCASE("Two entries - unspecified case 3")
  {
    model_identifier = {"ModelA", ""};
    start_stats_map.insert({{"ModelA", "3"}, old_stats});
    start_stats_map.insert({{"ModelA", "4"}, old_stats});
    end_stats_map.insert({{"ModelA", "3"}, new_stats});
    end_stats_map.insert({{"ModelA", "4"}, new_stats});
    expected_model_version = 4;
    expect_warning = 1;
  }
  SUBCASE("Two entries - specified hit")
  {
    model_identifier = {"ModelA", "3"};
    end_stats_map.insert({{"ModelA", "3"}, old_stats});
    end_stats_map.insert({{"ModelA", "4"}, old_stats});
    expected_model_version = 3;
  }
  SUBCASE("Two entries - specified miss")
  {
    model_identifier = {"ModelA", "2"};
    end_stats_map.insert({{"ModelA", "3"}, old_stats});
    end_stats_map.insert({{"ModelA", "4"}, old_stats});
    expected_model_version = -1;
    expect_exception = true;
  }


  std::stringstream captured_cerr;
  std::streambuf* old = std::cerr.rdbuf(captured_cerr.rdbuf());

  int64_t result_model_version;
  cb::Error result;
  result = tip.DetermineStatsModelVersion(
      model_identifier, start_stats_map, end_stats_map, &result_model_version);

  CHECK(result_model_version == expected_model_version);
  CHECK(result.IsOk() != expect_exception);
  CHECK(captured_cerr.str().empty() != expect_warning);

  std::cerr.rdbuf(old);
}

TEST_CASE(
    "valid_latency_measurement: testing the ValidLatencyMeasurement function")
{
  MockInferenceProfiler mock_inference_profiler{};

  SUBCASE("testing logic relevant to response throughput metric")
  {
    auto clock_epoch{std::chrono::time_point<std::chrono::system_clock>()};

    auto request1_timestamp{clock_epoch + std::chrono::nanoseconds(1)};
    auto response1_timestamp{clock_epoch + std::chrono::nanoseconds(2)};
    auto response2_timestamp{clock_epoch + std::chrono::nanoseconds(3)};
    auto request_record1{RequestRecord(
        request1_timestamp,
        std::vector<std::chrono::time_point<std::chrono::system_clock>>{
            response1_timestamp, response2_timestamp},
        0, false, 0)};

    auto request2_timestamp{clock_epoch + std::chrono::nanoseconds(4)};
    auto response3_timestamp{clock_epoch + std::chrono::nanoseconds(5)};
    auto response4_timestamp{clock_epoch + std::chrono::nanoseconds(6)};
    auto response5_timestamp{clock_epoch + std::chrono::nanoseconds(7)};
    auto request_record2{RequestRecord(
        request2_timestamp,
        std::vector<std::chrono::time_point<std::chrono::system_clock>>{
            response3_timestamp, response4_timestamp, response5_timestamp},
        0, false, 0)};

    mock_inference_profiler.all_request_records_ = {
        request_record1, request_record2};

    const std::pair<uint64_t, uint64_t> valid_range{
        std::make_pair(0, UINT64_MAX)};
    size_t valid_sequence_count{0};
    size_t delayed_request_count{0};
    std::vector<uint64_t> valid_latencies{};
    size_t response_count{0};
    std::vector<RequestRecord> valid_requests{};

    mock_inference_profiler.ValidLatencyMeasurement(
        valid_range, valid_sequence_count, delayed_request_count,
        &valid_latencies, response_count, valid_requests);

    CHECK(response_count == 5);
  }
  SUBCASE("testing logic relevant to valid request output")
  {
    auto clock_epoch{std::chrono::time_point<std::chrono::system_clock>()};

    auto request1_timestamp{clock_epoch + std::chrono::nanoseconds(1)};
    auto response1_timestamp{clock_epoch + std::chrono::nanoseconds(2)};
    auto request_record1{RequestRecord(
        request1_timestamp,
        std::vector<std::chrono::time_point<std::chrono::system_clock>>{
            response1_timestamp},
        0, false, 0)};

    auto request2_timestamp{clock_epoch + std::chrono::nanoseconds(3)};
    auto response2_timestamp{clock_epoch + std::chrono::nanoseconds(4)};
    auto request_record2{RequestRecord(
        request2_timestamp,
        std::vector<std::chrono::time_point<std::chrono::system_clock>>{
            response2_timestamp},
        0, false, 0)};

    auto request3_timestamp{clock_epoch + std::chrono::nanoseconds(5)};
    auto response3_timestamp{clock_epoch + std::chrono::nanoseconds(6)};
    auto request_record3{RequestRecord(
        request3_timestamp,
        std::vector<std::chrono::time_point<std::chrono::system_clock>>{
            response3_timestamp},
        0, false, 0)};

    mock_inference_profiler.all_request_records_ = {
        request_record1, request_record2, request_record3};

    const std::pair<uint64_t, uint64_t> valid_range{std::make_pair(0, 4)};
    size_t valid_sequence_count{0};
    size_t delayed_request_count{0};
    std::vector<uint64_t> valid_latencies{};
    size_t response_count{0};
    std::vector<RequestRecord> valid_requests{};

    mock_inference_profiler.ValidLatencyMeasurement(
        valid_range, valid_sequence_count, delayed_request_count,
        &valid_latencies, response_count, valid_requests);

    CHECK(valid_requests.size() == 2);
    CHECK(valid_requests[0].start_time_ == request1_timestamp);
    CHECK(valid_requests[1].start_time_ == request2_timestamp);
  }
}

TEST_CASE(
    "merge_perf_status_reports: testing the MergePerfStatusReports function")
{
  MockInferenceProfiler mock_inference_profiler{};

  SUBCASE("testing logic relevant to response throughput metric")
  {
    PerfStatus perf_status1{};
    perf_status1.client_stats.response_count = 8;
    perf_status1.client_stats.duration_ns = 2000000000;

    PerfStatus perf_status2{};
    perf_status2.client_stats.response_count = 10;
    perf_status2.client_stats.duration_ns = 4000000000;

    std::deque<PerfStatus> perf_status{perf_status1, perf_status2};
    PerfStatus summary_status{};

    cb::Error error{};

    EXPECT_CALL(
        mock_inference_profiler, MergeServerSideStats(testing::_, testing::_))
        .WillOnce(testing::Return(cb::Error::Success));
    EXPECT_CALL(
        mock_inference_profiler, SummarizeLatency(testing::_, testing::_))
        .WillOnce(testing::Return(cb::Error::Success));

    error = mock_inference_profiler.MergePerfStatusReports(
        perf_status, summary_status);

    REQUIRE(error.IsOk() == true);
    CHECK(summary_status.client_stats.response_count == 18);
    CHECK(
        summary_status.client_stats.responses_per_sec == doctest::Approx(3.0));
  }
}

TEST_CASE("summarize_client_stat: testing the SummarizeClientStat function")
{
  MockInferenceProfiler mock_inference_profiler{};

  SUBCASE("testing logic relevant to response throughput metric")
  {
    mock_inference_profiler.parser_ = std::make_shared<MockModelParser>();
    mock_inference_profiler.manager_ = std::make_unique<MockLoadManager>();

    const cb::InferStat start_stat{};
    const cb::InferStat end_stat{};
    const uint64_t duration_ns{2000000000};
    const size_t valid_request_count{0};
    const size_t delayed_request_count{0};
    const size_t valid_sequence_count{0};
    const size_t response_count{8};
    PerfStatus summary{};

    cb::Error error{};

    error = mock_inference_profiler.SummarizeClientStat(
        start_stat, end_stat, duration_ns, valid_request_count,
        delayed_request_count, valid_sequence_count, response_count, summary);

    REQUIRE(error.IsOk() == true);
    CHECK(summary.client_stats.response_count == 8);
    CHECK(summary.client_stats.responses_per_sec == doctest::Approx(4.0));
  }
}
}}  // namespace triton::perfanalyzer
