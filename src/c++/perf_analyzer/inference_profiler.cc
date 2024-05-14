// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "inference_profiler.h"

#include <math.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "client_backend/client_backend.h"
#include "constants.h"
#include "doctest.h"

namespace triton { namespace perfanalyzer {
cb::Error
ReportPrometheusMetrics(const Metrics& metrics)
{
  const size_t max_num_gpus_in_stdout{16};
  if (metrics.gpu_utilization_per_gpu.size() > max_num_gpus_in_stdout ||
      metrics.gpu_power_usage_per_gpu.size() > max_num_gpus_in_stdout ||
      metrics.gpu_memory_used_bytes_per_gpu.size() > max_num_gpus_in_stdout ||
      metrics.gpu_memory_total_bytes_per_gpu.size() > max_num_gpus_in_stdout) {
    std::cout << "Too many GPUs on system to print out individual Prometheus "
                 "metrics, use the CSV output feature to see metrics."
              << std::endl;
    return cb::Error::Success;
  }

  std::cout << "    Avg GPU Utilization:" << std::endl;
  for (const auto& gpu_uuid_metric_pair : metrics.gpu_utilization_per_gpu) {
    const auto gpu_uuid{gpu_uuid_metric_pair.first};
    const auto metric{gpu_uuid_metric_pair.second};
    std::cout << "      " << gpu_uuid << " : " << (metric * 100.0) << "%"
              << std::endl;
  }

  std::cout << "    Avg GPU Power Usage:" << std::endl;
  for (const auto& gpu_uuid_metric_pair : metrics.gpu_power_usage_per_gpu) {
    const auto gpu_uuid{gpu_uuid_metric_pair.first};
    const auto metric{gpu_uuid_metric_pair.second};
    std::cout << "      " << gpu_uuid << " : " << metric << " watts"
              << std::endl;
  }

  std::cout << "    Max GPU Memory Usage:" << std::endl;
  for (const auto& gpu_uuid_metric_pair :
       metrics.gpu_memory_used_bytes_per_gpu) {
    const auto gpu_uuid{gpu_uuid_metric_pair.first};
    const auto metric{gpu_uuid_metric_pair.second};
    std::cout << "      " << gpu_uuid << " : " << metric << " bytes"
              << std::endl;
  }

  std::cout << "    Total GPU Memory:" << std::endl;
  for (const auto& gpu_uuid_metric_pair :
       metrics.gpu_memory_total_bytes_per_gpu) {
    const auto gpu_uuid{gpu_uuid_metric_pair.first};
    const auto metric{gpu_uuid_metric_pair.second};
    std::cout << "      " << gpu_uuid << " : " << metric << " bytes"
              << std::endl;
  }

  return cb::Error::Success;
}

namespace {

inline uint64_t
AverageDurationInUs(const uint64_t total_time_in_ns, const uint64_t cnt)
{
  if (cnt == 0) {
    return 0;
  }
  return total_time_in_ns / (cnt * 1000);
}

EnsembleDurations
GetTotalEnsembleDurations(const ServerSideStats& stats)
{
  EnsembleDurations result;
  // Calculate avg cache hit latency and cache miss latency for ensemble model
  // in case top level response caching is enabled.
  const uint64_t ensemble_cache_hit_cnt = stats.cache_hit_count;
  const uint64_t ensemble_cache_miss_cnt = stats.cache_miss_count;
  result.total_cache_hit_time_avg_us +=
      AverageDurationInUs(stats.cache_hit_time_ns, ensemble_cache_hit_cnt);
  result.total_cache_miss_time_avg_us +=
      AverageDurationInUs(stats.cache_miss_time_ns, ensemble_cache_miss_cnt);
  for (const auto& model_stats : stats.composing_models_stat) {
    if (model_stats.second.composing_models_stat.empty()) {
      // Cache hit count covers cache hits, not related to compute times
      const uint64_t cache_hit_cnt = model_stats.second.cache_hit_count;
      // cache_miss_cnt should either equal infer_cnt or be zero if
      // cache is disabled or not supported for the model/scheduler type
      const uint64_t cache_miss_cnt = model_stats.second.cache_miss_count;

      result.total_queue_time_avg_us += AverageDurationInUs(
          model_stats.second.queue_time_ns, model_stats.second.queue_count);
      const uint64_t compute_time = model_stats.second.compute_input_time_ns +
                                    model_stats.second.compute_infer_time_ns +
                                    model_stats.second.compute_output_time_ns;
      if (model_stats.second.compute_input_count !=
              model_stats.second.compute_infer_count ||
          model_stats.second.compute_infer_count !=
              model_stats.second.compute_output_count) {
        throw std::runtime_error(
            "Server side statistics compute counts must be the same.");
      }
      const uint64_t compute_cnt = model_stats.second.compute_input_count;
      result.total_compute_time_avg_us +=
          AverageDurationInUs(compute_time, compute_cnt);
      result.total_cache_hit_time_avg_us += AverageDurationInUs(
          model_stats.second.cache_hit_time_ns, cache_hit_cnt);
      result.total_cache_miss_time_avg_us += AverageDurationInUs(
          model_stats.second.cache_miss_time_ns, cache_miss_cnt);
      // Track combined cache/compute total avg for reporting latency with cache
      // enabled
      result.total_combined_cache_compute_time_avg_us += AverageDurationInUs(
          compute_time + model_stats.second.cache_hit_time_ns +
              model_stats.second.cache_miss_time_ns,
          compute_cnt + cache_hit_cnt);
    } else {
      const auto this_ensemble_duration =
          GetTotalEnsembleDurations(model_stats.second);
      result.total_queue_time_avg_us +=
          this_ensemble_duration.total_queue_time_avg_us;
      result.total_compute_time_avg_us +=
          this_ensemble_duration.total_compute_time_avg_us;
      result.total_cache_hit_time_avg_us +=
          this_ensemble_duration.total_cache_hit_time_avg_us;
      result.total_cache_miss_time_avg_us +=
          this_ensemble_duration.total_cache_miss_time_avg_us;
      result.total_combined_cache_compute_time_avg_us +=
          this_ensemble_duration.total_combined_cache_compute_time_avg_us;
    }
  }
  return result;
}


size_t
GetOverheadDuration(size_t total_time, size_t queue_time, size_t compute_time)
{
  return (total_time > queue_time + compute_time)
             ? (total_time - queue_time - compute_time)
             : 0;
}

cb::Error
ReportServerSideStats(
    const ServerSideStats& stats, const int iteration,
    const std::shared_ptr<ModelParser>& parser)
{
  const std::string ident = std::string(2 * iteration, ' ');

  // Infer/exec counts cover compute time done in inference backends,
  // not related to cache hit times
  const uint64_t exec_cnt = stats.execution_count;
  const uint64_t infer_cnt = stats.inference_count;
  // Cache hit count covers cache hits, not related to compute times
  const uint64_t cache_hit_cnt = stats.cache_hit_count;
  const uint64_t cache_miss_cnt = stats.cache_miss_count;

  // Success count covers all successful requests, cumulative time, queue
  // time, compute, and cache
  const uint64_t cnt = stats.success_count;
  if (cnt == 0) {
    std::cout << ident << "  Request count: " << cnt << std::endl;
    return cb::Error::Success;
  }

  const uint64_t cumm_avg_us = AverageDurationInUs(stats.cumm_time_ns, cnt);

  std::cout << ident << "  Inference count: " << infer_cnt << std::endl
            << ident << "  Execution count: " << exec_cnt << std::endl;
  if (parser->ResponseCacheEnabled()) {
    std::cout << ident << "  Cache hit count: " << cache_hit_cnt << std::endl;
    std::cout << ident << "  Cache miss count: " << cache_miss_cnt << std::endl;
  }
  std::cout << ident << "  Successful request count: " << cnt << std::endl
            << ident << "  Avg request latency: " << cumm_avg_us << " usec";

  // Non-ensemble model
  if (stats.composing_models_stat.empty()) {
    const uint64_t queue_avg_us =
        AverageDurationInUs(stats.queue_time_ns, stats.queue_count);
    const uint64_t compute_input_avg_us = AverageDurationInUs(
        stats.compute_input_time_ns, stats.compute_input_count);
    const uint64_t compute_infer_avg_us = AverageDurationInUs(
        stats.compute_infer_time_ns, stats.compute_infer_count);
    const uint64_t compute_output_avg_us = AverageDurationInUs(
        stats.compute_output_time_ns, stats.compute_output_count);
    const uint64_t compute_time = stats.compute_input_time_ns +
                                  stats.compute_infer_time_ns +
                                  stats.compute_output_time_ns;
    if (stats.compute_input_count != stats.compute_infer_count ||
        stats.compute_infer_count != stats.compute_output_count) {
      throw std::runtime_error(
          "Server side statistics compute counts must be the same.");
    }
    const uint64_t compute_cnt = stats.compute_input_count;
    const uint64_t compute_avg_us =
        AverageDurationInUs(compute_time, compute_cnt);
    const uint64_t cache_hit_avg_us =
        AverageDurationInUs(stats.cache_hit_time_ns, cache_hit_cnt);
    const uint64_t cache_miss_avg_us =
        AverageDurationInUs(stats.cache_miss_time_ns, cache_miss_cnt);
    const uint64_t total_compute_time_ns = stats.compute_input_time_ns +
                                           stats.compute_infer_time_ns +
                                           stats.compute_output_time_ns;
    // Get the average of cache hits and misses across successful requests
    const uint64_t combined_cache_compute_avg_us = AverageDurationInUs(
        stats.cache_hit_time_ns + stats.cache_miss_time_ns +
            total_compute_time_ns,
        compute_cnt + cache_hit_cnt);

    if (parser->ResponseCacheEnabled()) {
      const uint64_t overhead_avg_us = GetOverheadDuration(
          cumm_avg_us, queue_avg_us, combined_cache_compute_avg_us);
      std::cout << " (overhead " << overhead_avg_us << " usec + "
                << "queue " << queue_avg_us << " usec + "
                << "cache hit/miss " << combined_cache_compute_avg_us
                << " usec)" << std::endl;
      std::cout << ident << ident
                << "  Average Cache Hit Latency: " << cache_hit_avg_us
                << " usec" << std::endl;
      std::cout << ident << ident << "  Average Cache Miss Latency: "
                << cache_miss_avg_us + compute_avg_us << " usec "
                << "(cache lookup/insertion " << cache_miss_avg_us << " usec + "
                << "compute input " << compute_input_avg_us << " usec + "
                << "compute infer " << compute_infer_avg_us << " usec + "
                << "compute output " << compute_output_avg_us << " usec)"
                << std::endl
                << std::endl;
    }
    // Response Cache Disabled
    else {
      std::cout << " (overhead "
                << GetOverheadDuration(
                       cumm_avg_us, queue_avg_us, compute_avg_us)
                << " usec + "
                << "queue " << queue_avg_us << " usec + "
                << "compute input " << compute_input_avg_us << " usec + "
                << "compute infer " << compute_infer_avg_us << " usec + "
                << "compute output " << compute_output_avg_us << " usec)"
                << std::endl
                << std::endl;

      if (cache_hit_avg_us > 0 || cache_miss_avg_us > 0) {
        std::cerr << "Response Cache is disabled for model ["
                  << parser->ModelName()
                  << "] but cache hit/miss latency is non-zero." << std::endl;
      }
    }
  }
  // Ensemble Model
  else {
    const auto ensemble_times = GetTotalEnsembleDurations(stats);
    // Response Cache Enabled
    if (parser->ResponseCacheEnabled()) {
      const uint64_t overhead_avg_us = GetOverheadDuration(
          cumm_avg_us, ensemble_times.total_queue_time_avg_us,
          ensemble_times.total_combined_cache_compute_time_avg_us);
      // FIXME - Refactor these calculations in case of ensemble top level
      // response cache is enabled
      if (!parser->TopLevelResponseCachingEnabled()) {
        std::cout << " (overhead " << overhead_avg_us << " usec + "
                  << "queue " << ensemble_times.total_queue_time_avg_us
                  << " usec + "
                  << "cache hit/miss "
                  << ensemble_times.total_combined_cache_compute_time_avg_us
                  << " usec)" << std::endl;
      } else {
        std::cout << std::endl;
      }
      std::cout << ident << ident << "  Average Cache Hit Latency: "
                << ensemble_times.total_cache_hit_time_avg_us << " usec"
                << std::endl;
      std::cout << ident << ident << "  Average Cache Miss Latency: "
                << ensemble_times.total_cache_miss_time_avg_us +
                       ensemble_times.total_compute_time_avg_us
                << " usec " << std::endl
                << std::endl;
    }
    // Response Cache Disabled
    else {
      std::cout << " (overhead "
                << GetOverheadDuration(
                       cumm_avg_us, ensemble_times.total_queue_time_avg_us,
                       ensemble_times.total_compute_time_avg_us)
                << " usec + "
                << "queue " << ensemble_times.total_queue_time_avg_us
                << " usec + "
                << "compute " << ensemble_times.total_compute_time_avg_us
                << " usec)" << std::endl
                << std::endl;
    }

    // List out composing models of ensemble model
    std::cout << ident << "Composing models: " << std::endl;
    for (const auto& model_stats : stats.composing_models_stat) {
      const auto& model_identifier = model_stats.first;
      std::cout << ident << model_identifier.first
                << ", version: " << model_identifier.second << std::endl;
      ReportServerSideStats(model_stats.second, iteration + 1, parser);
    }
  }

  return cb::Error::Success;
}

cb::Error
ReportClientSideStats(
    const ClientSideStats& stats, const int64_t percentile,
    const cb::ProtocolType protocol, const bool verbose,
    const bool on_sequence_model, const bool include_lib_stats,
    const double overhead_pct, const double send_request_rate,
    const bool is_decoupled_model)
{
  const uint64_t avg_latency_us = stats.avg_latency_ns / 1000;
  const uint64_t std_us = stats.std_us;

  const uint64_t avg_request_time_us = stats.avg_request_time_ns / 1000;
  const uint64_t avg_send_time_us = stats.avg_send_time_ns / 1000;
  const uint64_t avg_receive_time_us = stats.avg_receive_time_ns / 1000;
  const uint64_t avg_response_wait_time_us =
      avg_request_time_us - avg_send_time_us - avg_receive_time_us;

  std::string client_library_detail = "    ";
  if (include_lib_stats) {
    if (protocol == cb::ProtocolType::GRPC) {
      client_library_detail +=
          "Avg gRPC time: " + std::to_string(avg_request_time_us) + " usec (";
      if (!verbose) {
        client_library_detail +=
            "(un)marshal request/response " +
            std::to_string(avg_send_time_us + avg_receive_time_us) +
            " usec + response wait " +
            std::to_string(avg_response_wait_time_us) + " usec)";
      } else {
        client_library_detail += "marshal " + std::to_string(avg_send_time_us) +
                                 " usec + response wait " +
                                 std::to_string(avg_response_wait_time_us) +
                                 " usec + unmarshal " +
                                 std::to_string(avg_receive_time_us) + " usec)";
      }
    } else if (protocol == cb::ProtocolType::HTTP) {
      client_library_detail +=
          "Avg HTTP time: " + std::to_string(avg_request_time_us) + " usec (";
      if (!verbose) {
        client_library_detail +=
            "send/recv " +
            std::to_string(avg_send_time_us + avg_receive_time_us) +
            " usec + response wait " +
            std::to_string(avg_response_wait_time_us) + " usec)";
      } else {
        client_library_detail += "send " + std::to_string(avg_send_time_us) +
                                 " usec + response wait " +
                                 std::to_string(avg_response_wait_time_us) +
                                 " usec + receive " +
                                 std::to_string(avg_receive_time_us) + " usec)";
      }
    }
  }

  std::cout << "    Request count: " << stats.request_count << std::endl;
  double delay_pct =
      ((double)stats.delayed_request_count / stats.request_count) * 100;
  if (delay_pct > DELAY_PCT_THRESHOLD) {
    std::cout << "    "
              << "Avg send request rate: " << std::fixed << std::setprecision(2)
              << send_request_rate << " infer/sec" << std::endl;
    std::cout << "    "
              << "[WARNING] Perf Analyzer was not able to keep up with the "
                 "desired request rate. ";
    std::cout << delay_pct << "% of the requests were delayed. " << std::endl;
  }
  if (on_sequence_model) {
    std::cout << "    Sequence count: " << stats.sequence_count << " ("
              << stats.sequence_per_sec << " seq/sec)" << std::endl;
  }
  std::cout << "    Throughput: " << stats.infer_per_sec << " infer/sec"
            << std::endl;
  if (is_decoupled_model) {
    std::cout << "    Response Throughput: " << stats.responses_per_sec
              << " infer/sec" << std::endl;
  }

  if (verbose) {
    std::stringstream client_overhead{""};
    client_overhead << "    "
                    << "Avg client overhead: " << std::fixed
                    << std::setprecision(2) << overhead_pct << "%";
    std::cout << client_overhead.str() << std::endl;
  }

  if (percentile == -1) {
    std::cout << "    Avg latency: " << avg_latency_us << " usec"
              << " (standard deviation " << std_us << " usec)" << std::endl;
  }
  for (const auto& percentile : stats.percentile_latency_ns) {
    std::cout << "    p" << percentile.first
              << " latency: " << (percentile.second / 1000) << " usec"
              << std::endl;
  }

  std::cout << client_library_detail << std::endl;

  return cb::Error::Success;
}

cb::Error
Report(
    const PerfStatus& summary, const int64_t percentile,
    const cb::ProtocolType protocol, const bool verbose,
    const bool include_lib_stats, const bool include_server_stats,
    const std::shared_ptr<ModelParser>& parser,
    const bool should_collect_metrics, const double overhead_pct_threshold)
{
  std::cout << "  Client: " << std::endl;
  ReportClientSideStats(
      summary.client_stats, percentile, protocol, verbose,
      summary.on_sequence_model, include_lib_stats, summary.overhead_pct,
      summary.send_request_rate, parser->IsDecoupled());

  if (include_server_stats) {
    std::cout << "  Server: " << std::endl;
    ReportServerSideStats(summary.server_stats, 1, parser);
  }

  if (should_collect_metrics) {
    std::cout << "  Server Prometheus Metrics: " << std::endl;
    ReportPrometheusMetrics(summary.metrics.front());
  }

  if (summary.overhead_pct > overhead_pct_threshold) {
    std::cout << "[WARNING] Perf Analyzer is not able to keep up with the "
                 "desired load. The results may not be accurate."
              << std::endl;
  }
  return cb::Error::Success;
}

}  // namespace

cb::Error
InferenceProfiler::Create(
    const bool verbose, const double stability_threshold,
    const uint64_t measurement_window_ms, const size_t max_trials,
    const int64_t percentile, const uint64_t latency_threshold_ms_,
    const cb::ProtocolType protocol, std::shared_ptr<ModelParser>& parser,
    std::shared_ptr<cb::ClientBackend> profile_backend,
    std::unique_ptr<LoadManager> manager,
    std::unique_ptr<InferenceProfiler>* profiler,
    uint64_t measurement_request_count, MeasurementMode measurement_mode,
    std::shared_ptr<MPIDriver> mpi_driver, const uint64_t metrics_interval_ms,
    const bool should_collect_metrics, const double overhead_pct_threshold,
    const std::shared_ptr<ProfileDataCollector> collector,
    const bool should_collect_profile_data)
{
  std::unique_ptr<InferenceProfiler> local_profiler(new InferenceProfiler(
      verbose, stability_threshold, measurement_window_ms, max_trials,
      (percentile != -1), percentile, latency_threshold_ms_, protocol, parser,
      profile_backend, std::move(manager), measurement_request_count,
      measurement_mode, mpi_driver, metrics_interval_ms, should_collect_metrics,
      overhead_pct_threshold, collector, should_collect_profile_data));

  *profiler = std::move(local_profiler);
  return cb::Error::Success;
}

InferenceProfiler::InferenceProfiler(
    const bool verbose, const double stability_threshold,
    const int32_t measurement_window_ms, const size_t max_trials,
    const bool extra_percentile, const size_t percentile,
    const uint64_t latency_threshold_ms_, const cb::ProtocolType protocol,
    std::shared_ptr<ModelParser>& parser,
    std::shared_ptr<cb::ClientBackend> profile_backend,
    std::unique_ptr<LoadManager> manager, uint64_t measurement_request_count,
    MeasurementMode measurement_mode, std::shared_ptr<MPIDriver> mpi_driver,
    const uint64_t metrics_interval_ms, const bool should_collect_metrics,
    const double overhead_pct_threshold,
    const std::shared_ptr<ProfileDataCollector> collector,
    const bool should_collect_profile_data)
    : verbose_(verbose), measurement_window_ms_(measurement_window_ms),
      max_trials_(max_trials), extra_percentile_(extra_percentile),
      percentile_(percentile), latency_threshold_ms_(latency_threshold_ms_),
      protocol_(protocol), parser_(parser), profile_backend_(profile_backend),
      manager_(std::move(manager)),
      measurement_request_count_(measurement_request_count),
      measurement_mode_(measurement_mode), mpi_driver_(mpi_driver),
      should_collect_metrics_(should_collect_metrics),
      overhead_pct_threshold_(overhead_pct_threshold), collector_(collector),
      should_collect_profile_data_(should_collect_profile_data)
{
  load_parameters_.stability_threshold = stability_threshold;
  load_parameters_.stability_window = 3;
  if (profile_backend_->Kind() == cb::BackendKind::TRITON ||
      profile_backend_->Kind() == cb::BackendKind::TRITON_C_API) {
    // Measure and report client library stats only when the model
    // is not decoupled.
    include_lib_stats_ = (!parser_->IsDecoupled());
    // Measure and report server statistics only when the server
    // supports the statistics extension.
    std::set<std::string> extensions;
    profile_backend_->ServerExtensions(&extensions);
    include_server_stats_ = (extensions.find("statistics") != extensions.end());
  } else {
    include_lib_stats_ = true;
    include_server_stats_ = false;
  }
  if (should_collect_metrics_) {
    metrics_manager_ =
        std::make_shared<MetricsManager>(profile_backend, metrics_interval_ms);
  }
}

cb::Error
InferenceProfiler::Profile(
    const size_t concurrent_request_count, const size_t request_count,
    std::vector<PerfStatus>& perf_statuses, bool& meets_threshold,
    bool& is_stable)
{
  cb::Error err;
  PerfStatus perf_status{};

  perf_status.concurrency = concurrent_request_count;

  is_stable = false;
  meets_threshold = true;

  RETURN_IF_ERROR(
      dynamic_cast<ConcurrencyManager*>(manager_.get())
          ->ChangeConcurrencyLevel(concurrent_request_count, request_count));

  err = ProfileHelper(perf_status, request_count, &is_stable);
  if (err.IsOk()) {
    uint64_t stabilizing_latency_ms =
        perf_status.stabilizing_latency_ns / NANOS_PER_MILLIS;
    if ((stabilizing_latency_ms >= latency_threshold_ms_) &&
        (latency_threshold_ms_ != NO_LIMIT)) {
      std::cerr << "Measured latency went over the set limit of "
                << latency_threshold_ms_ << " msec. " << std::endl;
      meets_threshold = false;
    } else if (!is_stable) {
      if (measurement_mode_ == MeasurementMode::TIME_WINDOWS) {
        std::cerr << "Failed to obtain stable measurement within "
                  << max_trials_ << " measurement windows for concurrency "
                  << concurrent_request_count << ". Please try to "
                  << "increase the --measurement-interval." << std::endl;
      } else if (measurement_mode_ == MeasurementMode::COUNT_WINDOWS) {
        std::cerr << "Failed to obtain stable measurement within "
                  << max_trials_ << " measurement windows for concurrency "
                  << concurrent_request_count << ". Please try to "
                  << "increase the --measurement-request-count." << std::endl;
      }
      meets_threshold = false;
    } else {
      perf_statuses.push_back(perf_status);
      err = Report(
          perf_status, percentile_, protocol_, verbose_, include_lib_stats_,
          include_server_stats_, parser_, should_collect_metrics_,
          overhead_pct_threshold_);
      if (!err.IsOk()) {
        std::cerr << err;
        meets_threshold = false;
      }
    }
  } else {
    return err;
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::Profile(
    const double request_rate, const size_t request_count,
    std::vector<PerfStatus>& perf_statuses, bool& meets_threshold,
    bool& is_stable)
{
  cb::Error err;
  PerfStatus perf_status{};

  perf_status.request_rate = request_rate;

  is_stable = false;
  meets_threshold = true;

  RETURN_IF_ERROR(dynamic_cast<RequestRateManager*>(manager_.get())
                      ->ChangeRequestRate(request_rate, request_count));
  std::cout << "Request Rate: " << request_rate
            << " inference requests per seconds" << std::endl;

  err = ProfileHelper(perf_status, request_count, &is_stable);
  if (err.IsOk()) {
    uint64_t stabilizing_latency_ms =
        perf_status.stabilizing_latency_ns / NANOS_PER_MILLIS;
    if ((stabilizing_latency_ms >= latency_threshold_ms_) &&
        (latency_threshold_ms_ != NO_LIMIT)) {
      std::cerr << "Measured latency went over the set limit of "
                << latency_threshold_ms_ << " msec. " << std::endl;
      meets_threshold = false;
    } else if (!is_stable) {
      std::cerr << "Failed to obtain stable measurement." << std::endl;
      meets_threshold = false;
    } else {
      perf_statuses.push_back(perf_status);
      err = Report(
          perf_status, percentile_, protocol_, verbose_, include_lib_stats_,
          include_server_stats_, parser_, should_collect_metrics_,
          overhead_pct_threshold_);
      if (!err.IsOk()) {
        std::cerr << err;
        meets_threshold = false;
      }
    }
  } else {
    return err;
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::Profile(
    const size_t request_count, std::vector<PerfStatus>& perf_statuses,
    bool& meets_threshold, bool& is_stable)
{
  cb::Error err;
  PerfStatus perf_status{};

  RETURN_IF_ERROR(dynamic_cast<CustomLoadManager*>(manager_.get())
                      ->InitCustomIntervals(request_count));
  RETURN_IF_ERROR(dynamic_cast<CustomLoadManager*>(manager_.get())
                      ->GetCustomRequestRate(&perf_status.request_rate));

  is_stable = false;
  meets_threshold = true;

  err = ProfileHelper(perf_status, request_count, &is_stable);
  if (err.IsOk()) {
    uint64_t stabilizing_latency_ms =
        perf_status.stabilizing_latency_ns / NANOS_PER_MILLIS;
    if ((stabilizing_latency_ms >= latency_threshold_ms_) &&
        (latency_threshold_ms_ != NO_LIMIT)) {
      std::cerr << "Measured latency went over the set limit of "
                << latency_threshold_ms_ << " msec. " << std::endl;
      meets_threshold = false;
    } else if (!is_stable) {
      std::cerr << "Failed to obtain stable measurement." << std::endl;
      meets_threshold = false;
    } else {
      perf_statuses.push_back(perf_status);
      err = Report(
          perf_status, percentile_, protocol_, verbose_, include_lib_stats_,
          include_server_stats_, parser_, should_collect_metrics_,
          overhead_pct_threshold_);
      if (!err.IsOk()) {
        std::cerr << err;
        meets_threshold = false;
      }
    }
  } else {
    return err;
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::ProfileHelper(
    PerfStatus& experiment_perf_status, size_t request_count, bool* is_stable)
{
  // Start measurement
  LoadStatus load_status;
  size_t completed_trials = 0;
  std::queue<cb::Error> error;
  std::deque<PerfStatus> measurement_perf_statuses;
  all_request_records_.clear();
  previous_window_end_ns_ = 0;

  // Start with a fresh empty request records vector in the manager
  //
  std::vector<RequestRecord> empty_request_records;
  RETURN_IF_ERROR(manager_->SwapRequestRecords(empty_request_records));

  do {
    PerfStatus measurement_perf_status;
    measurement_perf_status.concurrency = experiment_perf_status.concurrency;
    measurement_perf_status.request_rate = experiment_perf_status.request_rate;
    RETURN_IF_ERROR(manager_->CheckHealth());

    if (measurement_mode_ == MeasurementMode::TIME_WINDOWS) {
      error.push(
          Measure(measurement_perf_status, measurement_window_ms_, false));
    } else {
      error.push(
          Measure(measurement_perf_status, measurement_request_count_, true));
    }
    measurement_perf_statuses.push_back(measurement_perf_status);

    if (error.size() > load_parameters_.stability_window) {
      error.pop();
      measurement_perf_statuses.pop_front();
    }

    if (error.back().IsOk()) {
      load_status.infer_per_sec.push_back(
          measurement_perf_status.client_stats.infer_per_sec);
      load_status.latencies.push_back(
          measurement_perf_status.stabilizing_latency_ns);
    } else {
      load_status.infer_per_sec.push_back(0);
      load_status.latencies.push_back(std::numeric_limits<uint64_t>::max());
    }

    load_status.avg_ips +=
        load_status.infer_per_sec.back() / load_parameters_.stability_window;
    load_status.avg_latency +=
        load_status.latencies.back() / load_parameters_.stability_window;
    if (verbose_) {
      if (error.back().IsOk()) {
        std::cout << "  Pass [" << (completed_trials + 1)
                  << "] throughput: " << load_status.infer_per_sec.back()
                  << " infer/sec. ";
        if (extra_percentile_) {
          std::cout << "p" << percentile_ << " latency: "
                    << (measurement_perf_status.client_stats
                            .percentile_latency_ns.find(percentile_)
                            ->second /
                        1000)
                    << " usec" << std::endl;
        } else {
          std::cout << "Avg latency: "
                    << (measurement_perf_status.client_stats.avg_latency_ns /
                        1000)
                    << " usec (std "
                    << measurement_perf_status.client_stats.std_us << " usec). "
                    << std::endl;
        }
      } else {
        std::cout << "  Pass [" << (completed_trials + 1)
                  << "] cb::Error: " << error.back().Message() << std::endl;
      }
    }

    // If request-count is specified, then only measure one window and exit
    if (request_count != 0) {
      *is_stable = true;
      break;
    }

    *is_stable = DetermineStability(load_status);

    if (IsDoneProfiling(load_status, is_stable)) {
      break;
    }

    completed_trials++;
  } while ((!early_exit) && (completed_trials < max_trials_));

  if (should_collect_metrics_) {
    metrics_manager_->StopQueryingMetrics();
  }

  // return the appropriate error which might have occurred in the
  // stability_window for its proper handling.
  while (!error.empty()) {
    if (!error.front().IsOk()) {
      return error.front();
    } else {
      error.pop();
    }
  }

  // Only merge the results if the results have stabilized.
  if (*is_stable) {
    RETURN_IF_ERROR(MergePerfStatusReports(
        measurement_perf_statuses, experiment_perf_status));
  }

  if (early_exit) {
    return cb::Error("Received exit signal.", pa::GENERIC_ERROR);
  }
  return cb::Error::Success;
}

bool
InferenceProfiler::DetermineStability(LoadStatus& load_status)
{
  bool stable = false;
  if (load_status.infer_per_sec.size() >= load_parameters_.stability_window) {
    stable = true;
    size_t idx =
        load_status.infer_per_sec.size() - load_parameters_.stability_window;

    for (size_t i = idx; i < load_status.infer_per_sec.size(); i++) {
      if (load_status.infer_per_sec[i] == 0) {
        stable = false;
      }
    }

    stable = stable && CheckWindowForStability(idx, load_status);
  }
  return stable;
}

bool
InferenceProfiler::CheckWindowForStability(size_t idx, LoadStatus& load_status)
{
  return IsInferWindowStable(idx, load_status) &&
         IsLatencyWindowStable(idx, load_status);
}

bool
InferenceProfiler::IsInferWindowStable(size_t idx, LoadStatus& load_status)
{
  auto infer_start = std::begin(load_status.infer_per_sec) + idx;
  auto infer_per_sec_measurements = std::minmax_element(
      infer_start, infer_start + load_parameters_.stability_window);

  auto max_infer_per_sec = *infer_per_sec_measurements.second;
  auto min_infer_per_sec = *infer_per_sec_measurements.first;

  return max_infer_per_sec / min_infer_per_sec <=
         1 + load_parameters_.stability_threshold;
}

bool
InferenceProfiler::IsLatencyWindowStable(size_t idx, LoadStatus& load_status)
{
  auto latency_start = std::begin(load_status.latencies) + idx;
  auto latencies_per_sec_measurements = std::minmax_element(
      latency_start, latency_start + load_parameters_.stability_window);

  double max_latency = *latencies_per_sec_measurements.second;
  double min_latency = *latencies_per_sec_measurements.first;

  return max_latency / min_latency <= 1 + load_parameters_.stability_threshold;
}

bool
InferenceProfiler::IsDoneProfiling(LoadStatus& load_status, bool* is_stable)
{
  bool done = false;
  bool within_threshold = true;
  if (load_status.infer_per_sec.size() >= load_parameters_.stability_window) {
    size_t idx =
        load_status.infer_per_sec.size() - load_parameters_.stability_window;

    for (; idx < load_status.infer_per_sec.size(); idx++) {
      within_threshold &= CheckWithinThreshold(idx, load_status);
    }
  }

  if (mpi_driver_->IsMPIRun()) {
    if (AllMPIRanksAreStable(*is_stable)) {
      done = true;
    }
  } else if (*is_stable) {
    done = true;
  }
  if ((!within_threshold) && (latency_threshold_ms_ != NO_LIMIT)) {
    done = true;
  }
  return done;
}

bool
InferenceProfiler::CheckWithinThreshold(size_t idx, LoadStatus& load_status)
{
  return load_status.latencies[idx] <
         (latency_threshold_ms_ * NANOS_PER_MILLIS);
}

cb::Error
InferenceProfiler::MergeServerSideStats(
    std::vector<ServerSideStats>& server_side_stats,
    ServerSideStats& server_side_summary)
{
  auto& server_side_stat = server_side_stats[0];

  // Make sure that the perf status reports profiling settings match with each
  // other.
  for (size_t i = 1; i < server_side_stats.size(); i++) {
    if (server_side_stats[i].composing_models_stat.size() !=
        server_side_stat.composing_models_stat.size()) {
      return cb::Error(
          "Inconsistent ensemble setting detected between the trials.",
          pa::GENERIC_ERROR);
    }
  }

  // Initialize the server stats for the merged report.
  server_side_summary.inference_count = 0;
  server_side_summary.execution_count = 0;
  server_side_summary.cache_hit_count = 0;
  server_side_summary.cache_miss_count = 0;
  server_side_summary.success_count = 0;
  server_side_summary.queue_count = 0;
  server_side_summary.compute_input_count = 0;
  server_side_summary.compute_output_count = 0;
  server_side_summary.compute_infer_count = 0;
  server_side_summary.cumm_time_ns = 0;
  server_side_summary.queue_time_ns = 0;
  server_side_summary.compute_input_time_ns = 0;
  server_side_summary.compute_infer_time_ns = 0;
  server_side_summary.compute_output_time_ns = 0;
  server_side_summary.cache_hit_time_ns = 0;
  server_side_summary.cache_miss_time_ns = 0;
  server_side_summary.composing_models_stat.clear();
  for (auto& composing_model_stat : server_side_stat.composing_models_stat) {
    std::vector<ServerSideStats> composing_model_stats;
    for (auto& server_side_stat : server_side_stats) {
      composing_model_stats.push_back(
          server_side_stat.composing_models_stat[composing_model_stat.first]);
    }

    ServerSideStats merged_composing_model_stats;
    RETURN_IF_ERROR(MergeServerSideStats(
        composing_model_stats, merged_composing_model_stats));
    server_side_summary.composing_models_stat.insert(
        {composing_model_stat.first, merged_composing_model_stats});
  }

  for (auto& server_side_stat : server_side_stats) {
    // Aggregated Server Stats
    server_side_summary.inference_count += server_side_stat.inference_count;
    server_side_summary.execution_count += server_side_stat.execution_count;
    server_side_summary.cache_hit_count += server_side_stat.cache_hit_count;
    server_side_summary.cache_miss_count += server_side_stat.cache_miss_count;
    server_side_summary.success_count += server_side_stat.success_count;
    server_side_summary.queue_count += server_side_stat.queue_count;
    server_side_summary.compute_input_count +=
        server_side_stat.compute_input_count;
    server_side_summary.compute_infer_count +=
        server_side_stat.compute_infer_count;
    server_side_summary.compute_output_count +=
        server_side_stat.compute_output_count;
    server_side_summary.cumm_time_ns += server_side_stat.cumm_time_ns;
    server_side_summary.queue_time_ns += server_side_stat.queue_time_ns;
    server_side_summary.compute_input_time_ns +=
        server_side_stat.compute_input_time_ns;
    server_side_summary.compute_infer_time_ns +=
        server_side_stat.compute_infer_time_ns;
    server_side_summary.compute_output_time_ns +=
        server_side_stat.compute_output_time_ns;
    server_side_summary.cache_hit_time_ns += server_side_stat.cache_hit_time_ns;
    server_side_summary.cache_miss_time_ns +=
        server_side_stat.cache_miss_time_ns;
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::MergePerfStatusReports(
    std::deque<PerfStatus>& perf_status_reports,
    PerfStatus& experiment_perf_status)
{
  auto& perf_status = perf_status_reports[0];

  // Make sure that the perf status reports profiling settings match with each
  // other.
  for (size_t i = 1; i < perf_status_reports.size(); i++) {
    perf_status.concurrency = experiment_perf_status.concurrency;
    perf_status.request_rate = experiment_perf_status.request_rate;

    if (perf_status_reports[i].on_sequence_model !=
        perf_status.on_sequence_model) {
      return cb::Error(
          "Inconsistent sequence setting detected.", pa::GENERIC_ERROR);
    }

    if (perf_status_reports[i].batch_size != perf_status.batch_size) {
      return cb::Error("Inconsistent batch size detected.", pa::GENERIC_ERROR);
    }

    if (perf_status_reports[i].server_stats.composing_models_stat.size() !=
        perf_status.server_stats.composing_models_stat.size()) {
      return cb::Error(
          "Inconsistent ensemble setting detected between the trials.",
          pa::GENERIC_ERROR);
    }
  }

  experiment_perf_status.batch_size = perf_status.batch_size;
  experiment_perf_status.on_sequence_model = perf_status.on_sequence_model;

  // Initialize the client stats for the merged report.
  experiment_perf_status.client_stats.request_count = 0;
  experiment_perf_status.client_stats.sequence_count = 0;
  experiment_perf_status.client_stats.delayed_request_count = 0;
  experiment_perf_status.client_stats.duration_ns = 0;
  experiment_perf_status.client_stats.avg_latency_ns = 0;
  experiment_perf_status.client_stats.percentile_latency_ns.clear();
  experiment_perf_status.client_stats.latencies.clear();
  experiment_perf_status.client_stats.std_us = 0;
  experiment_perf_status.client_stats.avg_request_time_ns = 0;
  experiment_perf_status.client_stats.avg_send_time_ns = 0;
  experiment_perf_status.client_stats.avg_receive_time_ns = 0;
  experiment_perf_status.client_stats.infer_per_sec = 0;
  experiment_perf_status.client_stats.sequence_per_sec = 0;
  experiment_perf_status.client_stats.completed_count = 0;
  experiment_perf_status.stabilizing_latency_ns = 0;
  experiment_perf_status.overhead_pct = 0;
  experiment_perf_status.send_request_rate = 0.0;

  std::vector<ServerSideStats> server_side_stats;
  for (auto& perf_status : perf_status_reports) {
    // Aggregated Client Stats
    experiment_perf_status.client_stats.request_count +=
        perf_status.client_stats.request_count;
    experiment_perf_status.client_stats.sequence_count +=
        perf_status.client_stats.sequence_count;
    experiment_perf_status.client_stats.delayed_request_count +=
        perf_status.client_stats.delayed_request_count;
    experiment_perf_status.client_stats.response_count +=
        perf_status.client_stats.response_count;
    experiment_perf_status.client_stats.duration_ns +=
        perf_status.client_stats.duration_ns;

    server_side_stats.push_back(perf_status.server_stats);

    experiment_perf_status.client_stats.latencies.insert(
        experiment_perf_status.client_stats.latencies.end(),
        perf_status.client_stats.latencies.begin(),
        perf_status.client_stats.latencies.end());
    // Accumulate the overhead percentage and send rate here to remove extra
    // traversals over the perf_status_reports
    experiment_perf_status.overhead_pct += perf_status.overhead_pct;
    experiment_perf_status.send_request_rate += perf_status.send_request_rate;
  }

  // Calculate the average overhead_pct for the experiment.
  experiment_perf_status.overhead_pct /= perf_status_reports.size();
  experiment_perf_status.send_request_rate /= perf_status_reports.size();

  if (include_lib_stats_) {
    for (auto& perf_status : perf_status_reports) {
      experiment_perf_status.client_stats.completed_count +=
          perf_status.client_stats.completed_count;

      experiment_perf_status.client_stats.avg_request_time_ns +=
          perf_status.client_stats.avg_request_time_ns *
          perf_status.client_stats.completed_count;

      experiment_perf_status.client_stats.avg_send_time_ns +=
          perf_status.client_stats.avg_send_time_ns *
          perf_status.client_stats.completed_count;

      experiment_perf_status.client_stats.avg_receive_time_ns +=
          perf_status.client_stats.avg_receive_time_ns *
          perf_status.client_stats.completed_count;
    }

    if (experiment_perf_status.client_stats.completed_count != 0) {
      experiment_perf_status.client_stats.avg_request_time_ns =
          experiment_perf_status.client_stats.avg_request_time_ns /
          experiment_perf_status.client_stats.completed_count;

      experiment_perf_status.client_stats.avg_send_time_ns =
          experiment_perf_status.client_stats.avg_send_time_ns /
          experiment_perf_status.client_stats.completed_count;

      experiment_perf_status.client_stats.avg_receive_time_ns =
          experiment_perf_status.client_stats.avg_receive_time_ns /
          experiment_perf_status.client_stats.completed_count;
    }
  }

  RETURN_IF_ERROR(MergeServerSideStats(
      server_side_stats, experiment_perf_status.server_stats));

  std::sort(
      experiment_perf_status.client_stats.latencies.begin(),
      experiment_perf_status.client_stats.latencies.end());

  float client_duration_sec =
      (float)experiment_perf_status.client_stats.duration_ns / NANOS_PER_SECOND;
  experiment_perf_status.client_stats.sequence_per_sec =
      experiment_perf_status.client_stats.sequence_count / client_duration_sec;
  experiment_perf_status.client_stats.infer_per_sec =
      (experiment_perf_status.client_stats.request_count *
       experiment_perf_status.batch_size) /
      client_duration_sec;
  experiment_perf_status.client_stats.responses_per_sec =
      experiment_perf_status.client_stats.response_count / client_duration_sec;
  RETURN_IF_ERROR(SummarizeLatency(
      experiment_perf_status.client_stats.latencies, experiment_perf_status));

  if (should_collect_metrics_) {
    // Put all Metric objects in a flat vector so they're easier to merge
    std::vector<std::reference_wrapper<const Metrics>> all_metrics{};
    std::for_each(
        perf_status_reports.begin(), perf_status_reports.end(),
        [&all_metrics](const PerfStatus& p) {
          std::for_each(
              p.metrics.begin(), p.metrics.end(),
              [&all_metrics](const Metrics& m) { all_metrics.push_back(m); });
        });

    Metrics merged_metrics{};
    RETURN_IF_ERROR(MergeMetrics(all_metrics, merged_metrics));
    experiment_perf_status.metrics.push_back(std::move(merged_metrics));
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::GetServerSideStatus(
    std::map<cb::ModelIdentifier, cb::ModelStatistics>* model_stats)
{
  if ((parser_->SchedulerType() == ModelParser::ENSEMBLE) ||
      (parser_->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE)) {
    RETURN_IF_ERROR(profile_backend_->ModelInferenceStatistics(model_stats));
  } else {
    RETURN_IF_ERROR(profile_backend_->ModelInferenceStatistics(
        model_stats, parser_->ModelName(), parser_->ModelVersion()));
  }
  return cb::Error::Success;
}

// Used for measurement
cb::Error
InferenceProfiler::Measure(
    PerfStatus& perf_status, uint64_t measurement_window, bool is_count_based)
{
  std::map<cb::ModelIdentifier, cb::ModelStatistics> start_status;
  std::map<cb::ModelIdentifier, cb::ModelStatistics> end_status;
  cb::InferStat start_stat;
  cb::InferStat end_stat;

  manager_->ResetIdleTime();

  // Set current window start time to end of previous window. For first
  // measurement window, capture start time, server side stats, and client side
  // stats.
  uint64_t window_start_ns = previous_window_end_ns_;
  start_stat = prev_client_side_stats_;
  start_status = prev_server_side_stats_;
  if (window_start_ns == 0) {
    window_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    if (should_collect_metrics_) {
      metrics_manager_->StartQueryingMetrics();
    }
    if (include_server_stats_) {
      RETURN_IF_ERROR(GetServerSideStatus(&start_status));
    }
    RETURN_IF_ERROR(manager_->GetAccumulatedClientStat(&start_stat));
  }

  if (should_collect_metrics_) {
    try {
      metrics_manager_->CheckQueryingStatus();
    }
    catch (const std::exception& e) {
      return cb::Error(e.what(), pa::GENERIC_ERROR);
    }
  }

  if (!is_count_based) {
    // Wait for specified time interval in msec
    std::this_thread::sleep_for(
        std::chrono::milliseconds((uint64_t)(measurement_window_ms_ * 1.2)));
  } else {
    do {
      // Check the health of the worker threads.
      RETURN_IF_ERROR(manager_->CheckHealth());

      // Wait for 1s until enough samples have been collected.
      std::this_thread::sleep_for(std::chrono::milliseconds((uint64_t)1000));
    } while (manager_->CountCollectedRequests() < measurement_window);
  }

  uint64_t window_end_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  previous_window_end_ns_ = window_end_ns;

  if (should_collect_metrics_) {
    metrics_manager_->GetLatestMetrics(perf_status.metrics);
  }

  // Get server status and then print report on difference between
  // before and after status.
  if (include_server_stats_) {
    RETURN_IF_ERROR(GetServerSideStatus(&end_status));
    prev_server_side_stats_ = end_status;
  }

  RETURN_IF_ERROR(manager_->GetAccumulatedClientStat(&end_stat));
  prev_client_side_stats_ = end_stat;

  std::vector<RequestRecord> current_request_records;
  RETURN_IF_ERROR(manager_->SwapRequestRecords(current_request_records));
  all_request_records_.insert(
      all_request_records_.end(), current_request_records.begin(),
      current_request_records.end());

  RETURN_IF_ERROR(Summarize(
      start_status, end_status, start_stat, end_stat, perf_status,
      window_start_ns, window_end_ns));

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::Summarize(
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    const cb::InferStat& start_stat, const cb::InferStat& end_stat,
    PerfStatus& summary, uint64_t window_start_ns, uint64_t window_end_ns)
{
  size_t valid_sequence_count = 0;
  size_t delayed_request_count = 0;
  size_t response_count = 0;

  // Get measurement from requests that fall within the time interval
  std::pair<uint64_t, uint64_t> valid_range{window_start_ns, window_end_ns};
  uint64_t window_duration_ns = valid_range.second - valid_range.first;
  std::vector<uint64_t> latencies;
  std::vector<RequestRecord> valid_requests{};
  ValidLatencyMeasurement(
      valid_range, valid_sequence_count, delayed_request_count, &latencies,
      response_count, valid_requests);

  if (should_collect_profile_data_) {
    CollectData(
        summary, window_start_ns, window_end_ns, std::move(valid_requests));
  }

  RETURN_IF_ERROR(SummarizeLatency(latencies, summary));
  RETURN_IF_ERROR(SummarizeClientStat(
      start_stat, end_stat, window_duration_ns, latencies.size(),
      valid_sequence_count, delayed_request_count, response_count, summary));
  summary.client_stats.latencies = std::move(latencies);

  SummarizeOverhead(window_duration_ns, manager_->GetIdleTime(), summary);

  double window_duration_s{
      window_duration_ns / static_cast<double>(NANOS_PER_SECOND)};

  SummarizeSendRequestRate(
      window_duration_s, manager_->GetAndResetNumSentRequests(), summary);

  if (include_server_stats_) {
    RETURN_IF_ERROR(SummarizeServerStats(
        start_status, end_status, &(summary.server_stats)));
  }

  return cb::Error::Success;
}

void
InferenceProfiler::ValidLatencyMeasurement(
    const std::pair<uint64_t, uint64_t>& valid_range,
    size_t& valid_sequence_count, size_t& delayed_request_count,
    std::vector<uint64_t>* valid_latencies, size_t& response_count,
    std::vector<RequestRecord>& valid_requests)
{
  valid_latencies->clear();
  valid_sequence_count = 0;
  response_count = 0;
  std::vector<size_t> erase_indices{};
  for (size_t i = 0; i < all_request_records_.size(); i++) {
    const auto& request_record = all_request_records_[i];
    uint64_t request_start_ns = CHRONO_TO_NANOS(request_record.start_time_);
    uint64_t request_end_ns;

    if (request_record.has_null_last_response_ == false) {
      request_end_ns =
          CHRONO_TO_NANOS(request_record.response_timestamps_.back());
    } else if (request_record.response_timestamps_.size() > 1) {
      size_t last_response_idx{request_record.response_timestamps_.size() - 2};
      request_end_ns = CHRONO_TO_NANOS(
          request_record.response_timestamps_[last_response_idx]);
    } else {
      erase_indices.push_back(i);
      continue;
    }

    if (request_start_ns <= request_end_ns) {
      // Only counting requests that end within the time interval
      if ((request_end_ns >= valid_range.first) &&
          (request_end_ns <= valid_range.second)) {
        valid_latencies->push_back(request_end_ns - request_start_ns);
        response_count += request_record.response_timestamps_.size();
        if (request_record.has_null_last_response_) {
          response_count--;
        }
        erase_indices.push_back(i);
        if (request_record.sequence_end_) {
          valid_sequence_count++;
        }
        if (request_record.delayed_) {
          delayed_request_count++;
        }
      }
    }
  }

  std::for_each(
      erase_indices.begin(), erase_indices.end(),
      [this, &valid_requests](size_t i) {
        valid_requests.push_back(std::move(this->all_request_records_[i]));
      });

  // Iterate through erase indices backwards so that erases from
  // `all_request_records_` happen from the back to the front to avoid using
  // wrong indices after subsequent erases
  std::for_each(erase_indices.rbegin(), erase_indices.rend(), [this](size_t i) {
    this->all_request_records_.erase(this->all_request_records_.begin() + i);
  });

  // Always sort measured latencies as percentile will be reported as default
  std::sort(valid_latencies->begin(), valid_latencies->end());
}

void
InferenceProfiler::CollectData(
    PerfStatus& summary, uint64_t window_start_ns, uint64_t window_end_ns,
    std::vector<RequestRecord>&& request_records)
{
  InferenceLoadMode id{summary.concurrency, summary.request_rate};
  collector_->AddWindow(id, window_start_ns, window_end_ns);
  collector_->AddData(id, std::move(request_records));
}

cb::Error
InferenceProfiler::SummarizeLatency(
    const std::vector<uint64_t>& latencies, PerfStatus& summary)
{
  if (latencies.size() == 0) {
    return cb::Error(
        "No valid requests recorded within time interval."
        " Please use a larger time window.",
        pa::OPTION_ERROR);
  }

  std::tie(summary.client_stats.avg_latency_ns, summary.client_stats.std_us) =
      GetMeanAndStdDev(latencies);

  // retrieve other interesting percentile
  summary.client_stats.percentile_latency_ns.clear();
  std::set<size_t> percentiles{50, 90, 95, 99};
  if (extra_percentile_) {
    percentiles.emplace(percentile_);
  }

  for (const auto percentile : percentiles) {
    size_t index = (percentile / 100.0) * (latencies.size() - 1) + 0.5;
    summary.client_stats.percentile_latency_ns.emplace(
        percentile, latencies[index]);
  }

  if (extra_percentile_) {
    summary.stabilizing_latency_ns =
        summary.client_stats.percentile_latency_ns.find(percentile_)->second;
  } else {
    summary.stabilizing_latency_ns = summary.client_stats.avg_latency_ns;
  }

  return cb::Error::Success;
}

std::tuple<uint64_t, uint64_t>
InferenceProfiler::GetMeanAndStdDev(const std::vector<uint64_t>& latencies)
{
  uint64_t avg_latency_ns{0};
  uint64_t std_dev_latency_us{0};

  // calculate mean of latencies
  uint64_t tol_latency_ns{
      std::accumulate(latencies.begin(), latencies.end(), 0ULL)};
  avg_latency_ns = tol_latency_ns / latencies.size();

  // calculate sample standard deviation of latencies
  uint64_t sq_sum_latency_avg_diff_ns{0};
  std::for_each(
      latencies.begin(), latencies.end(),
      [avg_latency_ns, &sq_sum_latency_avg_diff_ns](uint64_t l) {
        sq_sum_latency_avg_diff_ns += static_cast<int64_t>(l - avg_latency_ns) *
                                      static_cast<int64_t>(l - avg_latency_ns);
      });
  if (latencies.size() > 1) {
    std_dev_latency_us =
        std::sqrt(sq_sum_latency_avg_diff_ns / (latencies.size() - 1)) / 1000;
  } else {
    std_dev_latency_us = UINT64_MAX;
    std::cerr << "WARNING: Pass contained only one request, so sample latency "
                 "standard deviation will be infinity (UINT64_MAX)."
              << std::endl;
  }


  return std::make_tuple(avg_latency_ns, std_dev_latency_us);
}

cb::Error
InferenceProfiler::SummarizeClientStat(
    const cb::InferStat& start_stat, const cb::InferStat& end_stat,
    const uint64_t duration_ns, const size_t valid_request_count,
    const size_t valid_sequence_count, const size_t delayed_request_count,
    const size_t response_count, PerfStatus& summary)
{
  summary.on_sequence_model =
      ((parser_->SchedulerType() == ModelParser::SEQUENCE) ||
       (parser_->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE));
  summary.batch_size = std::max(manager_->BatchSize(), (size_t)1);
  summary.client_stats.request_count = valid_request_count;
  summary.client_stats.sequence_count = valid_sequence_count;
  summary.client_stats.delayed_request_count = delayed_request_count;
  summary.client_stats.response_count = response_count;
  summary.client_stats.duration_ns = duration_ns;
  float client_duration_sec =
      (float)summary.client_stats.duration_ns / NANOS_PER_SECOND;
  summary.client_stats.sequence_per_sec =
      valid_sequence_count / client_duration_sec;
  summary.client_stats.infer_per_sec =
      (valid_request_count * summary.batch_size) / client_duration_sec;
  summary.client_stats.responses_per_sec = response_count / client_duration_sec;

  if (include_lib_stats_) {
    size_t completed_count =
        end_stat.completed_request_count - start_stat.completed_request_count;
    uint64_t request_time_ns = end_stat.cumulative_total_request_time_ns -
                               start_stat.cumulative_total_request_time_ns;
    summary.client_stats.completed_count = completed_count;
    uint64_t send_time_ns =
        end_stat.cumulative_send_time_ns - start_stat.cumulative_send_time_ns;
    uint64_t receive_time_ns = end_stat.cumulative_receive_time_ns -
                               start_stat.cumulative_receive_time_ns;
    if (completed_count != 0) {
      summary.client_stats.avg_request_time_ns =
          request_time_ns / completed_count;
      summary.client_stats.avg_send_time_ns = send_time_ns / completed_count;
      summary.client_stats.avg_receive_time_ns =
          receive_time_ns / completed_count;
    }
  }

  return cb::Error::Success;
}

void
InferenceProfiler::SummarizeSendRequestRate(
    const double window_duration_s, const size_t num_sent_requests,
    PerfStatus& summary)
{
  if (window_duration_s <= 0.0) {
    throw std::runtime_error("window_duration_s must be positive");
  }

  summary.send_request_rate = num_sent_requests / window_duration_s;
}

cb::Error
InferenceProfiler::DetermineStatsModelVersion(
    const cb::ModelIdentifier& model_identifier,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_stats,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_stats,
    int64_t* status_model_version)
{
  // If model_version is unspecified then look in the stats to find the
  // version with stats that incremented during the measurement.
  //
  // If multiple versions had incremented stats, use the highest numbered one
  // and print a warning
  *status_model_version = -1;
  bool multiple_found = false;
  bool version_unspecified = model_identifier.second.empty();

  if (version_unspecified) {
    for (const auto& x : end_stats) {
      const auto& end_id = x.first;
      const auto& end_stat = x.second;

      bool is_correct_model_name =
          model_identifier.first.compare(end_id.first) == 0;

      if (is_correct_model_name) {
        uint64_t end_queue_count = end_stat.queue_count_;
        uint64_t start_queue_count = 0;

        const auto& itr = start_stats.find(end_id);
        if (itr != start_stats.end()) {
          start_queue_count = itr->second.queue_count_;
        }

        if (end_queue_count > start_queue_count) {
          int64_t this_version = std::stoll(end_id.second);
          if (*status_model_version != -1) {
            multiple_found = true;
          }
          *status_model_version = std::max(*status_model_version, this_version);
        }
      }
    }
  } else {
    const auto& itr = end_stats.find(model_identifier);
    if (itr != end_stats.end()) {
      *status_model_version = std::stoll(model_identifier.second);
    }
  }
  // FIXME - Investigate why composing model version is -1 in case of ensemble
  // cache hit.
  //
  // In case of ensemble models, if top level response caching is
  // enabled, the composing models versions are unavailable in case of a cache
  // hit. This is due to the scheduler sends cache response and composing models
  // do not get executed. It's a valid scenario and shouldn't throw error.
  bool model_version_unspecified_and_invalid =
      *status_model_version == -1 &&
      (parser_ == nullptr || !parser_->TopLevelResponseCachingEnabled());
  if (model_version_unspecified_and_invalid) {
    return cb::Error(
        "failed to find the requested model version", pa::GENERIC_ERROR);
  }

  if (multiple_found) {
    std::cerr << "WARNING: Multiple versions of model "
              << model_identifier.first
              << " are loaded in the triton server, and the version to use was "
                 "unspecified. The stats for that model may be inaccurate."
              << std::endl;
  }

  return cb::Error::Success;
}

// Only for unit-testing
#ifndef DOCTEST_CONFIG_DISABLE
cb::Error
InferenceProfiler::SetTopLevelResponseCaching(
    bool enable_top_level_response_caching)
{
  parser_ = std::make_shared<ModelParser>(cb::BackendKind::TRITON);
  if (parser_ == nullptr) {
    return cb::Error("Failed to initialize ModelParser");
  }
  parser_->SetTopLevelResponseCaching(enable_top_level_response_caching);
  return cb::Error::Success;
}
#endif

cb::Error
InferenceProfiler::SummarizeServerStats(
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    ServerSideStats* server_stats)
{
  RETURN_IF_ERROR(SummarizeServerStats(
      std::make_pair(parser_->ModelName(), parser_->ModelVersion()),
      start_status, end_status, server_stats));
  return cb::Error::Success;
}

cb::Error
InferenceProfiler::SummarizeServerStats(
    const cb::ModelIdentifier& model_identifier,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    ServerSideStats* server_stats)
{
  RETURN_IF_ERROR(SummarizeServerStatsHelper(
      model_identifier, start_status, end_status, server_stats));

  // Summarize the composing models, if any.
  for (auto composing_model_identifier :
       (*parser_->GetComposingModelMap())[model_identifier.first]) {
    int64_t model_version;
    RETURN_IF_ERROR(DetermineStatsModelVersion(
        composing_model_identifier, start_status, end_status, &model_version));
    composing_model_identifier.second = std::to_string(model_version);
    auto it = server_stats->composing_models_stat
                  .emplace(composing_model_identifier, ServerSideStats())
                  .first;
    RETURN_IF_ERROR(SummarizeServerStats(
        composing_model_identifier, start_status, end_status, &(it->second)));
  }

  return cb::Error::Success;
}

cb::Error
InferenceProfiler::SummarizeServerStatsHelper(
    const cb::ModelIdentifier& model_identifier,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    ServerSideStats* server_stats)
{
  int64_t model_version;
  RETURN_IF_ERROR(DetermineStatsModelVersion(
      model_identifier, start_status, end_status, &model_version));

  const std::pair<std::string, std::string> this_id(
      model_identifier.first, std::to_string(model_version));

  const auto& end_itr = end_status.find(this_id);
  if (end_itr == end_status.end()) {
    // In case of ensemble models, if top level response caching is enabled,
    // the composing models statistics are unavailable in case of a cache hit.
    // This is due to the scheduler sends cache response and composing models do
    // not get executed. It's a valid scenario and shouldn't throw error.
    bool stats_not_found_and_invalid =
        model_version == -1 && !parser_->TopLevelResponseCachingEnabled();
    if (stats_not_found_and_invalid) {
      return cb::Error(
          "missing statistics for requested model", pa::GENERIC_ERROR);
    } else {
      // Setting server stats 0 for composing model in case of ensemble request
      // cache hit since the composing model will not be executed
      server_stats->Reset();
    }
  } else {
    uint64_t start_infer_cnt = 0;
    uint64_t start_exec_cnt = 0;
    uint64_t start_cnt = 0;
    uint64_t start_queue_cnt = 0;
    uint64_t start_compute_input_cnt = 0;
    uint64_t start_compute_infer_cnt = 0;
    uint64_t start_compute_output_cnt = 0;
    uint64_t start_cumm_time_ns = 0;
    uint64_t start_queue_time_ns = 0;
    uint64_t start_compute_input_time_ns = 0;
    uint64_t start_compute_infer_time_ns = 0;
    uint64_t start_compute_output_time_ns = 0;
    uint64_t start_cache_hit_cnt = 0;
    uint64_t start_cache_hit_time_ns = 0;
    uint64_t start_cache_miss_cnt = 0;
    uint64_t start_cache_miss_time_ns = 0;

    const auto& start_itr = start_status.find(this_id);
    if (start_itr != start_status.end()) {
      start_infer_cnt = start_itr->second.inference_count_;
      start_exec_cnt = start_itr->second.execution_count_;
      start_cnt = start_itr->second.success_count_;
      start_queue_cnt = start_itr->second.queue_count_;
      start_compute_input_cnt = start_itr->second.compute_input_count_;
      start_compute_infer_cnt = start_itr->second.compute_infer_count_;
      start_compute_output_cnt = start_itr->second.compute_output_count_;
      start_cumm_time_ns = start_itr->second.cumm_time_ns_;
      start_queue_time_ns = start_itr->second.queue_time_ns_;
      start_compute_input_time_ns = start_itr->second.compute_input_time_ns_;
      start_compute_infer_time_ns = start_itr->second.compute_infer_time_ns_;
      start_compute_output_time_ns = start_itr->second.compute_output_time_ns_;
      start_cache_hit_cnt = start_itr->second.cache_hit_count_;
      start_cache_hit_time_ns = start_itr->second.cache_hit_time_ns_;
      start_cache_miss_cnt = start_itr->second.cache_miss_count_;
      start_cache_miss_time_ns = start_itr->second.cache_miss_time_ns_;
    }

    server_stats->inference_count =
        end_itr->second.inference_count_ - start_infer_cnt;
    server_stats->execution_count =
        end_itr->second.execution_count_ - start_exec_cnt;
    server_stats->success_count = end_itr->second.success_count_ - start_cnt;
    server_stats->queue_count = end_itr->second.queue_count_ - start_queue_cnt;
    server_stats->compute_input_count =
        end_itr->second.compute_input_count_ - start_compute_input_cnt;
    server_stats->compute_infer_count =
        end_itr->second.compute_infer_count_ - start_compute_infer_cnt;
    server_stats->compute_output_count =
        end_itr->second.compute_output_count_ - start_compute_output_cnt;
    server_stats->cumm_time_ns =
        end_itr->second.cumm_time_ns_ - start_cumm_time_ns;
    server_stats->queue_time_ns =
        end_itr->second.queue_time_ns_ - start_queue_time_ns;
    server_stats->compute_input_time_ns =
        end_itr->second.compute_input_time_ns_ - start_compute_input_time_ns;
    server_stats->compute_infer_time_ns =
        end_itr->second.compute_infer_time_ns_ - start_compute_infer_time_ns;
    server_stats->compute_output_time_ns =
        end_itr->second.compute_output_time_ns_ - start_compute_output_time_ns;
    server_stats->cache_hit_count =
        end_itr->second.cache_hit_count_ - start_cache_hit_cnt;
    server_stats->cache_hit_time_ns =
        end_itr->second.cache_hit_time_ns_ - start_cache_hit_time_ns;
    server_stats->cache_miss_count =
        end_itr->second.cache_miss_count_ - start_cache_miss_cnt;
    server_stats->cache_miss_time_ns =
        end_itr->second.cache_miss_time_ns_ - start_cache_miss_time_ns;
  }

  return cb::Error::Success;
}

void
InferenceProfiler::SummarizeOverhead(
    const uint64_t window_duration_ns, const uint64_t idle_ns,
    PerfStatus& summary)
{
  // The window start/stop is not instantaneous. It is possible that the PA
  // overhead is smaller than the delay in the window start/stop process. Treat
  // it as 0% overhead (100% idle) in that case
  //
  if (idle_ns > window_duration_ns) {
    summary.overhead_pct = 0;
  } else {
    uint64_t overhead_ns = window_duration_ns - idle_ns;
    double overhead_pct = double(overhead_ns) / window_duration_ns * 100;
    summary.overhead_pct = overhead_pct;
  }
}

bool
InferenceProfiler::AllMPIRanksAreStable(bool current_rank_stability)
{
  int world_size{mpi_driver_->MPICommSizeWorld()};
  std::vector<int> stabilities_per_rank{};
  stabilities_per_rank.resize(world_size, 0);
  int my_rank{mpi_driver_->MPICommRankWorld()};
  stabilities_per_rank[my_rank] = static_cast<int>(current_rank_stability);

  for (int rank{0}; rank < world_size; rank++) {
    mpi_driver_->MPIBcastIntWorld(stabilities_per_rank.data() + rank, 1, rank);
  }

  bool all_stable{true};
  for (int rank{0}; rank < world_size; rank++) {
    if (stabilities_per_rank[rank] == 0) {
      all_stable = false;
      break;
    }
  }

  if (verbose_ && all_stable) {
    std::cout << "All models on all MPI ranks are stable" << std::endl;
  }

  return all_stable;
}

cb::Error
InferenceProfiler::MergeMetrics(
    const std::vector<std::reference_wrapper<const Metrics>>& all_metrics,
    Metrics& merged_metrics)
{
  // Maps from each metric collection mapping gpu uuid to gpu utilization
  std::vector<std::reference_wrapper<const std::map<std::string, double>>>
      gpu_utilization_per_gpu_maps{};

  // Maps from each metric collection mapping gpu uuid to gpu power usage
  std::vector<std::reference_wrapper<const std::map<std::string, double>>>
      gpu_power_usage_per_gpu_maps{};

  // Maps from each metric collection mapping gpu uuid to gpu memory used bytes
  std::vector<std::reference_wrapper<const std::map<std::string, uint64_t>>>
      gpu_memory_used_bytes_per_gpu_maps{};

  // Maps from each metric collection mapping gpu uuid to gpu memory total bytes
  std::vector<std::reference_wrapper<const std::map<std::string, uint64_t>>>
      gpu_memory_total_bytes_per_gpu_maps{};

  // Put all metric maps in vector so they're easier to aggregate
  std::for_each(
      all_metrics.begin(), all_metrics.end(),
      [&gpu_utilization_per_gpu_maps, &gpu_power_usage_per_gpu_maps,
       &gpu_memory_used_bytes_per_gpu_maps,
       &gpu_memory_total_bytes_per_gpu_maps](
          const std::reference_wrapper<const Metrics> m) {
        gpu_utilization_per_gpu_maps.push_back(m.get().gpu_utilization_per_gpu);
        gpu_power_usage_per_gpu_maps.push_back(m.get().gpu_power_usage_per_gpu);
        gpu_memory_used_bytes_per_gpu_maps.push_back(
            m.get().gpu_memory_used_bytes_per_gpu);
        gpu_memory_total_bytes_per_gpu_maps.push_back(
            m.get().gpu_memory_total_bytes_per_gpu);
      });

  GetMetricAveragePerGPU<double>(
      gpu_utilization_per_gpu_maps, merged_metrics.gpu_utilization_per_gpu);
  GetMetricAveragePerGPU<double>(
      gpu_power_usage_per_gpu_maps, merged_metrics.gpu_power_usage_per_gpu);
  GetMetricMaxPerGPU<uint64_t>(
      gpu_memory_used_bytes_per_gpu_maps,
      merged_metrics.gpu_memory_used_bytes_per_gpu);
  GetMetricFirstPerGPU<uint64_t>(
      gpu_memory_total_bytes_per_gpu_maps,
      merged_metrics.gpu_memory_total_bytes_per_gpu);

  return cb::Error::Success;
}

}}  // namespace triton::perfanalyzer
