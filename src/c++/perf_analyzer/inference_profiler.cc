// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <limits>
#include <queue>
#include <stdexcept>
#include "client_backend/client_backend.h"
#include "doctest.h"

namespace triton { namespace perfanalyzer {
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
      std::cout << " (overhead " << overhead_avg_us << " usec + "
                << "queue " << ensemble_times.total_queue_time_avg_us
                << " usec + "
                << "cache hit/miss "
                << ensemble_times.total_combined_cache_compute_time_avg_us
                << " usec)" << std::endl;
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
    const bool on_sequence_model, const bool include_lib_stats)
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
  if (stats.delayed_request_count != 0) {
    std::cout << "    Delayed Request Count: " << stats.delayed_request_count
              << std::endl;
  }
  if (on_sequence_model) {
    std::cout << "    Sequence count: " << stats.sequence_count << " ("
              << stats.sequence_per_sec << " seq/sec)" << std::endl;
  }
  std::cout << "    Throughput: " << stats.infer_per_sec << " infer/sec"
            << std::endl;
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
    const std::shared_ptr<ModelParser>& parser)
{
  std::cout << "  Client: " << std::endl;
  ReportClientSideStats(
      summary.client_stats, percentile, protocol, verbose,
      summary.on_sequence_model, include_lib_stats);

  if (include_server_stats) {
    std::cout << "  Server: " << std::endl;
    ReportServerSideStats(summary.server_stats, 1, parser);
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
    const bool should_collect_metrics)
{
  std::unique_ptr<InferenceProfiler> local_profiler(new InferenceProfiler(
      verbose, stability_threshold, measurement_window_ms, max_trials,
      (percentile != -1), percentile, latency_threshold_ms_, protocol, parser,
      profile_backend, std::move(manager), measurement_request_count,
      measurement_mode, mpi_driver, metrics_interval_ms,
      should_collect_metrics));

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
    const uint64_t metrics_interval_ms, const bool should_collect_metrics)
    : verbose_(verbose), measurement_window_ms_(measurement_window_ms),
      max_trials_(max_trials), extra_percentile_(extra_percentile),
      percentile_(percentile), latency_threshold_ms_(latency_threshold_ms_),
      protocol_(protocol), parser_(parser), profile_backend_(profile_backend),
      manager_(std::move(manager)),
      measurement_request_count_(measurement_request_count),
      measurement_mode_(measurement_mode), mpi_driver_(mpi_driver),
      should_collect_metrics_(should_collect_metrics)
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
    const size_t concurrent_request_count, std::vector<PerfStatus>& summary,
    bool& meets_threshold, bool& is_stable)
{
  cb::Error err;
  PerfStatus status_summary;

  status_summary.concurrency = concurrent_request_count;

  is_stable = false;
  meets_threshold = true;

  RETURN_IF_ERROR(dynamic_cast<ConcurrencyManager*>(manager_.get())
                      ->ChangeConcurrencyLevel(concurrent_request_count));

  err = ProfileHelper(false /* clean_starts */, status_summary, &is_stable);
  if (err.IsOk()) {
    summary.push_back(status_summary);
    uint64_t stabilizing_latency_ms =
        status_summary.stabilizing_latency_ns / (1000 * 1000);
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
      err = Report(
          status_summary, percentile_, protocol_, verbose_, include_lib_stats_,
          include_server_stats_, parser_);
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
    const double request_rate, std::vector<PerfStatus>& summary,
    bool& meets_threshold, bool& is_stable)
{
  cb::Error err;
  PerfStatus status_summary;

  status_summary.request_rate = request_rate;

  is_stable = false;
  meets_threshold = true;

  RETURN_IF_ERROR(dynamic_cast<RequestRateManager*>(manager_.get())
                      ->ChangeRequestRate(request_rate));
  std::cout << "Request Rate: " << request_rate
            << " inference requests per seconds" << std::endl;

  err = ProfileHelper(false /*clean_starts*/, status_summary, &is_stable);
  if (err.IsOk()) {
    summary.push_back(status_summary);
    uint64_t stabilizing_latency_ms =
        status_summary.stabilizing_latency_ns / (1000 * 1000);
    if ((stabilizing_latency_ms >= latency_threshold_ms_) &&
        (latency_threshold_ms_ != NO_LIMIT)) {
      std::cerr << "Measured latency went over the set limit of "
                << latency_threshold_ms_ << " msec. " << std::endl;
      meets_threshold = false;
    } else if (!is_stable) {
      std::cerr << "Failed to obtain stable measurement." << std::endl;
      meets_threshold = false;
    } else {
      err = Report(
          status_summary, percentile_, protocol_, verbose_, include_lib_stats_,
          include_server_stats_, parser_);
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
    std::vector<PerfStatus>& summary, bool& meets_threshold, bool& is_stable)
{
  cb::Error err;
  PerfStatus status_summary;

  RETURN_IF_ERROR(
      dynamic_cast<CustomLoadManager*>(manager_.get())->InitCustomIntervals());
  RETURN_IF_ERROR(dynamic_cast<CustomLoadManager*>(manager_.get())
                      ->GetCustomRequestRate(&status_summary.request_rate));

  is_stable = false;
  meets_threshold = true;

  err = ProfileHelper(true /* clean_starts */, status_summary, &is_stable);
  if (err.IsOk()) {
    summary.push_back(status_summary);
    uint64_t stabilizing_latency_ms =
        status_summary.stabilizing_latency_ns / (1000 * 1000);
    if ((stabilizing_latency_ms >= latency_threshold_ms_) &&
        (latency_threshold_ms_ != NO_LIMIT)) {
      std::cerr << "Measured latency went over the set limit of "
                << latency_threshold_ms_ << " msec. " << std::endl;
      meets_threshold = false;
    } else if (!is_stable) {
      std::cerr << "Failed to obtain stable measurement." << std::endl;
      meets_threshold = false;
    } else {
      err = Report(
          status_summary, percentile_, protocol_, verbose_, include_lib_stats_,
          include_server_stats_, parser_);
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
    const bool clean_starts, PerfStatus& status_summary, bool* is_stable)
{
  // Start measurement
  LoadStatus load_status;
  size_t completed_trials = 0;
  std::queue<cb::Error> error;
  std::deque<PerfStatus> perf_status;
  all_timestamps_.clear();
  previous_window_end_ns_ = 0;

  do {
    PerfStatus status_summary;
    RETURN_IF_ERROR(manager_->CheckHealth());

    // Needed to obtain stable measurements
    if (clean_starts) {
      manager_->ResetWorkers();
    }

    if (measurement_mode_ == MeasurementMode::TIME_WINDOWS) {
      error.push(Measure(status_summary, measurement_window_ms_, false));
    } else {
      error.push(Measure(status_summary, measurement_request_count_, true));
    }
    perf_status.push_back(status_summary);

    if (error.size() > load_parameters_.stability_window) {
      error.pop();
      perf_status.pop_front();
    }

    if (error.back().IsOk()) {
      load_status.infer_per_sec.push_back(
          status_summary.client_stats.infer_per_sec);
      load_status.latencies.push_back(status_summary.stabilizing_latency_ns);
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
                    << (status_summary.client_stats.percentile_latency_ns
                            .find(percentile_)
                            ->second /
                        1000)
                    << " usec" << std::endl;
        } else {
          std::cout << "Avg latency: "
                    << (status_summary.client_stats.avg_latency_ns / 1000)
                    << " usec (std " << status_summary.client_stats.std_us
                    << " usec)" << std::endl;
        }
      } else {
        std::cout << "  Pass [" << (completed_trials + 1)
                  << "] cb::Error: " << error.back().Message() << std::endl;
      }
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

  // return the appropriate error which might have occured in the
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
    RETURN_IF_ERROR(MergePerfStatusReports(perf_status, status_summary));
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
  return load_status.latencies[idx] < (latency_threshold_ms_ * 1000 * 1000);
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
    std::deque<PerfStatus>& perf_status_reports, PerfStatus& summary_status)
{
  auto& perf_status = perf_status_reports[0];

  // Make sure that the perf status reports profiling settings match with each
  // other.
  for (size_t i = 1; i < perf_status_reports.size(); i++) {
    perf_status.concurrency = summary_status.concurrency;
    perf_status.request_rate = summary_status.request_rate;

    if (perf_status_reports[i].on_sequence_model !=
        perf_status.on_sequence_model) {
      return cb::Error(
          "Incosistent sequence setting detected.", pa::GENERIC_ERROR);
    }

    if (perf_status_reports[i].batch_size != perf_status.batch_size) {
      return cb::Error("Incosistent batch size detected.", pa::GENERIC_ERROR);
    }

    if (perf_status_reports[i].server_stats.composing_models_stat.size() !=
        perf_status.server_stats.composing_models_stat.size()) {
      return cb::Error(
          "Inconsistent ensemble setting detected between the trials.",
          pa::GENERIC_ERROR);
    }
  }

  summary_status.batch_size = perf_status.batch_size;
  summary_status.on_sequence_model = perf_status.on_sequence_model;

  // Initialize the client stats for the merged report.
  summary_status.client_stats.request_count = 0;
  summary_status.client_stats.sequence_count = 0;
  summary_status.client_stats.delayed_request_count = 0;
  summary_status.client_stats.duration_ns = 0;
  summary_status.client_stats.avg_latency_ns = 0;
  summary_status.client_stats.percentile_latency_ns.clear();
  summary_status.client_stats.latencies.clear();
  summary_status.client_stats.std_us = 0;
  summary_status.client_stats.avg_request_time_ns = 0;
  summary_status.client_stats.avg_send_time_ns = 0;
  summary_status.client_stats.avg_receive_time_ns = 0;
  summary_status.client_stats.infer_per_sec = 0;
  summary_status.client_stats.sequence_per_sec = 0;
  summary_status.client_stats.completed_count = 0;
  summary_status.stabilizing_latency_ns = 0;

  std::vector<ServerSideStats> server_side_stats;
  for (auto& perf_status : perf_status_reports) {
    // Aggregated Client Stats
    summary_status.client_stats.request_count +=
        perf_status.client_stats.request_count;
    summary_status.client_stats.sequence_count +=
        perf_status.client_stats.sequence_count;
    summary_status.client_stats.delayed_request_count +=
        perf_status.client_stats.delayed_request_count;
    summary_status.client_stats.duration_ns +=
        perf_status.client_stats.duration_ns;

    server_side_stats.push_back(perf_status.server_stats);

    summary_status.client_stats.latencies.insert(
        summary_status.client_stats.latencies.end(),
        perf_status.client_stats.latencies.begin(),
        perf_status.client_stats.latencies.end());
  }

  if (include_lib_stats_) {
    for (auto& perf_status : perf_status_reports) {
      summary_status.client_stats.completed_count +=
          perf_status.client_stats.completed_count;

      summary_status.client_stats.avg_request_time_ns +=
          perf_status.client_stats.avg_request_time_ns *
          perf_status.client_stats.completed_count;

      summary_status.client_stats.avg_send_time_ns +=
          perf_status.client_stats.avg_send_time_ns *
          perf_status.client_stats.completed_count;

      summary_status.client_stats.avg_receive_time_ns +=
          perf_status.client_stats.avg_receive_time_ns *
          perf_status.client_stats.completed_count;
    }

    if (summary_status.client_stats.completed_count != 0) {
      summary_status.client_stats.avg_request_time_ns =
          summary_status.client_stats.avg_request_time_ns /
          summary_status.client_stats.completed_count;

      summary_status.client_stats.avg_send_time_ns =
          summary_status.client_stats.avg_send_time_ns /
          summary_status.client_stats.completed_count;

      summary_status.client_stats.avg_receive_time_ns =
          summary_status.client_stats.avg_receive_time_ns /
          summary_status.client_stats.completed_count;
    }
  }

  RETURN_IF_ERROR(
      MergeServerSideStats(server_side_stats, summary_status.server_stats));

  std::sort(
      summary_status.client_stats.latencies.begin(),
      summary_status.client_stats.latencies.end());

  float client_duration_sec =
      (float)summary_status.client_stats.duration_ns / NANOS_PER_SECOND;
  summary_status.client_stats.sequence_per_sec =
      summary_status.client_stats.sequence_count / client_duration_sec;
  summary_status.client_stats.infer_per_sec =
      (summary_status.client_stats.request_count * summary_status.batch_size) /
      client_duration_sec;
  RETURN_IF_ERROR(
      SummarizeLatency(summary_status.client_stats.latencies, summary_status));

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
    summary_status.metrics.push_back(std::move(merged_metrics));
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
    PerfStatus& status_summary, uint64_t measurement_window,
    bool is_count_based)
{
  std::map<cb::ModelIdentifier, cb::ModelStatistics> start_status;
  std::map<cb::ModelIdentifier, cb::ModelStatistics> end_status;
  cb::InferStat start_stat;
  cb::InferStat end_stat;

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
    // Need to zero out start stats when window start is 0
    start_status = std::map<cb::ModelIdentifier, cb::ModelStatistics>();
    start_stat = cb::InferStat();
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
    metrics_manager_->GetLatestMetrics(status_summary.metrics);
  }

  // Get server status and then print report on difference between
  // before and after status.
  if (include_server_stats_) {
    RETURN_IF_ERROR(GetServerSideStatus(&end_status));
    prev_server_side_stats_ = end_status;
  }

  RETURN_IF_ERROR(manager_->GetAccumulatedClientStat(&end_stat));
  prev_client_side_stats_ = end_stat;

  TimestampVector current_timestamps;
  RETURN_IF_ERROR(manager_->SwapTimestamps(current_timestamps));
  all_timestamps_.insert(
      all_timestamps_.end(), current_timestamps.begin(),
      current_timestamps.end());

  RETURN_IF_ERROR(Summarize(
      start_status, end_status, start_stat, end_stat, status_summary,
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

  // Get measurement from requests that fall within the time interval
  std::pair<uint64_t, uint64_t> valid_range{window_start_ns, window_end_ns};
  std::vector<uint64_t> latencies;
  ValidLatencyMeasurement(
      valid_range, valid_sequence_count, delayed_request_count, &latencies);

  RETURN_IF_ERROR(SummarizeLatency(latencies, summary));
  RETURN_IF_ERROR(SummarizeClientStat(
      start_stat, end_stat, valid_range.second - valid_range.first,
      latencies.size(), valid_sequence_count, delayed_request_count, summary));
  summary.client_stats.latencies = std::move(latencies);

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
    std::vector<uint64_t>* valid_latencies)
{
  valid_latencies->clear();
  valid_sequence_count = 0;
  std::vector<size_t> erase_indices{};
  for (size_t i = 0; i < all_timestamps_.size(); i++) {
    const auto& timestamp = all_timestamps_[i];
    uint64_t request_start_ns = CHRONO_TO_NANOS(std::get<0>(timestamp));
    uint64_t request_end_ns = CHRONO_TO_NANOS(std::get<1>(timestamp));

    if (request_start_ns <= request_end_ns) {
      // Only counting requests that end within the time interval
      if ((request_end_ns >= valid_range.first) &&
          (request_end_ns <= valid_range.second)) {
        valid_latencies->push_back(request_end_ns - request_start_ns);
        erase_indices.push_back(i);
        // Just add the sequence_end flag here.
        if (std::get<2>(timestamp)) {
          valid_sequence_count++;
        }
        if (std::get<3>(timestamp)) {
          delayed_request_count++;
        }
      }
    }
  }

  // Iterate through erase indices backwards so that erases from
  // `all_timestamps_` happen from the back to the front to avoid using wrong
  // indices after subsequent erases
  std::for_each(erase_indices.rbegin(), erase_indices.rend(), [this](size_t i) {
    this->all_timestamps_.erase(this->all_timestamps_.begin() + i);
  });

  // Always sort measured latencies as percentile will be reported as default
  std::sort(valid_latencies->begin(), valid_latencies->end());
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
    PerfStatus& summary)
{
  summary.on_sequence_model =
      ((parser_->SchedulerType() == ModelParser::SEQUENCE) ||
       (parser_->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE));
  summary.batch_size = std::max(manager_->BatchSize(), (size_t)1);
  summary.client_stats.request_count = valid_request_count;
  summary.client_stats.sequence_count = valid_sequence_count;
  summary.client_stats.delayed_request_count = delayed_request_count;
  summary.client_stats.duration_ns = duration_ns;
  float client_duration_sec =
      (float)summary.client_stats.duration_ns / NANOS_PER_SECOND;
  summary.client_stats.sequence_per_sec =
      valid_sequence_count / client_duration_sec;
  summary.client_stats.infer_per_sec =
      (valid_request_count * summary.batch_size) / client_duration_sec;

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

cb::Error
InferenceProfiler::SummarizeServerStatsHelper(
    const cb::ModelIdentifier& model_identifier,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
    const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
    ServerSideStats* server_stats)
{
  // If model_version is an empty string then look in the end status to find
  // the latest (highest valued version) and use that as the version.
  int64_t status_model_version = -1;
  if (model_identifier.second.empty()) {
    for (const auto& id : end_status) {
      // Model name should match
      if (model_identifier.first.compare(id.first.first) == 0) {
        int64_t this_version = std::stoll(id.first.second);
        status_model_version = std::max(status_model_version, this_version);
      }
    }
  } else {
    status_model_version = std::stoll(model_identifier.second);
  }

  if (status_model_version == -1) {
    return cb::Error(
        "failed to determine the requested model version", pa::GENERIC_ERROR);
  }

  const std::pair<std::string, std::string> this_id(
      model_identifier.first, std::to_string(status_model_version));

  const auto& end_itr = end_status.find(this_id);
  if (end_itr == end_status.end()) {
    return cb::Error(
        "missing statistics for requested model", pa::GENERIC_ERROR);
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
  for (const auto& composing_model_identifier :
       (*parser_->GetComposingModelMap())[model_identifier.first]) {
    auto it = server_stats->composing_models_stat
                  .emplace(composing_model_identifier, ServerSideStats())
                  .first;
    RETURN_IF_ERROR(SummarizeServerStats(
        composing_model_identifier, start_status, end_status, &(it->second)));
  }

  return cb::Error::Success;
}

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
