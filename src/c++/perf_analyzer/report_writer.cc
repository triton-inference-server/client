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

#include "report_writer.h"

#include <algorithm>
#include <fstream>

namespace triton { namespace perfanalyzer {


cb::Error
ReportWriter::Create(
    const std::string& filename, const bool target_concurrency,
    const std::vector<pa::PerfStatus>& summary, const bool verbose_csv,
    const bool include_server_stats, const int32_t percentile,
    const std::shared_ptr<ModelParser>& parser,
    std::unique_ptr<ReportWriter>* writer)
{
  std::unique_ptr<ReportWriter> local_writer(new ReportWriter(
      filename, target_concurrency, summary, verbose_csv, include_server_stats,
      percentile, parser));

  *writer = std::move(local_writer);

  return cb::Error::Success;
}

ReportWriter::ReportWriter(
    const std::string& filename, const bool target_concurrency,
    const std::vector<pa::PerfStatus>& summary, const bool verbose_csv,
    const bool include_server_stats, const int32_t percentile,
    const std::shared_ptr<ModelParser>& parser)
    : filename_(filename), target_concurrency_(target_concurrency),
      summary_(summary), verbose_csv_(verbose_csv),
      include_server_stats_(include_server_stats), percentile_(percentile),
      parser_(parser)
{
}


void
ReportWriter::GenerateReport()
{
  if (!filename_.empty()) {
    std::ofstream ofs(filename_, std::ofstream::out);
    if (target_concurrency_) {
      ofs << "Concurrency,";
    } else {
      ofs << "Request Rate,";
    }
    ofs << "Inferences/Second,Client Send,";
    if (include_server_stats_) {
      ofs << "Network+Server Send/Recv,Server Queue,"
          << "Server Compute Input,Server Compute Infer,"
          << "Server Compute Output,";
      // Only include cache hit if enabled, keep out for backwards
      // compatibility if disabled
      if (parser_->ResponseCacheEnabled()) {
        ofs << "Server Cache Hit,";
        ofs << "Server Cache Miss,";
      }
    }
    ofs << "Client Recv";
    for (const auto& percentile :
         summary_[0].client_stats.percentile_latency_ns) {
      ofs << ",p" << percentile.first << " latency";
    }
    if (verbose_csv_) {
      ofs << ",";
      if (percentile_ == -1) {
        ofs << "Avg latency,";
      }
      ofs << "request/response,";
      ofs << "response wait";
    }
    ofs << std::endl;

    // Sort summary results in order of increasing infer/sec.
    std::sort(
        summary_.begin(), summary_.end(),
        [](const pa::PerfStatus& a, const pa::PerfStatus& b) -> bool {
          return a.client_stats.infer_per_sec < b.client_stats.infer_per_sec;
        });

    for (pa::PerfStatus& status : summary_) {
      if (target_concurrency_) {
        ofs << status.concurrency << ",";
      } else {
        ofs << status.request_rate << ",";
      }

      ofs << status.client_stats.infer_per_sec << ","
          << (status.client_stats.avg_send_time_ns / 1000) << ",";
      if (include_server_stats_) {
        uint64_t avg_queue_ns = status.server_stats.queue_count > 0
                                    ? (status.server_stats.queue_time_ns /
                                       status.server_stats.queue_count)
                                    : 0;
        uint64_t avg_compute_input_ns =
            status.server_stats.compute_input_count > 0
                ? (status.server_stats.compute_input_time_ns /
                   status.server_stats.compute_input_count)
                : 0;
        uint64_t avg_compute_infer_ns =
            status.server_stats.compute_infer_count > 0
                ? (status.server_stats.compute_infer_time_ns /
                   status.server_stats.compute_infer_count)
                : 0;
        uint64_t avg_compute_output_ns =
            status.server_stats.compute_output_count > 0
                ? (status.server_stats.compute_output_time_ns /
                   status.server_stats.compute_output_count)
                : 0;
        uint64_t compute_time_ns = status.server_stats.compute_input_time_ns +
                                   status.server_stats.compute_infer_time_ns +
                                   status.server_stats.compute_output_time_ns;
        if (status.server_stats.compute_input_count !=
                status.server_stats.compute_infer_count ||
            status.server_stats.compute_infer_count !=
                status.server_stats.compute_output_count) {
          throw std::runtime_error(
              "Server side statistics compute counts must be the same.");
        }
        uint64_t compute_cnt = status.server_stats.compute_input_count;
        uint64_t avg_compute_ns =
            compute_cnt > 0 ? compute_time_ns / compute_cnt : 0;
        uint64_t avg_cache_hit_ns =
            status.server_stats.cache_hit_count > 0
                ? (status.server_stats.cache_hit_time_ns /
                   status.server_stats.cache_hit_count)
                : 0;
        uint64_t avg_cache_miss_ns =
            status.server_stats.cache_miss_count > 0
                ? (status.server_stats.cache_miss_time_ns /
                   status.server_stats.cache_miss_count)
                : 0;

        uint64_t avg_client_wait_ns = status.client_stats.avg_latency_ns -
                                      status.client_stats.avg_send_time_ns -
                                      status.client_stats.avg_receive_time_ns;
        // Network misc is calculated by subtracting data from different
        // measurements (server v.s. client), so the result needs to be capped
        // at 0
        uint64_t avg_accounted_time = avg_queue_ns + avg_compute_ns +
                                      avg_cache_hit_ns + avg_cache_miss_ns;
        uint64_t avg_network_misc_ns =
            avg_client_wait_ns > avg_accounted_time
                ? (avg_client_wait_ns - avg_accounted_time)
                : 0;

        if (avg_network_misc_ns == 0) {
          std::cerr << "Server average accounted time was larger than client "
                       "average wait time due to small sample size. Increase "
                       "the measurement interval with `--measurement-interval`."
                    << std::endl;
        }

        ofs << (avg_network_misc_ns / 1000) << "," << (avg_queue_ns / 1000)
            << "," << (avg_compute_input_ns / 1000) << ","
            << (avg_compute_infer_ns / 1000) << ","
            << (avg_compute_output_ns / 1000) << ",";

        if (parser_->ResponseCacheEnabled()) {
          ofs << (avg_cache_hit_ns / 1000) << ",";
          ofs << (avg_cache_miss_ns / 1000) << ",";
        }
      }
      ofs << (status.client_stats.avg_receive_time_ns / 1000);
      for (const auto& percentile : status.client_stats.percentile_latency_ns) {
        ofs << "," << (percentile.second / 1000);
      }
      if (verbose_csv_) {
        const uint64_t avg_latency_us =
            status.client_stats.avg_latency_ns / 1000;
        const uint64_t avg_send_time_us =
            status.client_stats.avg_send_time_ns / 1000;
        const uint64_t avg_receive_time_us =
            status.client_stats.avg_receive_time_ns / 1000;
        const uint64_t avg_request_time_us =
            status.client_stats.avg_request_time_ns / 1000;
        const uint64_t avg_response_wait_time_us =
            avg_request_time_us - avg_send_time_us - avg_receive_time_us;
        ofs << ",";
        if (percentile_ == -1) {
          ofs << avg_latency_us << ",";
        }
        ofs << std::to_string(avg_send_time_us + avg_receive_time_us) << ",";
        ofs << std::to_string(avg_response_wait_time_us);
      }
      ofs << std::endl;
    }
    ofs.close();

    if (include_server_stats_) {
      // Record composing model stat in a separate file.
      if (!summary_.front().server_stats.composing_models_stat.empty()) {
        // For each of the composing model, generate CSV file in the same
        // format as the one for ensemble.
        for (const auto& model_identifier :
             summary_[0].server_stats.composing_models_stat) {
          const auto& name = model_identifier.first.first;
          const auto& version = model_identifier.first.second;
          const auto name_ver = name + "_v" + version;

          std::ofstream ofs(name_ver + "." + filename_, std::ofstream::out);
          if (target_concurrency_) {
            ofs << "Concurrency,";
          } else {
            ofs << "Request Rate,";
          }
          ofs << "Inferences/Second,Client Send,"
              << "Network+Server Send/Recv,Server Queue,"
              << "Server Compute Input,Server Compute Infer,"
              << "Server Compute Output,";

          // Only include cache hit if enabled, keep out for backwards
          // compatibility if disabled
          if (parser_->ResponseCacheEnabled()) {
            ofs << "Server Cache Hit,";
            ofs << "Server Cache Miss,";
          }
          ofs << "Client Recv";

          for (pa::PerfStatus& status : summary_) {
            auto it = status.server_stats.composing_models_stat.find(
                model_identifier.first);
            const auto& stats = it->second;
            uint64_t avg_queue_ns =
                stats.queue_count > 0 ? stats.queue_time_ns / stats.queue_count
                                      : 0;
            uint64_t avg_compute_input_ns =
                stats.compute_input_count > 0
                    ? stats.compute_input_time_ns / stats.compute_input_count
                    : 0;
            uint64_t avg_compute_infer_ns =
                stats.compute_infer_count > 0
                    ? stats.compute_infer_time_ns / stats.compute_infer_count
                    : 0;
            uint64_t avg_compute_output_ns =
                stats.compute_output_count > 0
                    ? stats.compute_output_time_ns / stats.compute_output_count
                    : 0;
            uint64_t compute_time_ns = stats.compute_input_time_ns +
                                       stats.compute_infer_time_ns +
                                       stats.compute_output_time_ns;
            if (stats.compute_input_count != stats.compute_infer_count ||
                stats.compute_infer_count != stats.compute_output_count) {
              throw std::runtime_error(
                  "Server side statistics compute counts must be the same.");
            }
            uint64_t compute_cnt = stats.compute_input_count;
            uint64_t avg_compute_ns =
                compute_cnt > 0 ? compute_time_ns / compute_cnt : 0;
            uint64_t avg_cache_hit_ns =
                stats.cache_hit_count > 0
                    ? stats.cache_hit_time_ns / stats.cache_hit_count
                    : 0;
            uint64_t avg_cache_miss_ns =
                stats.cache_miss_count > 0
                    ? stats.cache_miss_time_ns / stats.cache_miss_count
                    : 0;

            uint64_t avg_overhead_ns =
                stats.success_count > 0
                    ? stats.cumm_time_ns / stats.success_count
                    : 0;
            const uint64_t avg_accounted_time = avg_queue_ns + avg_compute_ns +
                                                avg_cache_hit_ns +
                                                avg_cache_miss_ns;
            avg_overhead_ns = (avg_overhead_ns > avg_accounted_time)
                                  ? (avg_overhead_ns - avg_accounted_time)
                                  : 0;

            if (avg_overhead_ns == 0) {
              std::cerr
                  << "Server average accounted time was larger than client "
                     "average wait time due to small sample size. Increase "
                     "the measurement interval with `--measurement-interval`."
                  << std::endl;
            }

            // infer / sec of the composing model is calculated using the
            // request count ratio between the composing model and the
            // ensemble
            double infer_ratio = status.server_stats.success_count > 0
                                     ? (1.0 * stats.success_count /
                                        status.server_stats.success_count)
                                     : 0.0;
            double infer_per_sec =
                infer_ratio * status.client_stats.infer_per_sec;
            if (target_concurrency_) {
              ofs << status.concurrency << ",";
            } else {
              ofs << status.request_rate << ",";
            }
            ofs << infer_per_sec << ",0," << (avg_overhead_ns / 1000) << ","
                << (avg_queue_ns / 1000) << "," << (avg_compute_input_ns / 1000)
                << "," << (avg_compute_infer_ns / 1000) << ","
                << (avg_compute_output_ns / 1000) << ",";

            // Only include cache hit if enabled, keep out for backwards
            // compatibility if disabled
            if (parser_->ResponseCacheEnabled()) {
              ofs << (avg_cache_hit_ns / 1000) << ",";
              ofs << (avg_cache_miss_ns / 1000) << ",";
            }
            // Client recv
            ofs << "0" << std::endl;
          }
        }
        ofs.close();
      }
    }
  }
}

}}  // namespace triton::perfanalyzer
