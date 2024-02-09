// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <ostream>

#include "client_backend/client_backend.h"
#include "inference_profiler.h"
#include "metrics.h"
#include "model_parser.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class NaggyMockReportWriter;
#endif

//==============================================================================
/// ReportWriter is a helper class to generate csv files from the profiled data.
///
class ReportWriter {
 public:
  ~ReportWriter() = default;

  /// Create a ReportWriter that is responsible for generating csv output files.
  /// \param filename Name of csv file.
  /// \param target_concurrency Is there a concurrency range or request rate
  /// range?
  /// \param summary Returns the trace of the measurement along the
  /// search path.
  /// \param verbose_csv Print extra information for Model Analyzer
  /// \param include_server_stats Are server stats included in output
  /// \param percentile The percentile in terms of latency to be reported.
  /// if it is a valid percentile value, the percentile latency will reported
  /// and used as stable criteria instead of average latency. If it is -1,
  /// average latency will be reported and used as stable criteria.
  /// \param parser The ModelParse object which holds all the details about the
  /// model.
  /// \param writer Returns a new ReportWriter object.
  /// \param should_output_metrics Whether server-side inference server metrics
  /// should be output.
  /// \return cb::Error object indicating success or failure.
  static cb::Error Create(
      const std::string& filename, const bool target_concurrency,
      const std::vector<pa::PerfStatus>& summary, const bool verbose_csv,
      const bool include_server_stats, const int32_t percentile,
      const std::shared_ptr<ModelParser>& parser,
      std::unique_ptr<ReportWriter>* writer, const bool should_output_metrics,
      std::shared_ptr<ProfileDataCollector> collector,
      const bool should_output_llm_metrics);

  void GenerateReport();

  /// Output gpu metrics to a stream
  /// \param ofs A stream to output the csv data
  /// \param metric The metric container for a particular concurrency or request
  /// rate
  void WriteGPUMetrics(std::ostream& ofs, const Metrics& metric);

  /// Output LLM metrics (e.g. average first token latency) to a stream.
  /// \param ofs A stream to output the csv data
  /// \param status Profile summary and statistics of a single experiment
  void WriteLLMMetrics(std::ostream& ofs, const PerfStatus& status);

 private:
  ReportWriter(
      const std::string& filename, const bool target_concurrency,
      const std::vector<pa::PerfStatus>& summary, const bool verbose_csv,
      const bool include_server_stats, const int32_t percentile,
      const std::shared_ptr<ModelParser>& parser,
      const bool should_output_metrics,
      std::shared_ptr<ProfileDataCollector> collector,
      const bool should_output_llm_metrics);

  /// Calculate LLM metrics (e.g., average first token latency) using the
  /// profile data collected during a single inference experiment.
  /// \param experiment A profile data that contains request and response
  /// timestamps of a single inference experiment.
  std::tuple<std::optional<double>, std::optional<double>> CalculateLLMMetrics(
      const Experiment& experiment);


  const std::string& filename_{""};
  const bool target_concurrency_{true};
  const bool include_server_stats_{true};
  const bool verbose_csv_{true};
  const int32_t percentile_{90};
  std::vector<pa::PerfStatus> summary_{};
  const std::shared_ptr<ModelParser>& parser_{nullptr};
  const bool should_output_metrics_{false};
  std::shared_ptr<ProfileDataCollector> collector_{nullptr};
  const bool should_output_llm_metrics_{false};

#ifndef DOCTEST_CONFIG_DISABLE
  friend NaggyMockReportWriter;

 public:
  ReportWriter() = default;
#endif
};

}}  // namespace triton::perfanalyzer
