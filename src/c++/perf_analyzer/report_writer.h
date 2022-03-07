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
#pragma once

#include "client_backend/client_backend.h"
#include "inference_profiler.h"
#include "model_parser.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

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
  /// \return cb::Error object indicating success or
  /// failure.
  static cb::Error Create(
      const std::string& filename, const bool target_concurrency,
      const std::vector<pa::PerfStatus>& summary, const bool verbose_csv,
      const bool include_server_stats, const int32_t percentile,
      const std::shared_ptr<ModelParser>& parser,
      std::unique_ptr<ReportWriter>* writer);

  void GenerateReport();

 private:
  ReportWriter(
      const std::string& filename, const bool target_concurrency,
      const std::vector<pa::PerfStatus>& summary, const bool verbose_csv,
      const bool include_server_stats, const int32_t percentile,
      const std::shared_ptr<ModelParser>& parser);


  const std::string& filename_;
  const bool target_concurrency_;
  const bool include_server_stats_;
  const bool verbose_csv_;
  const int32_t percentile_;
  std::vector<pa::PerfStatus> summary_;
  const std::shared_ptr<ModelParser>& parser_;
};

}}  // namespace triton::perfanalyzer
