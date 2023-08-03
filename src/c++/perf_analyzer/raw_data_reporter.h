// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <rapidjson/document.h>

#include "client_backend/client_backend.h"
#include "raw_data_collector.h"

namespace triton { namespace perfanalyzer {


class RawDataReporter {
 public:
  static cb::Error Create(std::shared_ptr<RawDataReporter>* reporter);
  ~RawDataReporter() = default;

  /// Convert the raw data collected to json output
  /// @param raw_experiments All of the raw data for the experiments run by perf
  /// analyzer
  /// @param raw_version String containing the version number for the json
  /// output
  void ConvertToJson(
      std::vector<Experiment>& raw_experiments, std::string& raw_version);

  /// Output to stdout
  void Print();

  void OutputToFile(std::string& file_path);

 private:
  RawDataReporter() = default;
  void ClearDocument();
  void AddExperiment(
      rapidjson::Value& entry, rapidjson::Value& experiment,
      const Experiment& raw_experiment);
  void AddRequests(
      rapidjson::Value& entry, rapidjson::Value& requests,
      const Experiment& raw_experiment);
  void AddResponses(
      rapidjson::Value& responses,
      const std::vector<std::chrono::time_point<std::chrono::system_clock>>&
          response_times);
  void AddWindowBoundaries(
      rapidjson::Value& entry, rapidjson::Value& window_boundaries,
      const Experiment& raw_experiment);
  void AddVersion(std::string& raw_version);

  rapidjson::Document document_{};
};
}}  // namespace triton::perfanalyzer
