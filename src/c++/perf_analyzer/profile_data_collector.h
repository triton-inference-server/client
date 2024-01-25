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

#include <algorithm>
#include <map>
#include <tuple>

#include "client_backend/client_backend.h"
#include "constants.h"
#include "perf_utils.h"
#include "request_record.h"

namespace triton { namespace perfanalyzer {

/// Data structure to hold which inference load mode was used for an experiment.
/// Only one data member will be nonzero, indicating the inference load mode for
/// a particular experiment.
struct InferenceLoadMode {
  uint32_t concurrency;
  double request_rate;

  InferenceLoadMode()
  {
    concurrency = 0;
    request_rate = 0.0;
  }

  InferenceLoadMode(uint64_t c, double rr)
  {
    concurrency = c;
    request_rate = rr;
  }

  bool operator==(const InferenceLoadMode& rhs) const
  {
    return (concurrency == rhs.concurrency) &&
           (request_rate == rhs.request_rate);
  }
};

/// Data structure to hold profile export data for an experiment (e.g.
/// concurrency 4 or request rate 50)
struct Experiment {
  InferenceLoadMode mode;
  std::vector<RequestRecord> requests;
  std::vector<uint64_t> window_boundaries;
};

#ifndef DOCTEST_CONFIG_DISABLE
class NaggyMockProfileDataCollector;
#endif

/// Data structure and methods for storing profile export data.
class ProfileDataCollector {
 public:
  static cb::Error Create(std::shared_ptr<ProfileDataCollector>* collector);
  ~ProfileDataCollector() = default;


  /// Add a measurement window to the collector
  /// @param id Identifier for the experiment
  /// @param window_start_ns The window start timestamp in nanoseconds.
  /// @param window_end_ns The window end timestamp in nanoseconds.
  void AddWindow(
      InferenceLoadMode& id, uint64_t window_start_ns, uint64_t window_end_ns);

  /// Add request records to an experiment
  /// @param id Identifier for the experiment
  /// @param request_records The request information for the current experiment.
  void AddData(
      InferenceLoadMode& id, std::vector<RequestRecord>&& request_records);

  /// Get the experiment data for the profile
  /// @return Experiment data
  std::vector<Experiment>& GetData() { return experiments_; }

  std::string& GetVersion() { return version_; }

  bool IsEmpty() { return experiments_.empty(); }

 private:
  ProfileDataCollector() = default;

  virtual std::vector<Experiment>::iterator FindExperiment(
      InferenceLoadMode& id)
  {
    return std::find_if(
        experiments_.begin(), experiments_.end(),
        [&id](const Experiment& e) { return e.mode == id; });
  };

  std::vector<Experiment> experiments_{};
  std::string version_{VERSION};

#ifndef DOCTEST_CONFIG_DISABLE
  friend NaggyMockProfileDataCollector;
#endif
};
}}  // namespace triton::perfanalyzer
