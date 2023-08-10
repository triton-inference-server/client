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

#include "profile_data_collector.h"

#include <memory>

#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

cb::Error
ProfileDataCollector::Create(std::shared_ptr<ProfileDataCollector>* collector)
{
  std::shared_ptr<ProfileDataCollector> local_collector{
      new ProfileDataCollector()};
  *collector = std::move(local_collector);
  return cb::Error::Success;
}

void
ProfileDataCollector::AddWindow(
    InferenceLoadMode& id, uint64_t window_start_ns, uint64_t window_end_ns)
{
  auto it = FindExperiment(id);

  if (it == experiments_.end()) {
    Experiment new_experiment{};
    new_experiment.mode = id;
    new_experiment.window_boundaries.push_back(window_start_ns);
    new_experiment.window_boundaries.push_back(window_end_ns);

    experiments_.push_back(new_experiment);
  } else {
    // Window timestamps are always increasing so it is safe to check only the
    // last element
    if (it->window_boundaries.back() != window_start_ns) {
      it->window_boundaries.push_back(window_start_ns);
    }
    it->window_boundaries.push_back(window_end_ns);
  }
}

void
ProfileDataCollector::AddData(
    InferenceLoadMode& id, std::vector<RequestRecord>&& request_records)
{
  auto it = FindExperiment(id);

  if (it == experiments_.end()) {
    Experiment new_experiment{};
    new_experiment.mode = id;
    new_experiment.requests = std::move(request_records);
  } else {
    it->requests.insert(
        it->requests.end(), std::make_move_iterator(request_records.begin()),
        std::make_move_iterator(request_records.end()));
  }
}

}}  // namespace triton::perfanalyzer
