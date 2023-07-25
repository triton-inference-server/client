// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <cstdint>
#include <tuple>
#include <vector>

namespace triton { namespace perfanalyzer {

/// The properties of a request required in the callback to effectively
/// interpret the response.
struct RequestProperties {
  RequestProperties() = default;
  RequestProperties(
      std::chrono::time_point<std::chrono::system_clock> start_time,
      std::vector<std::chrono::time_point<std::chrono::system_clock>> end_times,
      bool sequence_end, bool delayed,

      uint32_t sequence_status_index)
      : start_time_(start_time), end_times_(end_times),
        sequence_end_(sequence_end), delayed_(delayed),
        sequence_status_index_(sequence_status_index)
  {
  }
  bool operator==(const RequestProperties& other) const
  {
    return std::tie(
               start_time_, end_times_, sequence_end_, delayed_,
               sequence_status_index_) ==
           std::tie(
               other.start_time_, other.end_times_, other.sequence_end_,
               other.delayed_, other.sequence_status_index_);
  }
  // The timestamp of when the request was started.
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  // Collection of response times
  std::vector<std::chrono::time_point<std::chrono::system_clock>> end_times_;
  // Whether or not the request is at the end of a sequence.
  bool sequence_end_;
  // Whether or not the request is delayed as per schedule.
  bool delayed_;
  // Sequence status index of the request
  uint32_t sequence_status_index_;
};

}}  // namespace triton::perfanalyzer
