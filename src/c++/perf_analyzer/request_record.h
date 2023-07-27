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

/// A record of an individual request
struct RequestRecord {
  RequestRecord() = default;
  RequestRecord(
      std::chrono::time_point<std::chrono::system_clock> start_time,
      std::vector<std::chrono::time_point<std::chrono::system_clock>>
          response_times,
      bool sequence_end, bool delayed, uint64_t sequence_id)
      : start_time_(start_time), response_times_(response_times),
        sequence_end_(sequence_end), delayed_(delayed),
        sequence_id_(sequence_id)
  {
  }
  // The timestamp of when the request was started.
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  // Collection of response times
  std::vector<std::chrono::time_point<std::chrono::system_clock>>
      response_times_;
  // Whether or not the request is at the end of a sequence.
  bool sequence_end_;
  // Whether or not the request is delayed as per schedule.
  bool delayed_;
  // Sequence ID of the request
  uint64_t sequence_id_;
};

}}  // namespace triton::perfanalyzer
