// Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unordered_map>
#include <vector>

namespace triton { namespace perfanalyzer {

/// A record containing the data of a single response
struct ResponseData {
  ResponseData(const uint8_t* buf, size_t size)
  {
    uint8_t* array = new uint8_t[size];
    std::memcpy(array, buf, size);
    data_ = std::shared_ptr<uint8_t>(array, [](uint8_t* p) { delete[] p; });
    size_ = size;
  }

  // Define equality comparison operator so it can be inserted into maps
  bool operator==(const ResponseData& other) const
  {
    if (size_ != other.size_)
      return false;
    // Compare the contents of the arrays
    return std::memcmp(data_.get(), other.data_.get(), size_) == 0;
  }

  std::shared_ptr<uint8_t> data_;
  size_t size_;
};


/// A record of an individual request
struct RequestRecord {
  using ResponseOutput = std::unordered_map<std::string, ResponseData>;

  RequestRecord(
      std::chrono::time_point<std::chrono::system_clock> start_time =
          std::chrono::time_point<std::chrono::system_clock>(),
      std::vector<std::chrono::time_point<std::chrono::system_clock>>
          response_timestamps = {},
      std::vector<ResponseOutput> response_outputs = {},
      bool sequence_end = true, bool delayed = false, uint64_t sequence_id = 0,
      bool has_null_last_response = false)
      : start_time_(start_time), response_timestamps_(response_timestamps),
        response_outputs_(response_outputs), sequence_end_(sequence_end),
        delayed_(delayed), sequence_id_(sequence_id),
        has_null_last_response_(has_null_last_response)
  {
  }
  // The timestamp of when the request was started.
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  // Collection of response timestamps
  std::vector<std::chrono::time_point<std::chrono::system_clock>>
      response_timestamps_;

  // Collection of response outputs
  std::vector<ResponseOutput> response_outputs_;
  // Whether or not the request is at the end of a sequence.
  bool sequence_end_;
  // Whether or not the request is delayed as per schedule.
  bool delayed_;
  // Sequence ID of the request
  uint64_t sequence_id_;
  // Whether the last response is null
  bool has_null_last_response_;
};

}}  // namespace triton::perfanalyzer
