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

#include <atomic>
#include "client_backend/client_backend.h"

namespace triton { namespace perfanalyzer {

/// Wraps the information required to send an inference to the
/// server
struct InferContext {
  explicit InferContext() : inflight_request_cnt_(0) {}
  InferContext(InferContext&&) = delete;
  InferContext(const InferContext&) = delete;
  ~InferContext()
  {
    for (const auto input : inputs_) {
      delete input;
    }
    for (const auto output : outputs_) {
      delete output;
    }
  }
  // The backend to communicate with the server
  std::unique_ptr<cb::ClientBackend> infer_backend_;
  // The vector of pointers to InferInput objects for all possible inputs,
  // potentially including optional inputs with no provided data.
  std::vector<cb::InferInput*> inputs_;
  // The vector of pointers to InferInput objects to be
  // used for inference request.
  std::vector<cb::InferInput*> valid_inputs_;
  // The vector of pointers to InferRequestedOutput objects
  // to be used with the inference request.
  std::vector<const cb::InferRequestedOutput*> outputs_;
  // If not empty, the expected output data in the same order as 'outputs_'
  std::vector<std::vector<std::pair<const uint8_t*, size_t>>> expected_outputs_;
  // The InferOptions object holding the details of the
  // inference.
  std::unique_ptr<cb::InferOptions> options_;
  // The total number of inference in-flight.
  std::atomic<size_t> inflight_request_cnt_;
};

}}  // namespace triton::perfanalyzer