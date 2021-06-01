// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "common.h"

namespace tc = triton::client;

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {

/// This class is used to pass inference status and id to upstream backend.
/// Created so that the API is similar to `triton, torchserver,
/// tensorflow_serving` APIs
class InferResult {
 public:
  static void Create(
      InferResult** infer_result, const tc::Error& err, const std::string& id)
  {
    *infer_result = reinterpret_cast<InferResult*>(new InferResult(err, id));
  }

  tc::Error Id(std::string* id) const
  {
    *id = request_id_;
    return tc::Error::Success;
  }
  tc::Error RequestStatus() const { return status_; }

 private:
  InferResult(const tc::Error& err, const std::string& id)
      : status_(err), request_id_(id)
  {
  }

  std::string request_id_;
  tc::Error status_;
};
}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
