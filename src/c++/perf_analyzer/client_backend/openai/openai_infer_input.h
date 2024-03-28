// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "../../perf_utils.h"
#include "../client_backend.h"


namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

//==============================================================
/// OpenAiInferInput instance holds the information regarding
/// model input tensors and their corresponding generated data.
///
class OpenAiInferInput : public InferInput {
 public:
  static Error Create(
      InferInput** infer_input, const std::string& name,
      const std::vector<int64_t>& dims, const std::string& datatype);
  /// See InferInput::Shape()
  const std::vector<int64_t>& Shape() const override { return shape_; }
  /// See InferInput::SetShape()
  Error SetShape(const std::vector<int64_t>& shape) override;
  /// See InferInput::Reset()
  Error Reset() override;
  /// See InferInput::AppendRaw()
  Error AppendRaw(const uint8_t* input, size_t input_byte_size) override;
  /// See InferInput::RawData()
  Error RawData(const uint8_t** buf, size_t* byte_size) override;
  /// Prepare the input to be in the form expected by an OpenAI client,
  /// must call before accessing the data.
  Error PrepareForRequest();
  /// Get the contiguous request body string
  std::string& GetRequestBody() { return data_str_; }

 private:
  explicit OpenAiInferInput(
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype);

  std::vector<int64_t> shape_;
  size_t byte_size_{0};

  std::vector<const uint8_t*> bufs_;
  std::vector<size_t> buf_byte_sizes_;
  std::string data_str_;
};

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
