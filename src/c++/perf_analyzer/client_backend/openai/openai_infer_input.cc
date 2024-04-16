// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "openai_infer_input.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

Error
OpenAiInferInput::Create(
    InferInput** infer_input, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  OpenAiInferInput* local_infer_input =
      new OpenAiInferInput(name, dims, datatype);

  *infer_input = local_infer_input;
  return Error::Success;
}

Error
OpenAiInferInput::SetShape(const std::vector<int64_t>& shape)
{
  shape_ = shape;
  return Error::Success;
}

Error
OpenAiInferInput::Reset()
{
  data_str_.clear();

  bufs_.clear();
  buf_byte_sizes_.clear();
  byte_size_ = 0;
  return Error::Success;
}

Error
OpenAiInferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  data_str_.clear();

  byte_size_ += input_byte_size;

  bufs_.push_back(input);
  buf_byte_sizes_.push_back(input_byte_size);
  return Error::Success;
}

Error
OpenAiInferInput::RawData(const uint8_t** buf, size_t* byte_size)
{
  // TMA-1775 - handle multi-batch case
  *buf = bufs_[0];
  *byte_size = buf_byte_sizes_[0];
  return Error::Success;
}

Error
OpenAiInferInput::PrepareForRequest()
{
  // Reset position so request sends entire input.
  if (data_str_.empty() && (byte_size_ != 0)) {
    for (size_t i = 0; i < bufs_.size(); ++i) {
      data_str_.append(
          reinterpret_cast<const char*>(bufs_[i]), buf_byte_sizes_[i]);
    }
  }
  return Error::Success;
}

OpenAiInferInput::OpenAiInferInput(
    const std::string& name, const std::vector<int64_t>& dims,
    const std::string& datatype)
    : InferInput(BackendKind::OPENAI, name, datatype), shape_(dims)
{
}

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
