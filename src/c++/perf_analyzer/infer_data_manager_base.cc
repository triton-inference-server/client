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

#include <algorithm>

#include "infer_data_manager_base.h"

namespace triton { namespace perfanalyzer {

cb::Error
InferDataManagerBase::UpdateInputs(
    int stream_index, int step_index, InferData& infer_data)
{
  // Validate update parameters here
  size_t data_stream_count = data_loader_->GetDataStreamsCount();
  if (stream_index < 0 || stream_index >= (int)data_stream_count) {
    return cb::Error(
        "stream_index for retrieving the data should be less than " +
            std::to_string(data_stream_count) + ", got " +
            std::to_string(stream_index),
        pa::GENERIC_ERROR);
  }
  size_t step_count = data_loader_->GetTotalSteps(stream_index);
  if (step_index < 0 || step_index >= (int)step_count) {
    return cb::Error(
        "step_id for retrieving the data should be less than " +
            std::to_string(step_count) + ", got " + std::to_string(step_index),
        pa::GENERIC_ERROR);
  }

  RETURN_IF_ERROR(SetInputs(stream_index, step_index, infer_data));

  return cb::Error::Success;
}

cb::Error
InferDataManagerBase::UpdateValidationOutputs(
    int stream_index, int step_index, InferData& infer_data)
{
  infer_data.expected_outputs_.clear();
  // Validate update parameters here
  size_t data_stream_count = data_loader_->GetDataStreamsCount();
  if (stream_index < 0 || stream_index >= (int)data_stream_count) {
    return cb::Error(
        "stream_index for retrieving the data should be less than " +
            std::to_string(data_stream_count) + ", got " +
            std::to_string(stream_index),
        pa::GENERIC_ERROR);
  }
  size_t step_count = data_loader_->GetTotalSteps(stream_index);
  if (step_index < 0 || step_index >= (int)step_count) {
    return cb::Error(
        "step_id for retrieving the data should be less than " +
            std::to_string(step_count) + ", got " + std::to_string(step_index),
        pa::GENERIC_ERROR);
  }

  for (const auto& output : infer_data.outputs_) {
    const auto& model_output = (*(parser_->Outputs()))[output->Name()];
    const uint8_t* data_ptr;
    size_t batch1_bytesize;
    const int* set_shape_values = nullptr;
    int set_shape_value_cnt = 0;

    std::vector<std::pair<const uint8_t*, size_t>> output_data;
    for (size_t i = 0; i < batch_size_; ++i) {
      RETURN_IF_ERROR(data_loader_->GetOutputData(
          output->Name(), stream_index,
          (step_index + i) % data_loader_->GetTotalSteps(0), &data_ptr,
          &batch1_bytesize));
      if (data_ptr == nullptr) {
        break;
      }
      output_data.emplace_back(data_ptr, batch1_bytesize);
      // Shape tensor only need the first batch element
      if (model_output.is_shape_tensor_) {
        break;
      }
    }
    if (!output_data.empty()) {
      infer_data.expected_outputs_.emplace_back(std::move(output_data));
    }
  }
  return cb::Error::Success;
}

cb::Error
InferDataManagerBase::CreateInferInput(
    cb::InferInput** infer_input, const cb::BackendKind kind,
    const std::string& name, const std::vector<int64_t>& dims,
    const std::string& datatype)
{
  return cb::InferInput::Create(infer_input, kind, name, dims, datatype);
}

}}  // namespace triton::perfanalyzer
