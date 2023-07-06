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

#include "infer_data_manager_base.h"

#include <algorithm>

namespace triton { namespace perfanalyzer {

cb::Error
InferDataManagerBase::GetInputData(
    const std::string& name, const ModelTensor& tensor, int stream_id,
    int step_id, std::vector<TensorData>& input_datas)
{
  size_t max_count = tensor.is_shape_tensor_ ? 1 : batch_size_;
  std::vector<int64_t> shape;
  std::vector<int64_t> prev_shape;

  for (size_t count = 0; count < max_count; count++) {
    int local_step_id =
        (step_id + count) % data_loader_->GetTotalSteps(stream_id);

    TensorData input_data;

    RETURN_IF_ERROR(
        data_loader_->GetInputShape(tensor, stream_id, local_step_id, &shape));
    if (!shape.empty()) {
      if (count == 0) {
        prev_shape = shape;
      } else {
        if (!std::equal(shape.begin(), shape.end(), prev_shape.begin())) {
          return cb::Error(
              "can not batch tensors with different shapes together "
              "(input '" +
                  name + "' expected shape " + ShapeVecToString(prev_shape) +
                  " and received " + ShapeVecToString(shape),
              pa::GENERIC_ERROR);
        }
      }
    }

    RETURN_IF_ERROR(data_loader_->GetInputData(
        tensor, stream_id, local_step_id, input_data));

    input_datas.push_back(input_data);
  }

  return cb::Error::Success;
}

cb::Error
InferDataManagerBase::ValidateShapeTensor(
    const ModelTensor& tensor, int stream_id, int step_id,
    const std::vector<TensorData>& input_datas)
{
  // Validate that steps 1 through N are exactly the same as step 0, since step
  // 0 is the only one we send for shape tensors
  for (size_t count = 1; count < batch_size_; count++) {
    int local_step_id =
        (step_id + count) % data_loader_->GetTotalSteps(stream_id);

    TensorData input_data;
    RETURN_IF_ERROR(data_loader_->GetInputData(
        tensor, stream_id, local_step_id, input_data));

    if (input_data.batch1_size != input_datas.back().batch1_size) {
      return cb::Error(
          "The shape tensors should be identical in a batch (mismatch "
          "in size)",
          pa::GENERIC_ERROR);
    }

    for (size_t data_idx = 0; data_idx < input_data.batch1_size; data_idx++) {
      if (*(input_data.data_ptr + data_idx) !=
          *(input_datas.back().data_ptr + data_idx)) {
        return cb::Error(
            "The shape tensors should be identical in a batch "
            "(mismatch in content)",
            pa::GENERIC_ERROR);
      }
    }
  }
  return cb::Error::Success;
}

cb::Error
InferDataManagerBase::InitInferData(InferData& infer_data)
{
  // Initialize inputs
  for (const auto& input : *(parser_->Inputs())) {
    RETURN_IF_ERROR(InitInferDataInput(input.first, input.second, infer_data));
  }

  for (const auto& output : *(parser_->Outputs())) {
    RETURN_IF_ERROR(InitInferDataOutput(output.first, infer_data));
  }

  return cb::Error::Success;
}

cb::Error
InferDataManagerBase::UpdateInferData(
    size_t thread_id, int stream_index, int step_index, InferData& infer_data)
{
  RETURN_IF_ERROR(data_loader_->ValidateIndexes(stream_index, step_index));
  RETURN_IF_ERROR(
      UpdateInputs(thread_id, stream_index, step_index, infer_data));
  RETURN_IF_ERROR(
      UpdateValidationOutputs(stream_index, step_index, infer_data));
  return cb::Error::Success;
}

cb::Error
InferDataManagerBase::UpdateValidationOutputs(
    int stream_index, int step_index, InferData& infer_data)
{
  RETURN_IF_ERROR(data_loader_->ValidateIndexes(stream_index, step_index));

  infer_data.expected_outputs_.clear();

  for (const auto& output : infer_data.outputs_) {
    const auto& model_output = (*(parser_->Outputs()))[output->Name()];

    TensorData output_data;
    const int* set_shape_values = nullptr;
    int set_shape_value_cnt = 0;

    std::vector<std::pair<const uint8_t*, size_t>> outputs;
    for (size_t i = 0; i < batch_size_; ++i) {
      RETURN_IF_ERROR(data_loader_->GetOutputData(
          output->Name(), stream_index,
          (step_index + i) % data_loader_->GetTotalSteps(0), output_data));
      if (!output_data.is_valid) {
        break;
      }

      outputs.emplace_back(output_data.data_ptr, output_data.batch1_size);
      // Shape tensor only need the first batch element
      if (model_output.is_shape_tensor_) {
        break;
      }
    }
    if (!outputs.empty()) {
      infer_data.expected_outputs_.emplace_back(std::move(outputs));
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
