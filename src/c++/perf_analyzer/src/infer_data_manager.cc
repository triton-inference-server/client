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

#include "infer_data_manager.h"

#include <algorithm>

namespace triton { namespace perfanalyzer {

cb::Error
InferDataManager::Init()
{
  RETURN_IF_ERROR(CreateAndPopulateInputs());
  return cb::Error::Success;
}

cb::Error
InferDataManager::CreateAndPopulateInputs()
{
  // All combinations of thread + input + stream + step
  //
  for (size_t thread_id = 0; thread_id < max_threads_; thread_id++) {
    for (const auto& input : *(parser_->Inputs())) {
      const std::string& name = input.first;
      const ModelTensor& tensor = input.second;
      for (int stream_id = 0;
           stream_id < (int)data_loader_->GetDataStreamsCount(); stream_id++) {
        for (int step_id = 0;
             step_id < (int)data_loader_->GetTotalSteps(stream_id);
             step_id += 1) {
          RETURN_IF_ERROR(CreateAndPopulateInput(
              thread_id, name, tensor, stream_id, step_id));
        }
      }
    }
  }
  return cb::Error::Success;
}

cb::Error
InferDataManager::CreateAndPopulateInput(
    const size_t thread_id, const std::string& name, const ModelTensor& tensor,
    int stream_id, int step_id)
{
  std::vector<TensorData> input_datas;
  size_t count = 0;

  RETURN_IF_ERROR(GetInputData(name, tensor, stream_id, step_id, input_datas));

  if (tensor.is_shape_tensor_) {
    RETURN_IF_ERROR(
        ValidateShapeTensor(tensor, stream_id, step_id, input_datas));
  }

  std::vector<int64_t> shape;
  RETURN_IF_ERROR(
      data_loader_->GetInputShape(tensor, stream_id, step_id, &shape));
  if (!shape.empty()) {
    if ((parser_->MaxBatchSize() != 0) && (!tensor.is_shape_tensor_)) {
      shape.insert(shape.begin(), (int64_t)batch_size_);
    }
  }

  cb::InferInput* input;
  RETURN_IF_ERROR(
      CreateInferInput(&input, backend_kind_, name, shape, tensor.datatype_));


  // Number of missing pieces of data for optional inputs
  int missing_data_cnt = 0;
  int total_cnt = input_datas.size();

  for (size_t i = 0; i < total_cnt; i++) {
    if (!input_datas[i].is_valid) {
      missing_data_cnt++;
    } else {
      RETURN_IF_ERROR(input->AppendRaw(
          input_datas[i].data_ptr, input_datas[i].batch1_size));
    }
  }

  // If all optional inputs had data provided, this is a valid input. But if
  // some inferences in the batch provided data for an optional input and
  // some inferences did not, this is an invalid case and an error is
  // thrown.
  if (missing_data_cnt == 0) {
    inputs_.insert({{thread_id, name, stream_id, step_id}, input});
  } else if (missing_data_cnt > 0 && missing_data_cnt < total_cnt) {
    return cb::Error(
        "For batch sizes larger than 1, the same set of inputs must be "
        "specified for each batch. You cannot use different set of "
        "optional inputs for each individual batch.");
  }

  return cb::Error::Success;
}

cb::InferInput*
InferDataManager::GetInput(
    const size_t thread_id, const std::string& name, int stream_id, int step_id)
{
  auto input = inputs_.find({thread_id, name, stream_id, step_id});
  if (input == inputs_.end()) {
    return nullptr;
  } else {
    return input->second;
  }
}


cb::Error
InferDataManager::InitInferDataInput(
    const std::string& name, const ModelTensor& model_tensor,
    InferData& infer_data)
{
  std::vector<int64_t> shape;
  RETURN_IF_ERROR(data_loader_->GetInputShape(model_tensor, 0, 0, &shape));
  if (shape.empty() && (backend_kind_ == cb::BackendKind::TRITON)) {
    return cb::Error("unable to set shape for the input", pa::GENERIC_ERROR);
  }

  if ((parser_->MaxBatchSize() != 0) && (!model_tensor.is_shape_tensor_)) {
    shape.insert(shape.begin(), (int64_t)batch_size_);
  }

  cb::InferInput* infer_input;
  RETURN_IF_ERROR(CreateInferInput(
      &infer_input, backend_kind_, name, shape, model_tensor.datatype_));
  infer_data.inputs_.push_back(infer_input);


  TensorData input_data;
  RETURN_IF_ERROR(data_loader_->GetInputData(model_tensor, 0, 0, input_data));

  // Add optional input to request if data was found
  if (input_data.is_valid) {
    infer_data.valid_inputs_.push_back(infer_input);
  }

  if (!shape.empty()) {
    size_t max_count = (parser_->MaxBatchSize() == 0) ? 1 : batch_size_;
    for (size_t i = 0; i < max_count; ++i) {
      RETURN_IF_ERROR(
          infer_input->AppendRaw(input_data.data_ptr, input_data.batch1_size));
    }
  }

  AddInferDataParameters(infer_data);

  return cb::Error::Success;
}

cb::Error
InferDataManager::InitInferDataOutput(
    const std::string& name, const ModelTensor& model_tensor,
    InferData& infer_data)
{
  cb::InferRequestedOutput* requested_output;
  RETURN_IF_ERROR(cb::InferRequestedOutput::Create(
      &requested_output, backend_kind_, name, model_tensor.datatype_));
  infer_data.outputs_.push_back(requested_output);

  return cb::Error::Success;
}

cb::Error
InferDataManager::UpdateInputs(
    const size_t thread_id, const int stream_index, const int step_index,
    InferData& infer_data)
{
  // Reset inputs for this inference request
  infer_data.valid_inputs_.clear();

  for (const auto& input : infer_data.inputs_) {
    const auto& name = input->Name();

    cb::InferInput* tmp_input =
        GetInput(thread_id, name, stream_index, step_index);
    if (tmp_input != nullptr) {
      infer_data.valid_inputs_.push_back(tmp_input);
    }
  }
  return cb::Error::Success;
}


}}  // namespace triton::perfanalyzer
