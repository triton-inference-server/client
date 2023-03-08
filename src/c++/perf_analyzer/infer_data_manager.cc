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

#include "infer_data_manager.h"

namespace triton { namespace perfanalyzer {

cb::Error
InferDataManager::Init()
{
  // FIXME TKG update this to create a mapping of infer input objects
  // similar to shm's CreateAndPopulateInputMemoryRegion
  return cb::Error::Success;
}

cb::Error
InferDataManager::InitInferDataInput(
    const std::string& name, const ModelTensor& model_tensor,
    InferData& infer_data)
{
  // FIXME TKG update this to no longer create and populate inputs.
  // How can it determine validity of valid_inputs?

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


  const uint8_t* data_ptr{nullptr};
  size_t batch1_bytesize;
  RETURN_IF_ERROR(data_loader_->GetInputData(
      model_tensor, 0, 0, &data_ptr, &batch1_bytesize));

  // Add optional input to request if data was found
  if (data_ptr != nullptr) {
    infer_data.valid_inputs_.push_back(infer_input);
  }

  if (!shape.empty()) {
    size_t max_count = (parser_->MaxBatchSize() == 0) ? 1 : batch_size_;
    for (size_t i = 0; i < max_count; ++i) {
      RETURN_IF_ERROR(infer_input->AppendRaw(data_ptr, batch1_bytesize));
    }
  }
  return cb::Error::Success;
}

cb::Error
InferDataManager::InitInferDataOutput(
    const std::string& name, InferData& infer_data)
{
  cb::InferRequestedOutput* requested_output;
  RETURN_IF_ERROR(
      cb::InferRequestedOutput::Create(&requested_output, backend_kind_, name));
  infer_data.outputs_.push_back(requested_output);

  return cb::Error::Success;
}

cb::Error
InferDataManager::UpdateInputs(
    const int stream_index, const int step_index, InferData& infer_data)
{
  // FIXME TKG update this to point valid_inputs to the correct pregenerated
  // inputs via stream_index/step_index

  // Reset inputs for this inference request
  infer_data.valid_inputs_.clear();

  for (const auto& input : infer_data.inputs_) {
    RETURN_IF_ERROR(input->Reset());

    const auto& model_input = (*(parser_->Inputs()))[input->Name()];

    const uint8_t* data_ptr{nullptr};
    size_t batch1_bytesize;
    const int* set_shape_values = nullptr;
    int set_shape_value_cnt = 0;

    // Number of missing pieces of data for optional inputs
    int missing_data_cnt = 0;

    for (size_t i = 0; i < batch_size_; ++i) {
      std::vector<int64_t> shape;
      RETURN_IF_ERROR(data_loader_->GetInputShape(
          model_input, stream_index,
          (step_index + i) % data_loader_->GetTotalSteps(stream_index),
          &shape));
      if ((parser_->MaxBatchSize() != 0) && (!model_input.is_shape_tensor_)) {
        shape.insert(shape.begin(), (int64_t)batch_size_);
      }
      if (!shape.empty()) {
        if (i == 0) {
          input->SetShape(shape);
        } else {
          if (!std::equal(shape.begin(), shape.end(), input->Shape().begin())) {
            return cb::Error(
                "can not batch tensors with different shapes together "
                "(input '" +
                    input->Name() + "' expected shape " +
                    ShapeVecToString(input->Shape(), true /* skip_first */) +
                    " and received " +
                    ShapeVecToString(shape, true /* skip_first */),
                pa::GENERIC_ERROR);
          }
        }
      }
      data_ptr = nullptr;
      // FIXME TKG -- the hardcoded 0 here seems wrong?
      RETURN_IF_ERROR(data_loader_->GetInputData(
          model_input, stream_index,
          (step_index + i) % data_loader_->GetTotalSteps(0), &data_ptr,
          &batch1_bytesize));

      // Update number of missing pieces of data for optional inputs to
      // potentially detect error
      if (data_ptr == nullptr) {
        missing_data_cnt++;
        continue;
      }

      if (!model_input.is_shape_tensor_) {
        RETURN_IF_ERROR(input->AppendRaw(data_ptr, batch1_bytesize));
      } else {
        if (i == 0) {
          // Set data only once for shape tensors
          RETURN_IF_ERROR(input->AppendRaw(data_ptr, batch1_bytesize));
          set_shape_values = (const int*)data_ptr;
          set_shape_value_cnt = batch1_bytesize / sizeof(int);
        } else {
          // Validate if the shape values are identical in the batch
          bool is_identical = true;
          if ((size_t)set_shape_value_cnt != (batch1_bytesize / sizeof(int))) {
            is_identical = false;
          } else {
            for (int i = 0; i < set_shape_value_cnt; i++) {
              if (*(set_shape_values + i) != *((const int*)data_ptr + i)) {
                is_identical = false;
                break;
              }
            }
          }
          if (!is_identical) {
            return cb::Error(
                "can not batch shape tensors with different values together "
                "(input '" +
                    input->Name() + "' expected shape values" +
                    ShapeTensorValuesToString(
                        set_shape_values, set_shape_value_cnt) +
                    " and received " +
                    ShapeTensorValuesToString(
                        (int*)data_ptr, (batch1_bytesize / sizeof(int))),
                pa::GENERIC_ERROR);
          }
        }
      }
    }

    // If all optional inputs had data provided, this is a valid input. But if
    // some inferences in the batch provided data for an optional input and
    // some inferences did not, this is an invalid case and an error is
    // thrown.
    if (missing_data_cnt == 0) {
      infer_data.valid_inputs_.push_back(input);
    } else if (missing_data_cnt > 0 && missing_data_cnt < batch_size_) {
      return cb::Error(
          "For batch sizes larger than 1, the same set of inputs must be "
          "specified for each batch. You cannot use different set of "
          "optional "
          "inputs for each individual batch.");
    }
  }
  return cb::Error::Success;
}


}}  // namespace triton::perfanalyzer
