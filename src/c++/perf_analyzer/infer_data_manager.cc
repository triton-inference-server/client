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


InferDataManager::InferDataManager(
    const int32_t batch_size, const SharedMemoryType shared_memory_type,
    const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    const std::shared_ptr<DataLoader>& data_loader)
    : batch_size_(batch_size), shared_memory_type_(shared_memory_type),
      output_shm_size_(output_shm_size), parser_(parser), factory_(factory),
      data_loader_(data_loader), backend_kind_(factory->Kind())
{
}

InferDataManager::~InferDataManager()
{
  cb::Error err;
  if (using_shared_memory_ && backend_.get() != nullptr) {
    err = backend_->UnregisterAllSharedMemory();
    if (!err.IsOk()) {
      std::cerr << "Unable to unregister all shared memory regions"
                << std::endl;
    }
    if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
      for (auto& region : shared_memory_regions_) {
        if (factory_->Kind() !=
            triton::perfanalyzer::clientbackend::BackendKind::TRITON_C_API) {
          err = backend_->UnmapSharedMemory(
              shared_memory_regions_[region.first].data_.get(),
              shared_memory_regions_[region.first].byte_size_);
          if (!err.IsOk()) {
            std::cerr << "Unable to unmap shared memory with key ("
                      << region.first << "): Starting: "
                      << static_cast<void*>(
                             shared_memory_regions_[region.first].data_.get())
                      << ", size: "
                      << shared_memory_regions_[region.first].byte_size_
                      << std::endl;
          }
          err = backend_->UnlinkSharedMemoryRegion(region.first);
          if (!err.IsOk()) {
            std::cerr << "Unable to unlink shared memory with key: "
                      << region.first << std::endl;
          }
        }
      }
    }
  }
}


cb::Error
InferDataManager::InitSharedMemory()
{
  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    return cb::Error::Success;
  }

  using_shared_memory_ = true;
  // TMA-1062 remove the factory from this class and use only the backend
  RETURN_IF_ERROR(factory_->CreateClientBackend(&backend_));
  // Calling this function for the clean start
  backend_->UnregisterAllSharedMemory();

  // Allocate the shared memory for outputs
  for (const auto& output : *(parser_->Outputs())) {
    int64_t batch1_bytesize =
        ByteSize(output.second.shape_, output.second.datatype_);
    if (batch1_bytesize < 0) {
      batch1_bytesize = output_shm_size_;
    }
    uint8_t* output_shm_ptr;
    size_t alloc_size = batch1_bytesize * batch_size_;
    std::string region_name(TensorToRegionName(output.first));
    RETURN_IF_ERROR(CreateMemoryRegion(
        region_name, shared_memory_type_, alloc_size,
        reinterpret_cast<void**>(&output_shm_ptr)));
  }

  for (const auto& input : *(parser_->Inputs())) {
    for (int i = 0; i < (int)data_loader_->GetDataStreamsCount(); i++) {
      for (int j = 0; j < (int)data_loader_->GetTotalSteps(i);
           j += batch_size_) {
        // Extract the data for requested batch size
        std::vector<const uint8_t*> data_ptrs;
        std::vector<size_t> byte_size;
        size_t alloc_size = 0;
        size_t count = 0;
        size_t max_count = input.second.is_shape_tensor_ ? 1 : batch_size_;
        std::vector<int64_t> shape;
        std::vector<int64_t> prev_shape;
        while (count < max_count) {
          const uint8_t* data_ptr;
          size_t batch1_bytesize;

          RETURN_IF_ERROR(data_loader_->GetInputShape(
              input.second, i, (j + count) % data_loader_->GetTotalSteps(i),
              &shape));
          if (!shape.empty()) {
            if (count == 0) {
              prev_shape = shape;
            } else {
              if (!std::equal(shape.begin(), shape.end(), prev_shape.begin())) {
                return cb::Error(
                    "can not batch tensors with different shapes together "
                    "(input '" +
                        input.first + "' expected shape " +
                        ShapeVecToString(prev_shape) + " and received " +
                        ShapeVecToString(shape),
                    pa::GENERIC_ERROR);
              }
            }
          }

          RETURN_IF_ERROR(data_loader_->GetInputData(
              input.second, i, (j + count) % data_loader_->GetTotalSteps(i),
              &data_ptr, &batch1_bytesize));
          data_ptrs.push_back(data_ptr);
          byte_size.push_back(batch1_bytesize);
          alloc_size += batch1_bytesize;
          count++;
        }

        // Validate if the shape tensors specified in the batch are identical.
        while (count < batch_size_) {
          const uint8_t* data_ptr;
          size_t batch1_bytesize;
          RETURN_IF_ERROR(data_loader_->GetInputData(
              input.second, i, (j + count) % data_loader_->GetTotalSteps(i),
              &data_ptr, &batch1_bytesize));
          if (batch1_bytesize != byte_size.back()) {
            return cb::Error(
                "The shape tensors should be identical in a batch (mismatch "
                "in size)",
                pa::GENERIC_ERROR);
          }

          for (size_t data_idx = 0; data_idx < batch1_bytesize; data_idx++) {
            if (*(data_ptr + data_idx) != *(data_ptrs.back() + data_idx)) {
              return cb::Error(
                  "The shape tensors should be identical in a batch "
                  "(mismatch in content)",
                  pa::GENERIC_ERROR);
            }
          }
          count++;
        }

        // Generate the shared memory region name
        std::string region_name(
            TensorToRegionName(input.first) + "_" + std::to_string(i) + "_" +
            std::to_string(j));
        uint8_t* input_shm_ptr;
        RETURN_IF_ERROR(CreateMemoryRegion(
            region_name, shared_memory_type_, alloc_size,
            reinterpret_cast<void**>(&input_shm_ptr)));
        RETURN_IF_ERROR(CopySharedMemory(
            input_shm_ptr, data_ptrs, byte_size, input.second.is_shape_tensor_,
            region_name));
      }
    }
  }
  return cb::Error::Success;
}

cb::Error
InferDataManager::CreateMemoryRegion(
    const std::string& shm_region_name, const SharedMemoryType& memory_type,
    const size_t byte_size, void** ptr)
{
  if (memory_type == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
    if (factory_->Kind() ==
        triton::perfanalyzer::clientbackend::BackendKind::TRITON_C_API) {
      *ptr = new uint8_t[byte_size];
      RETURN_IF_ERROR(
          backend_->RegisterSystemMemory(shm_region_name, *ptr, byte_size));

      // Set free as the destructor.
      shared_memory_regions_.emplace(
          std::piecewise_construct, std::forward_as_tuple(shm_region_name),
          std::forward_as_tuple(SharedMemoryData(
              byte_size,
              std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>(
                  reinterpret_cast<uint8_t*>(*ptr),
                  [](uint8_t* memory) { free(memory); }))));
    } else {
      std::string shm_key("/" + shm_region_name);
      int shm_fd_op;
      RETURN_IF_ERROR(
          backend_->CreateSharedMemoryRegion(shm_key, byte_size, &shm_fd_op));
      RETURN_IF_ERROR(backend_->MapSharedMemory(shm_fd_op, 0, byte_size, ptr));

      RETURN_IF_ERROR(backend_->RegisterSystemSharedMemory(
          shm_region_name, shm_key, byte_size));

      // No-op destruction
      shared_memory_regions_.emplace(
          std::piecewise_construct, std::forward_as_tuple(shm_region_name),
          std::forward_as_tuple(SharedMemoryData(
              byte_size,
              std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>(
                  reinterpret_cast<uint8_t*>(*ptr), [](uint8_t* memory) {}))));
    }
  } else if (memory_type == SharedMemoryType::CUDA_SHARED_MEMORY) {
#ifdef TRITON_ENABLE_GPU
    cudaError_t cuda_err = cudaMalloc((void**)ptr, byte_size);
    if (cuda_err != cudaSuccess) {
      return cb::Error(
          "unable to allocate memory of " + std::to_string(byte_size) +
              " bytes on gpu for output: " +
              std::string(cudaGetErrorString(cuda_err)),
          pa::GENERIC_ERROR);
    }

    if (factory_->Kind() ==
        triton::perfanalyzer::clientbackend::BackendKind::TRITON_C_API) {
      RETURN_IF_ERROR(
          backend_->RegisterCudaMemory(shm_region_name, *ptr, byte_size));

      // Set cudaFree as the destructor
      shared_memory_regions_.emplace(
          std::piecewise_construct, std::forward_as_tuple(shm_region_name),
          std::forward_as_tuple(SharedMemoryData(
              byte_size,
              std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>(
                  reinterpret_cast<uint8_t*>(*ptr),
                  [shm_region_name, byte_size](uint8_t* memory) {
                    cudaError_t cuda_err = cudaFree(memory);
                    if (cuda_err != cudaSuccess) {
                      std::cerr
                          << "Unable to free cuda shared memory for "
                          << shm_region_name
                          << ": Starting: " << static_cast<void*>(memory)
                          << ", size: " << byte_size
                          << " bytes, Details: " << cudaGetErrorString(cuda_err)
                          << std::endl;
                    }
                  }))));
    } else {
      cudaIpcMemHandle_t cuda_handle;
      RETURN_IF_ERROR(
          CreateCUDAIPCHandle(&cuda_handle, reinterpret_cast<void*>(*ptr)));
      RETURN_IF_ERROR(backend_->RegisterCudaSharedMemory(
          shm_region_name, cuda_handle, byte_size));

      // No operation required for deleting the memory
      shared_memory_regions_.emplace(
          std::piecewise_construct, std::forward_as_tuple(shm_region_name),
          std::forward_as_tuple(SharedMemoryData(
              byte_size,
              std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>(
                  reinterpret_cast<uint8_t*>(*ptr), [](uint8_t* memory) {}))));
    }
#endif  // TRITON_ENABLE_GPU
  } else {
    return cb::Error(
        "CreateMemoryRegion called with invalid memory region type.",
        pa::GENERIC_ERROR);
  }

  return cb::Error::Success;
}

cb::Error
InferDataManager::CopySharedMemory(
    uint8_t* input_shm_ptr, std::vector<const uint8_t*>& data_ptrs,
    std::vector<size_t>& byte_size, bool is_shape_tensor,
    std::string& region_name)
{
  if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
    // Populate the region with data
    size_t count = 0;
    size_t offset = 0;
    size_t max_count = is_shape_tensor ? 1 : batch_size_;
    while (count < max_count) {
      memcpy(input_shm_ptr + offset, data_ptrs[count], byte_size[count]);
      offset += byte_size[count];
      count++;
    }
  } else {
#ifdef TRITON_ENABLE_GPU
    // Populate the region with data
    size_t count = 0;
    size_t offset = 0;
    size_t max_count = is_shape_tensor ? 1 : batch_size_;
    while (count < max_count) {
      cudaError_t cuda_err = cudaMemcpy(
          (void*)(input_shm_ptr + offset), (void*)data_ptrs[count],
          byte_size[count], cudaMemcpyHostToDevice);
      if (cuda_err != cudaSuccess) {
        return cb::Error(
            "Failed to copy data to cuda shared memory for " + region_name +
                " : " + std::string(cudaGetErrorString(cuda_err)),
            pa::GENERIC_ERROR);
      }
      offset += byte_size[count];
      count++;
    }
#endif  // TRITON_ENABLE_GPU
  }
  return cb::Error::Success;
}

cb::Error
InferDataManager::CreateInferInput(
    cb::InferInput** infer_input, const cb::BackendKind kind,
    const std::string& name, const std::vector<int64_t>& dims,
    const std::string& datatype)
{
  return cb::InferInput::Create(infer_input, kind, name, dims, datatype);
}

cb::Error
InferDataManager::PrepareInfer(InferData& infer_data)
{
  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    return PrepareInferNoSharedMemory(infer_data);
  } else {
    return PrepareInferSharedMemory(infer_data);
  }
}

cb::Error
InferDataManager::PrepareInferNoSharedMemory(InferData& infer_data)
{
  // Initialize inputs
  for (const auto& input : *(parser_->Inputs())) {
    const uint8_t* data_ptr{nullptr};
    size_t batch1_bytesize;
    // Set input shape before getting the input data
    std::vector<int64_t> shape;
    RETURN_IF_ERROR(data_loader_->GetInputShape(input.second, 0, 0, &shape));
    if (shape.empty() && (backend_kind_ == cb::BackendKind::TRITON)) {
      return cb::Error("unable to set shape for the input", pa::GENERIC_ERROR);
    }

    if ((parser_->MaxBatchSize() != 0) && (!input.second.is_shape_tensor_)) {
      shape.insert(shape.begin(), (int64_t)batch_size_);
    }

    cb::InferInput* infer_input;
    RETURN_IF_ERROR(CreateInferInput(
        &infer_input, backend_kind_, input.first, shape,
        input.second.datatype_));
    infer_data.inputs_.push_back(infer_input);


    data_ptr = nullptr;
    RETURN_IF_ERROR(data_loader_->GetInputData(
        input.second, 0, 0, &data_ptr, &batch1_bytesize));

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
  }

  for (const auto& output : *(parser_->Outputs())) {
    std::string region_name(TensorToRegionName(output.first));

    cb::InferRequestedOutput* requested_output;
    RETURN_IF_ERROR(cb::InferRequestedOutput::Create(
        &requested_output, backend_kind_, output.first));
    infer_data.outputs_.push_back(requested_output);
  }
  RETURN_IF_ERROR(UpdateValidationOutputs(
      infer_data.outputs_, 0, 0, infer_data.expected_outputs_));

  return cb::Error::Success;
}

cb::Error
InferDataManager::PrepareInferSharedMemory(InferData& infer_data)
{
  for (const auto& input : *(parser_->Inputs())) {
    std::string region_name(
        TensorToRegionName(input.first) + "_" + std::to_string(0) + "_" +
        std::to_string(0));

    std::vector<int64_t> shape;
    RETURN_IF_ERROR(data_loader_->GetInputShape(input.second, 0, 0, &shape));
    if (!shape.empty()) {
      if ((parser_->MaxBatchSize() != 0) && (!input.second.is_shape_tensor_)) {
        shape.insert(shape.begin(), (int64_t)batch_size_);
      }
    } else {
      return cb::Error("unable to set shape for the input", pa::GENERIC_ERROR);
    }

    cb::InferInput* infer_input;
    RETURN_IF_ERROR(CreateInferInput(
        &infer_input, backend_kind_, input.first, shape,
        input.second.datatype_));
    infer_data.inputs_.push_back(infer_input);

    // FIXME: TMA-765 - Shared memory mode does not support optional inputs,
    // currently, and will be implemented in the associated story.
    infer_data.valid_inputs_.push_back(infer_input);

    RETURN_IF_ERROR(infer_input->SetSharedMemory(
        region_name, shared_memory_regions_[region_name].byte_size_));
    const uint8_t* data_ptr{nullptr};
  }

  for (const auto& output : *(parser_->Outputs())) {
    std::string region_name(TensorToRegionName(output.first));

    cb::InferRequestedOutput* requested_output;
    RETURN_IF_ERROR(cb::InferRequestedOutput::Create(
        &requested_output, backend_kind_, output.first));
    infer_data.outputs_.push_back(requested_output);

    RETURN_IF_ERROR(requested_output->SetSharedMemory(
        region_name, shared_memory_regions_[region_name].byte_size_));
  }

  return cb::Error::Success;
}

cb::Error
InferDataManager::UpdateValidationOutputs(
    const std::vector<const cb::InferRequestedOutput*>& outputs,
    int stream_index, int step_index,
    std::vector<std::vector<std::pair<const uint8_t*, size_t>>>& data)
{
  data.clear();
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

  for (const auto& output : outputs) {
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
      data.emplace_back(std::move(output_data));
    }
  }
  return cb::Error::Success;
}

cb::Error
InferDataManager::SetInputs(
    const std::vector<cb::InferInput*>& inputs,
    std::vector<cb::InferInput*>& valid_inputs, const int stream_index,
    const int step_index)
{
  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    return SetInputsNoSharedMemory(
        inputs, valid_inputs, stream_index, step_index);
  } else {
    return SetInputsSharedMemory(inputs, stream_index, step_index);
  }
}

cb::Error
InferDataManager::SetInputsNoSharedMemory(
    const std::vector<cb::InferInput*>& inputs,
    std::vector<cb::InferInput*>& valid_inputs, const int stream_index,
    const int step_index)
{
  // Reset inputs for this inference request
  valid_inputs.clear();

  for (const auto& input : inputs) {
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
      valid_inputs.push_back(input);
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

cb::Error
InferDataManager::SetInputsSharedMemory(
    const std::vector<cb::InferInput*>& inputs, const int stream_index,
    const int step_index)
{
  for (const auto& input : inputs) {
    RETURN_IF_ERROR(input->Reset());
    const auto& model_input = (*(parser_->Inputs()))[input->Name()];

    std::string region_name(
        TensorToRegionName(input->Name()) + '_' + std::to_string(stream_index) +
        "_" + std::to_string(step_index));

    std::vector<int64_t> shape;
    RETURN_IF_ERROR(data_loader_->GetInputShape(
        model_input, stream_index, step_index, &shape));
    if (!shape.empty()) {
      if ((parser_->MaxBatchSize() != 0) && (!model_input.is_shape_tensor_)) {
        shape.insert(shape.begin(), (int64_t)batch_size_);
      }
      input->SetShape(shape);
    }
    RETURN_IF_ERROR(input->SetSharedMemory(
        region_name, shared_memory_regions_[region_name].byte_size_));
  }
  return cb::Error::Success;
}

}}  // namespace triton::perfanalyzer
