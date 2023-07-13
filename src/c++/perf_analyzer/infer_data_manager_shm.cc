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

#include "infer_data_manager_shm.h"

#include <algorithm>

namespace triton { namespace perfanalyzer {

InferDataManagerShm::~InferDataManagerShm()
{
  cb::Error err;
  if (backend_.get() != nullptr) {
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
InferDataManagerShm::Init()
{
  // TMA-1062 remove the factory from this class and use only the backend
  RETURN_IF_ERROR(factory_->CreateClientBackend(&backend_));
  // Calling this function for the clean start
  backend_->UnregisterAllSharedMemory();

  RETURN_IF_ERROR(CreateOutputMemoryRegions());
  RETURN_IF_ERROR(CreateAndPopulateInputMemoryRegions());

  return cb::Error::Success;
}

cb::Error
InferDataManagerShm::CreateOutputMemoryRegions()
{
  // Allocate the shared memory for outputs
  for (const auto& output : *(parser_->Outputs())) {
    const std::string& name = output.first;
    const ModelTensor& tensor = output.second;
    int64_t batch1_bytesize = ByteSize(tensor.shape_, tensor.datatype_);
    if (batch1_bytesize < 0) {
      batch1_bytesize = output_shm_size_;
    }
    uint8_t* output_shm_ptr;
    size_t alloc_size = batch1_bytesize * batch_size_;
    std::string region_name(TensorToRegionName(name));
    RETURN_IF_ERROR(CreateMemoryRegion(
        region_name, shared_memory_type_, alloc_size,
        reinterpret_cast<void**>(&output_shm_ptr)));
  }
  return cb::Error::Success;
}

cb::Error
InferDataManagerShm::CreateAndPopulateInputMemoryRegions()
{
  // All combinations of input + stream + step
  //
  for (const auto& input : *(parser_->Inputs())) {
    const std::string& name = input.first;
    const ModelTensor& tensor = input.second;
    for (int stream_id = 0;
         stream_id < (int)data_loader_->GetDataStreamsCount(); stream_id++) {
      for (int step_id = 0;
           step_id < (int)data_loader_->GetTotalSteps(stream_id);
           step_id += 1) {
        RETURN_IF_ERROR(CreateAndPopulateInputMemoryRegion(
            name, tensor, stream_id, step_id));
      }
    }
  }
  return cb::Error::Success;
}

cb::Error
InferDataManagerShm::CreateAndPopulateInputMemoryRegion(
    const std::string& name, const ModelTensor& tensor, int stream_id,
    int step_id)
{
  std::vector<TensorData> input_datas;
  size_t count = 0;

  RETURN_IF_ERROR(GetInputData(name, tensor, stream_id, step_id, input_datas));

  if (tensor.is_shape_tensor_) {
    RETURN_IF_ERROR(
        ValidateShapeTensor(tensor, stream_id, step_id, input_datas));
  }

  size_t alloc_size = 0;
  for (size_t i = 0; i < input_datas.size(); i++) {
    if (!input_datas[i].is_valid) {
      return cb::Error(
          "Shared memory support in Perf Analyzer does not support "
          "optional inputs at this time");
    }
    alloc_size += input_datas[i].batch1_size;
  }

  // Generate the shared memory region name
  std::string region_name(
      TensorToRegionName(name) + "_" + std::to_string(stream_id) + "_" +
      std::to_string(step_id));
  uint8_t* input_shm_ptr;
  RETURN_IF_ERROR(CreateMemoryRegion(
      region_name, shared_memory_type_, alloc_size,
      reinterpret_cast<void**>(&input_shm_ptr)));
  RETURN_IF_ERROR(CopySharedMemory(
      input_shm_ptr, input_datas, tensor.is_shape_tensor_, region_name));

  return cb::Error::Success;
}

cb::Error
InferDataManagerShm::CreateMemoryRegion(
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
InferDataManagerShm::CopySharedMemory(
    uint8_t* input_shm_ptr, const std::vector<TensorData>& tensor_datas,
    bool is_shape_tensor, std::string& region_name)
{
  if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
    // Populate the region with data
    size_t count = 0;
    size_t offset = 0;
    size_t max_count = is_shape_tensor ? 1 : batch_size_;
    while (count < max_count) {
      memcpy(
          input_shm_ptr + offset, tensor_datas[count].data_ptr,
          tensor_datas[count].batch1_size);
      offset += tensor_datas[count].batch1_size;
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
          (void*)(input_shm_ptr + offset), (void*)tensor_datas[count].data_ptr,
          tensor_datas[count].batch1_size, cudaMemcpyHostToDevice);
      if (cuda_err != cudaSuccess) {
        return cb::Error(
            "Failed to copy data to cuda shared memory for " + region_name +
                " : " + std::string(cudaGetErrorString(cuda_err)),
            pa::GENERIC_ERROR);
      }
      offset += tensor_datas[count].batch1_size;
      count++;
    }
#endif  // TRITON_ENABLE_GPU
  }
  return cb::Error::Success;
}

cb::Error
InferDataManagerShm::InitInferDataInput(
    const std::string& name, const ModelTensor& model_tensor,
    InferData& infer_data)
{
  std::vector<int64_t> shape;
  RETURN_IF_ERROR(data_loader_->GetInputShape(model_tensor, 0, 0, &shape));
  if (!shape.empty()) {
    if ((parser_->MaxBatchSize() != 0) && (!model_tensor.is_shape_tensor_)) {
      shape.insert(shape.begin(), (int64_t)batch_size_);
    }
  } else {
    return cb::Error("unable to set shape for the input", pa::GENERIC_ERROR);
  }

  cb::InferInput* infer_input;
  RETURN_IF_ERROR(CreateInferInput(
      &infer_input, backend_kind_, name, shape, model_tensor.datatype_));
  infer_data.inputs_.push_back(infer_input);

  // FIXME: TMA-765 - Shared memory mode does not support optional inputs,
  // currently, and will be implemented in the associated story.
  infer_data.valid_inputs_.push_back(infer_input);

  std::string region_name(
      TensorToRegionName(name) + "_" + std::to_string(0) + "_" +
      std::to_string(0));
  RETURN_IF_ERROR(infer_input->SetSharedMemory(
      region_name, shared_memory_regions_[region_name].byte_size_));

  return cb::Error::Success;
}

cb::Error
InferDataManagerShm::InitInferDataOutput(
    const std::string& name, InferData& infer_data)
{
  cb::InferRequestedOutput* requested_output;
  RETURN_IF_ERROR(
      cb::InferRequestedOutput::Create(&requested_output, backend_kind_, name));
  infer_data.outputs_.push_back(requested_output);

  std::string region_name(TensorToRegionName(name));
  RETURN_IF_ERROR(requested_output->SetSharedMemory(
      region_name, shared_memory_regions_[region_name].byte_size_));

  return cb::Error::Success;
}

cb::Error
InferDataManagerShm::UpdateInputs(
    const size_t thread_id, const int stream_index, const int step_index,
    InferData& infer_data)
{
  for (const auto& input : infer_data.inputs_) {
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
