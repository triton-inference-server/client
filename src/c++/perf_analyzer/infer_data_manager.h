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

#include "client_backend/client_backend.h"
#include "constants.h"
#include "data_loader.h"
#include "infer_data.h"
#include "model_parser.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

namespace {

#ifdef TRITON_ENABLE_GPU

#include <cuda_runtime_api.h>

#define RETURN_IF_CUDA_ERR(FUNC)                               \
  {                                                            \
    const cudaError_t result = FUNC;                           \
    if (result != cudaSuccess) {                               \
      return cb::Error(                                        \
          "CUDA exception (line " + std::to_string(__LINE__) + \
              "): " + cudaGetErrorName(result) + " (" +        \
              cudaGetErrorString(result) + ")",                \
          pa::GENERIC_ERROR);                                  \
    }                                                          \
  }

cb::Error
CreateCUDAIPCHandle(
    cudaIpcMemHandle_t* cuda_handle, void* input_d_ptr, int device_id = 0)
{
  // Set the GPU device to the desired GPU
  RETURN_IF_CUDA_ERR(cudaSetDevice(device_id));

  //  Create IPC handle for data on the gpu
  RETURN_IF_CUDA_ERR(cudaIpcGetMemHandle(cuda_handle, input_d_ptr));

  return cb::Error::Success;
}

#endif  // TRITON_ENABLE_GPU

}  // namespace

/// Holds information about the shared memory locations
struct SharedMemoryData {
  SharedMemoryData(
      size_t byte_size,
      std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> data)
      : byte_size_(byte_size), data_(std::move(data))
  {
  }

  SharedMemoryData() {}

  // Byte size
  size_t byte_size_;

  // Unique pointer holding the shared memory data
  std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> data_;
};

/// Manages infer data to prepare an inference request and the resulting
/// inference output from triton server
class InferDataManager {
 public:
  InferDataManager(
      const int32_t batch_size, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader);

  virtual ~InferDataManager();

  /// Helper function to allocate and prepare shared memory.
  /// \return cb::Error object indicating success or failure.
  cb::Error InitSharedMemory();


  /// Creates inference input object
  /// \param infer_input Output parameter storing newly created inference input
  /// \param kind Backend kind
  /// \param name Name of inference input
  /// \param dims Shape of inference input
  /// \param datatype Data type of inference input
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error CreateInferInput(
      cb::InferInput** infer_input, const cb::BackendKind kind,
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype);

  /// Wrapper function to prepare the InferData for sending
  /// inference request.
  /// \param ctx The target InferData object.
  /// \return cb::Error object indicating success or failure.
  cb::Error PrepareInfer(InferData& ctx);


  /// Updates the expected output data to use for inference request. Empty
  /// vector will be returned if there is no expected output associated to the
  /// step.
  /// \param outputs The vector of outputs to get the expected data
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \param data The vector of pointer and size of the expected outputs
  /// \return cb::Error object indicating success or failure.
  cb::Error UpdateValidationOutputs(
      const std::vector<const cb::InferRequestedOutput*>& outputs,
      int stream_index, int step_index,
      std::vector<std::vector<std::pair<const uint8_t*, size_t>>>& data);

  /// Helper function to update the inputs
  /// \param inputs The vector of pointers to InferInput objects for all
  /// possible inputs, potentially including optional inputs with no provided
  /// data
  /// \param valid_inputs The vector of pointers to InferInput objects to be
  /// used for inference request.
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputs(
      const std::vector<cb::InferInput*>& inputs,
      std::vector<cb::InferInput*>& valid_inputs, const int stream_index,
      const int step_index);


 protected:
  /// Create a memory region.
  /// \return cb::Error object indicating success or failure.
  cb::Error CreateMemoryRegion(
      const std::string& shm_region_name, const SharedMemoryType& memory_type,
      const size_t byte_size, void** ptr);

  /// \brief Helper function to handle copying shared memory to the correct
  /// memory region
  /// \param input_shm_ptr Pointer to the shared memory for a specific input
  /// \param data_ptrs Pointer to the data for the batch
  /// \param byte_size Size of the data being copied
  /// \param is_shape_tensor Is the input a shape tensor
  /// \param region_name Name of the shared memory region
  /// \return cb::Error object indicating success or failure
  virtual cb::Error CopySharedMemory(
      uint8_t* input_shm_ptr, std::vector<const uint8_t*>& data_ptrs,
      std::vector<size_t>& byte_size, bool is_shape_tensor,
      std::string& region_name);

  /// Helper function to prepare the InferData for sending
  /// inference request.
  /// \param ctx The target InferData object.
  /// \return cb::Error object indicating success or failure.
  cb::Error PrepareInferNoSharedMemory(InferData& ctx);

  /// Helper function to prepare the InferData for sending
  /// inference request in shared memory. \param ctx The target
  /// InferData object. \return cb::Error object indicating
  /// success or failure.
  cb::Error PrepareInferSharedMemory(InferData& ctx);

  /// Helper function to update the inputs
  /// \param inputs The vector of pointers to InferInput objects for all
  /// possible inputs, potentially including optional inputs with no provided
  /// data
  /// \param valid_inputs The vector of pointers to InferInput objects to be
  /// used for inference request.
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputsNoSharedMemory(
      const std::vector<cb::InferInput*>& inputs,
      std::vector<cb::InferInput*>& valid_inputs, const int stream_index,
      const int step_index);


  /// Helper function to update the shared memory inputs
  /// \param inputs The vector of pointers to InferInput objects
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputsSharedMemory(
      const std::vector<cb::InferInput*>& inputs, const int stream_index,
      const int step_index);

  size_t batch_size_;
  SharedMemoryType shared_memory_type_;
  size_t output_shm_size_;
  std::shared_ptr<ModelParser> parser_;
  std::shared_ptr<cb::ClientBackendFactory> factory_;
  std::shared_ptr<DataLoader> data_loader_;
  std::unique_ptr<cb::ClientBackend> backend_;
  cb::BackendKind backend_kind_;
  // Map from shared memory key to its starting address and size
  std::unordered_map<std::string, SharedMemoryData> shared_memory_regions_;

  bool using_shared_memory_;
};

}}  // namespace triton::perfanalyzer
