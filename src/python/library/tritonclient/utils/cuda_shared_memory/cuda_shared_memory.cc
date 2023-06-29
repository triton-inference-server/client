// Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_shared_memory.h"

#include <cuda_runtime_api.h>
#include <cstring>
#include <iostream>
#include "../shared_memory/shared_memory_handle.h"

extern "C" {
#include "cencode.h"
}

//==============================================================================
// SharedMemoryControlContext

namespace {

void*
CudaSharedMemoryHandleCreate(
    std::string triton_shm_name, cudaIpcMemHandle_t cuda_shm_handle,
    void* base_addr, size_t byte_size, int device_id)
{
  SharedMemoryHandle* handle = new SharedMemoryHandle();
  handle->triton_shm_name_ = triton_shm_name;
  handle->cuda_shm_handle_ = cuda_shm_handle;
  handle->base_addr_ = base_addr;
  handle->byte_size_ = byte_size;
  handle->device_id_ = device_id;
  handle->offset_ = 0;
  handle->shm_key_ = "";
  handle->shm_fd_ = 0;
  return reinterpret_cast<void*>(handle);
}

int
SupportUVA(int shm_device_id, int ext_device_id)
{
  int support_uva = 1;
  cudaError_t err = cudaDeviceGetAttribute(
      &support_uva, cudaDevAttrUnifiedAddressing, shm_device_id);
  if (err != cudaSuccess) {
    return -6;
  }
  if ((support_uva != 0) && (ext_device_id != -1)) {
    err = cudaDeviceGetAttribute(
        &support_uva, cudaDevAttrUnifiedAddressing, ext_device_id);
    if (err != cudaSuccess) {
      return -6;
    }
  }
  if (support_uva == 0) {
    return -7;
  }
  return 0;
}

}  // namespace

int
CudaSharedMemoryRegionCreate(
    const char* triton_shm_name, size_t byte_size, int device_id,
    void** cuda_shm_handle)
{
  // remember previous device and set to new device
  int previous_device;
  cudaGetDevice(&previous_device);
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -1;
  }

  void* base_addr;
  cudaIpcMemHandle_t cuda_handle;

  // Allocate data and create cuda IPC handle for data on the gpu
  err = cudaMalloc(&base_addr, byte_size);
  err = cudaIpcGetMemHandle(&cuda_handle, base_addr);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -2;
  }

  // create a handle for the shared memory region
  *cuda_shm_handle = CudaSharedMemoryHandleCreate(
      std::string(triton_shm_name), cuda_handle, base_addr, byte_size,
      device_id);

  // Set device to previous GPU
  cudaSetDevice(previous_device);

  return 0;
}

int
CudaSharedMemoryGetRawHandle(
    void* cuda_shm_handle, char** serialized_raw_handle)
{
  if (cuda_shm_handle == nullptr) {
    return -1;
  }

  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle);

  // Encode the handle object to base64
  base64_encodestate es;
  base64_init_encodestate(&es);
  size_t handle_size = sizeof(cudaIpcMemHandle_t);
  *serialized_raw_handle = (char*)malloc(handle_size * 2); /* ~4/3 x input */
  int offset = base64_encode_block(
      (char*)((void*)&handle->cuda_shm_handle_), handle_size,
      *serialized_raw_handle, &es);
  base64_encode_blockend(*serialized_raw_handle + offset, &es);
  int padding_size =
      base64_encode_blockend(*serialized_raw_handle + offset, &es);
  offset += (padding_size - 1);
  // The base64_encode_blockend does not null-terminate the string but adds
  // the new line character. Adding the null character here for proper
  // termination of ctypes.
  (*serialized_raw_handle)[offset] = '\0';

  return 0;
}

int
CudaSharedMemoryRegionSet(
    void* cuda_shm_handle, size_t offset, size_t byte_size, const void* data,
    int device_id)
{
  auto lhandle = reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle);

  {
    // Unified virtual addressing is the prerequisite for cudaMemcpyDefault,
    // unsupported platform is rare so only for sanity check.
    auto res = SupportUVA(lhandle->device_id_, device_id);
    if (res != 0) {
      return res;
    }
  }

  // Copy data into cuda shared memory
  void* base_addr = lhandle->base_addr_;
  cudaError_t err = cudaMemcpy(
      reinterpret_cast<uint8_t*>(base_addr) + offset, data, byte_size,
      cudaMemcpyDefault);
  if (err != cudaSuccess) {
    return -3;
  }

  return 0;
}

int
GetCudaSharedMemoryHandleInfo(
    void* shm_handle, void** shm_addr, size_t* offset, size_t* byte_size,
    int* device_id)
{
  auto handle = reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  *shm_addr = handle->base_addr_;
  *offset = handle->offset_;
  *byte_size = handle->byte_size_;
  *device_id = handle->device_id_;
  return 0;
}

// Must call CudaSharedMemoryReleaseBuffer to destroy 'new' object
// after writing into results. Numpy cannot read buffer from GPU and hence
// this is needed to maintain a copy of the data on GPU shared memory.
int
CudaSharedMemoryAllocateAndReadToHostBuffer(void* shm_handle, char** ptr)
{
  auto lhandle = reinterpret_cast<SharedMemoryHandle*>(shm_handle);

  {
    // Unified virtual addressing is the prerequisite for cudaMemcpyDefault,
    // unsupported platform is rare so only for sanity check.
    auto res = SupportUVA(lhandle->device_id_, -1 /* ext_device_id */);
    if (res != 0) {
      return res;
    }
  }

  *ptr = new char[lhandle->byte_size_];
  cudaError_t err = cudaMemcpy(
      *ptr, lhandle->base_addr_, lhandle->byte_size_, cudaMemcpyDefault);
  if (err != cudaSuccess) {
    return -5;
  }
  return 0;
}

int
CudaSharedMemoryReleaseHostBuffer(char* ptr)
{
  if (ptr) {
    delete ptr;
  }
  return 0;
}

int
CudaSharedMemoryRegionDestroy(void* cuda_shm_handle)
{
  // remember previous device and set to new device
  int previous_device;
  cudaGetDevice(&previous_device);
  cudaError_t err = cudaSetDevice(
      reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle)->device_id_);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -1;
  }

  SharedMemoryHandle* shm_hand =
      reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle);

  // Free GPU device memory
  err = cudaFree(shm_hand->base_addr_);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -4;
  }

  // Set device to previous GPU
  cudaSetDevice(previous_device);

  return 0;
}

//==============================================================================
