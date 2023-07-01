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
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// SharedMemoryControlContext
int CudaSharedMemoryRegionCreate(
    const char* triton_shm_name, size_t byte_size, int device_id,
    void** cuda_shm_handle);
int CudaSharedMemoryGetRawHandle(
    void* cuda_shm_handle, char** serialized_raw_handle);
// 'device_id' should be -1 if 'data' is in system memory
int CudaSharedMemoryRegionSet(
    void* cuda_shm_handle, size_t offset, size_t byte_size, const void* data,
    int device_id);
int GetCudaSharedMemoryHandleInfo(
    void* shm_handle, void** shm_addr, size_t* offset, size_t* byte_size,
    int* device_id);
int CudaSharedMemoryAllocateAndReadToHostBuffer(void* shm_handle, char** ptr);
int CudaSharedMemoryReleaseHostBuffer(char* ptr);
int CudaSharedMemoryRegionDestroy(void* cuda_shm_handle);

int CudaStreamCreate(void** cuda_stream);
int CudaStreamDestroy(void* cuda_stream);
int CudaStreamSynchronize(void* cuda_stream);

//==============================================================================

#ifdef __cplusplus
}
#endif
