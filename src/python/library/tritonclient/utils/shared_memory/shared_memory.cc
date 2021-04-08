// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "shared_memory.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "shared_memory_handle.h"

//==============================================================================
// SharedMemoryControlContext

namespace {

void*
SharedMemoryHandleCreate(
    std::string triton_shm_name, void* shm_addr, std::string shm_key,
    int shm_fd, size_t offset, size_t byte_size)
{
  SharedMemoryHandle* handle = new SharedMemoryHandle();
  handle->triton_shm_name_ = triton_shm_name;
  handle->base_addr_ = shm_addr;
  handle->shm_key_ = shm_key;
  handle->shm_fd_ = shm_fd;
  handle->offset_ = offset;
  handle->byte_size_ = byte_size;
  return reinterpret_cast<void*>(handle);
}

int
SharedMemoryRegionMap(
    int shm_fd, size_t offset, size_t byte_size, void** shm_addr)
{
  // map shared memory to process address space
  *shm_addr = mmap(NULL, byte_size, PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (*shm_addr == MAP_FAILED) {
    return -1;
  }

  // close shared memory descriptor, return 0 if success else return -1
  return close(shm_fd);
}

}  // namespace

int
SharedMemoryRegionCreate(
    const char* triton_shm_name, const char* shm_key, size_t byte_size,
    void** shm_handle)
{
  // get shared memory region descriptor
  int shm_fd = shm_open(shm_key, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return -2;
  }

  // extend shared memory object as by default it's initialized with size 0
  int res = ftruncate(shm_fd, byte_size);
  if (res == -1) {
    return -3;
  }

  // get base address of shared memory region
  void* shm_addr = nullptr;
  int err = SharedMemoryRegionMap(shm_fd, 0, byte_size, &shm_addr);
  if (err == -1) {
    return -4;
  }

  // create a handle for the shared memory region
  *shm_handle = SharedMemoryHandleCreate(
      std::string(triton_shm_name), shm_addr, std::string(shm_key), shm_fd, 0,
      byte_size);
  return 0;
}

int
SharedMemoryRegionSet(
    void* shm_handle, size_t offset, size_t byte_size, const void* data)
{
  void* shm_addr =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle)->base_addr_;
  char* shm_addr_offset = reinterpret_cast<char*>(shm_addr);
  std::memcpy(shm_addr_offset + offset, data, byte_size);
  return 0;
}

int
GetSharedMemoryHandleInfo(
    void* shm_handle, char** shm_addr, const char** shm_key, int* shm_fd,
    size_t* offset, size_t* byte_size)
{
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  *shm_addr = reinterpret_cast<char*>(handle->base_addr_);
  *shm_key = handle->shm_key_.c_str();
  *shm_fd = handle->shm_fd_;
  *offset = handle->offset_;
  *byte_size = handle->byte_size_;
  return 0;
}

int
SharedMemoryRegionDestroy(void* shm_handle)
{
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  void* shm_addr = reinterpret_cast<char*>(handle->base_addr_);
  int status = munmap(shm_addr, handle->byte_size_);
  if (status == -1) {
    return -6;
  }

  int shm_fd = shm_unlink(handle->shm_key_.c_str());
  if (shm_fd == -1) {
    return -5;
  }

  return 0;
}

//==============================================================================
