// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <triton/core/tritonserver.h>

#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#include "../client_backend.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {

class SharedMemoryManager {
 public:
  SharedMemoryManager() = default;
  ~SharedMemoryManager();

#ifdef TRITON_ENABLE_GPU
  /// Add a memory block representing memory in CUDA (GPU) memory
  /// to the manager. Return an Error if a memory block of the same name
  /// already exists in the manager.
  /// \param name The name of the memory block.
  /// \param dev_ptr The device pointer
  /// \param byte_size The size, in bytes of the block.
  /// \param device id The GPU number the memory region is in.
  /// \return an Error indicating success or failure.
  Error RegisterCUDAMemory(
      const std::string& name, void* dev_ptr, const size_t byte_size,
      const int device_id);
#endif  // TRITON_ENABLE_GPU

  /// Add a system memory block to the manager.
  /// Return an Error if a shared memory block of the same name
  /// already exists in the manager.
  /// \param name The name of the memory block.
  /// \param ptr The device pointer
  /// \param byte_size The size, in bytes of the block.
  /// \return an Error indicating success or failure.
  Error RegisterSystemMemory(
      const std::string& name, void* ptr, const size_t byte_size);

  /// Get the access information for the shared memory block with the specified
  /// name. Return an Error if named block doesn't exist.
  /// \param name The name of the shared memory block to get.
  /// \param offset The offset in the block
  /// \param shm_mapped_addr Returns the pointer to the shared
  /// memory block with the specified name and offset
  /// \param memory_type Returns the type of the memory
  /// \param device_id Returns the device id associated with the
  /// memory block
  /// \return an Error indicating success or failure.
  Error GetMemoryInfo(
      const std::string& name, size_t offset, void** shm_mapped_addr,
      TRITONSERVER_MemoryType* memory_type, int64_t* device_id);

  /// Removes the named shared memory block of the specified type from
  /// the manager. Any future attempt to get the details of this block
  /// will result in an array till another block with the same name is
  /// added to the manager.
  /// \param name The name of the shared memory block to remove.
  /// \param memory_type The type of memory to unregister.
  /// \return an Error indicating success or failure.
  Error Unregister(
      const std::string& name, TRITONSERVER_MemoryType memory_type);

  /// Unregister all shared memory blocks of specified type from the manager.
  /// \param memory_type The type of memory to unregister.
  /// \return an Error indicating success or failure.
  Error UnregisterAll(TRITONSERVER_MemoryType memory_type);

 private:
  /// A helper function to remove the named shared memory blocks of
  /// specified type
  Error UnregisterHelper(
      const std::string& name, TRITONSERVER_MemoryType memory_type);

  /// A struct that records the shared memory regions registered by the shared
  /// memory manager.
  struct MemoryInfo {
    MemoryInfo(
        const std::string& name, const size_t offset, const size_t byte_size,
        void* mapped_addr, const TRITONSERVER_MemoryType kind,
        const int64_t device_id)
        : name_(name), offset_(offset), byte_size_(byte_size),
          mapped_addr_(mapped_addr), kind_(kind), device_id_(device_id)
    {
    }

    std::string name_;
    size_t offset_;
    size_t byte_size_;
    void* mapped_addr_;
    TRITONSERVER_MemoryType kind_;
    int64_t device_id_;
  };

  using SharedMemoryStateMap =
      std::map<std::string, std::unique_ptr<MemoryInfo>>;

  // A map between the name and the details of the associated
  // shared memory block
  SharedMemoryStateMap shared_memory_map_;

  // A mutex to protect the concurrent access to shared_memory_map_
  std::mutex mu_;
};
}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
