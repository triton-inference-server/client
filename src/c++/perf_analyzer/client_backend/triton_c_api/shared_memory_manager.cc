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

#include "shared_infer_data_manager.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {

SharedInferDataManager::~SharedInferDataManager()
{
  UnregisterAll(TRITONSERVER_MEMORY_CPU);
  UnregisterAll(TRITONSERVER_MEMORY_GPU);
}

#ifdef TRITON_ENABLE_GPU
Error
SharedInferDataManager::RegisterCUDAMemory(
    const std::string& name, void* dev_ptr, const size_t byte_size,
    const int device_id)
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(mu_);

  // If name is already in shared_memory_map_ then return error saying already
  // registered
  if (shared_memory_map_.find(name) != shared_memory_map_.end()) {
    return Error(
        std::string("shared memory region '" + name + "' already in manager"));
  }

  shared_memory_map_.insert(std::make_pair(
      name, std::unique_ptr<MemoryInfo>(new MemoryInfo(
                name, 0 /* offset */, byte_size, dev_ptr,
                TRITONSERVER_MEMORY_GPU, device_id))));
  return Error::Success;
}
#endif  // TRITON_ENABLE_GPU

Error
SharedInferDataManager::RegisterSystemMemory(
    const std::string& name, void* ptr, const size_t byte_size)
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(mu_);

  // If name is already in shared_memory_map_ then return error saying already
  // registered
  if (shared_memory_map_.find(name) != shared_memory_map_.end()) {
    return Error("shared memory region '" + name + "' already in manager");
  }

  shared_memory_map_.insert(std::make_pair(
      name, std::make_unique<MemoryInfo>(
                name, 0 /* offset */, byte_size, ptr, TRITONSERVER_MEMORY_CPU,
                0 /* device id */)));

  return Error::Success;
}

Error
SharedInferDataManager::GetMemoryInfo(
    const std::string& name, size_t offset, void** shm_mapped_addr,
    TRITONSERVER_MemoryType* memory_type, int64_t* device_id)
{
  // protect shared_memory_map_ from concurrent access
  std::lock_guard<std::mutex> lock(mu_);

  auto it = shared_memory_map_.find(name);
  if (it == shared_memory_map_.end()) {
    return Error(
        std::string("Unable to find shared memory region: '" + name + "'"));
  }
  if (it->second->kind_ == TRITONSERVER_MEMORY_CPU) {
    *shm_mapped_addr =
        (void*)((uint8_t*)it->second->mapped_addr_ + it->second->offset_ + offset);
  } else {
    *shm_mapped_addr = (void*)((uint8_t*)it->second->mapped_addr_ + offset);
  }

  *memory_type = it->second->kind_;
  *device_id = it->second->device_id_;

  return Error::Success;
}


Error
SharedInferDataManager::Unregister(
    const std::string& name, TRITONSERVER_MemoryType memory_type)
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(mu_);

  return UnregisterHelper(name, memory_type);
}

Error
SharedInferDataManager::UnregisterAll(TRITONSERVER_MemoryType memory_type)
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(mu_);
  std::string error_message = "Failed to unregister the following ";
  std::vector<std::string> unregister_fails;

  if (memory_type == TRITONSERVER_MEMORY_CPU) {
    error_message += "system shared memory regions: ";
    for (auto& it : shared_memory_map_) {
      if (it.second->kind_ == TRITONSERVER_MEMORY_CPU) {
        Error err = UnregisterHelper(it.first, memory_type);
        if (!err.IsOk()) {
          unregister_fails.push_back(it.first);
        }
      }
    }
  } else if (memory_type == TRITONSERVER_MEMORY_GPU) {
    error_message += "cuda shared memory regions: ";
    for (auto& it : shared_memory_map_) {
      if (it.second->kind_ == TRITONSERVER_MEMORY_GPU) {
        Error err = UnregisterHelper(it.first, memory_type);
        if (!err.IsOk()) {
          unregister_fails.push_back(it.first);
        }
      }
    }
  }

  if (!unregister_fails.empty()) {
    for (auto unreg_fail : unregister_fails) {
      error_message += unreg_fail + " ,";
    }
    return Error(error_message);
  }

  return Error::Success;
}

Error
SharedInferDataManager::UnregisterHelper(
    const std::string& name, TRITONSERVER_MemoryType memory_type)
{
  // Must hold the lock on register_mu_ while calling this function.
  auto it = shared_memory_map_.find(name);

  if (it == shared_memory_map_.end()) {
    return Error("Shared memory region " + name + " doesn't exist.");
  }

  // Remove region information from shared_memory_map_
  shared_memory_map_.erase(it);

  return Error::Success;
}

}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
