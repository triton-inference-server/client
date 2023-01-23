// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "load_manager.h"

#include <algorithm>

#include "client_backend/client_backend.h"
#include "shm_utils.h"

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

#endif  // TRITON_ENABLE_GPU

namespace triton { namespace perfanalyzer {

namespace {

#ifdef TRITON_ENABLE_GPU
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

LoadManager::~LoadManager()
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
LoadManager::CheckHealth()
{
  // Check thread status to make sure that the load setting is
  // consistent to the one being reported
  // If some thread return early, main thread will return and
  // the worker thread's error message will be reported
  // when derived class destructor gets called.
  for (auto& thread_stat : threads_stat_) {
    if (!thread_stat->status_.IsOk()) {
      return cb::Error(
          "Failed to maintain requested inference load."
          " Worker thread(s) failed to generate concurrent requests.",
          pa::GENERIC_ERROR);
    }
    if (!thread_stat->cb_status_.IsOk()) {
      return cb::Error(
          "Failed to retrieve results from inference request.",
          pa::GENERIC_ERROR);
    }
  }
  return cb::Error::Success;
}

cb::Error
LoadManager::SwapTimestamps(TimestampVector& new_timestamps)
{
  TimestampVector total_timestamp;
  // Gather request timestamps with proper locking from all the worker
  // threads
  for (auto& thread_stat : threads_stat_) {
    std::lock_guard<std::mutex> lock(thread_stat->mu_);
    total_timestamp.insert(
        total_timestamp.end(), thread_stat->request_timestamps_.begin(),
        thread_stat->request_timestamps_.end());
    thread_stat->request_timestamps_.clear();
  }
  // Swap the results
  total_timestamp.swap(new_timestamps);
  return cb::Error::Success;
}

uint64_t
LoadManager::CountCollectedRequests()
{
  uint64_t num_of_requests = 0;
  for (auto& thread_stat : threads_stat_) {
    std::lock_guard<std::mutex> lock(thread_stat->mu_);
    num_of_requests += thread_stat->request_timestamps_.size();
  }
  return num_of_requests;
}

cb::Error
LoadManager::GetAccumulatedClientStat(cb::InferStat* contexts_stat)
{
  contexts_stat->completed_request_count = 0;
  contexts_stat->cumulative_receive_time_ns = 0;
  contexts_stat->cumulative_send_time_ns = 0;
  contexts_stat->cumulative_total_request_time_ns = 0;

  for (auto& thread_stat : threads_stat_) {
    std::lock_guard<std::mutex> lock(thread_stat->mu_);
    for (auto& context_stat : thread_stat->contexts_stat_) {
      contexts_stat->completed_request_count +=
          context_stat.completed_request_count;
      contexts_stat->cumulative_total_request_time_ns +=
          context_stat.cumulative_total_request_time_ns;
      contexts_stat->cumulative_send_time_ns +=
          context_stat.cumulative_send_time_ns;
      contexts_stat->cumulative_receive_time_ns +=
          context_stat.cumulative_receive_time_ns;
    }
  }
  return cb::Error::Success;
}

LoadManager::LoadManager(
    const bool async, const bool streaming, const int32_t batch_size,
    const size_t max_threads, const size_t sequence_length,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const uint64_t start_sequence_id, const uint64_t sequence_id_range,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory)
    : async_(async), streaming_(streaming), batch_size_(batch_size),
      max_threads_(max_threads), sequence_length_(sequence_length),
      shared_memory_type_(shared_memory_type),
      output_shm_size_(output_shm_size), start_sequence_id_(start_sequence_id),
      sequence_id_range_(sequence_id_range), parser_(parser), factory_(factory),
      using_json_data_(false), using_shared_memory_(false), curr_seq_id_(0)
{
  on_sequence_model_ =
      ((parser_->SchedulerType() == ModelParser::SEQUENCE) ||
       (parser_->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE));
}

void
LoadManager::InitManager(
    const size_t string_length, const std::string& string_data,
    const bool zero_input, std::vector<std::string>& user_data)
{
  data_loader_.reset(new DataLoader(batch_size_));

  auto status =
      InitManagerInputs(string_length, string_data, zero_input, user_data);
  THROW_IF_ERROR(status, "Failed to init manager inputs");

  if (shared_memory_type_ != SharedMemoryType::NO_SHARED_MEMORY) {
    THROW_IF_ERROR(InitSharedMemory(), "Unable to init shared memory");
  }
}

cb::Error
LoadManager::InitManagerInputs(
    const size_t string_length, const std::string& string_data,
    const bool zero_input, std::vector<std::string>& user_data)
{
  RETURN_IF_ERROR(factory_->CreateClientBackend(&backend_));

  // Read provided data
  if (!user_data.empty()) {
    if (IsDirectory(user_data[0])) {
      RETURN_IF_ERROR(data_loader_->ReadDataFromDir(
          parser_->Inputs(), parser_->Outputs(), user_data[0]));
    } else {
      using_json_data_ = true;
      for (const auto& json_file : user_data) {
        RETURN_IF_ERROR(data_loader_->ReadDataFromJSON(
            parser_->Inputs(), parser_->Outputs(), json_file));
      }
      distribution_ = std::uniform_int_distribution<uint64_t>(
          0, data_loader_->GetDataStreamsCount() - 1);
      std::cout << " Successfully read data for "
                << data_loader_->GetDataStreamsCount() << " stream/streams";
      if (data_loader_->GetDataStreamsCount() == 1) {
        std::cout << " with " << data_loader_->GetTotalSteps(0)
                  << " step/steps";
      }
      std::cout << "." << std::endl;
    }
  } else {
    RETURN_IF_ERROR(data_loader_->GenerateData(
        parser_->Inputs(), zero_input, string_length, string_data));
  }

  // Reserve the required vector space
  threads_stat_.reserve(max_threads_);

  return cb::Error::Success;
}

cb::Error
LoadManager::CreateMemoryRegion(
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
LoadManager::InitSharedMemory()
{
  using_shared_memory_ = true;

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
LoadManager::CopySharedMemory(
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

void
LoadManager::StopWorkerThreads()
{
  early_exit = true;
  // wake up all threads
  wake_signal_.notify_all();

  size_t cnt = 0;
  for (auto& thread : threads_) {
    thread.join();
    if (!threads_stat_[cnt]->status_.IsOk()) {
      std::cerr << "Thread [" << cnt
                << "] had error: " << (threads_stat_[cnt]->status_)
                << std::endl;
    }
    if (!threads_stat_[cnt]->cb_status_.IsOk()) {
      std::cerr << "Thread [" << cnt
                << "] had error: " << (threads_stat_[cnt]->cb_status_)
                << std::endl;
    }
    cnt++;
  }
  threads_.clear();
}

void
LoadManager::UnpauseAllSequences()
{
  for (auto seq : sequence_stat_) {
    std::lock_guard<std::mutex> guard(seq->mtx_);
    seq->paused_ = false;
  }
}

}}  // namespace triton::perfanalyzer
