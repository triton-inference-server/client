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

std::string
TensorToRegionName(std::string name)
{
  // Remove slashes from the name, if any.
  name.erase(
      std::remove_if(
          name.begin(), name.end(),
          [](const char& c) { return ((c == '/') || (c == '\\')); }),
      name.end());
  return name;
}

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
      for (auto region : shared_memory_regions_) {
        err = backend_->UnmapSharedMemory(
            shared_memory_regions_[region.first].first,
            shared_memory_regions_[region.first].second);
        if (!err.IsOk()) {
          std::cerr << "Unable to unmap shared memory with key ("
                    << region.first << "): Starting: "
                    << static_cast<void*>(
                           shared_memory_regions_[region.first].first)
                    << ", size: " << shared_memory_regions_[region.first].second
                    << std::endl;
        }
        err = backend_->UnlinkSharedMemoryRegion(region.first);
        if (!err.IsOk()) {
          std::cerr << "Unable to unlink shared memory with key: "
                    << region.first << std::endl;
        }
      }
    } else if (shared_memory_type_ == SharedMemoryType::CUDA_SHARED_MEMORY) {
#ifdef TRITON_ENABLE_GPU
      for (auto region : shared_memory_regions_) {
        cudaError_t cuda_err =
            cudaFree(shared_memory_regions_[region.first].first);
        if (cuda_err != cudaSuccess) {
          std::cerr << "Unable to free cuda shared memory for " << region.first
                    << ": Starting: "
                    << static_cast<void*>(
                           shared_memory_regions_[region.first].first)
                    << ", size: " << shared_memory_regions_[region.first].second
                    << " bytes, Details: " << cudaGetErrorString(cuda_err)
                    << std::endl;
        }
      }
#endif  // TRITON_ENABLE_GPU
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
      using_json_data_(false), using_shared_memory_(false), next_seq_id_(1)
{
  on_sequence_model_ =
      ((parser_->SchedulerType() == ModelParser::SEQUENCE) ||
       (parser->SchedulerType() == ModelParser::ENSEMBLE_SEQUENCE));

  data_loader_.reset(new DataLoader(batch_size));
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
    if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
      std::string shm_key("/" + region_name);
      int shm_fd_op;
      RETURN_IF_ERROR(
          backend_->CreateSharedMemoryRegion(shm_key, alloc_size, &shm_fd_op));
      RETURN_IF_ERROR(backend_->MapSharedMemory(
          shm_fd_op, 0, alloc_size, (void**)&output_shm_ptr));

      shared_memory_regions_[region_name] =
          std::pair<uint8_t*, size_t>(output_shm_ptr, alloc_size);

      RETURN_IF_ERROR(backend_->RegisterSystemSharedMemory(
          region_name, shm_key, alloc_size));
    } else {
#ifdef TRITON_ENABLE_GPU
      cudaError_t cuda_err = cudaMalloc((void**)&output_shm_ptr, alloc_size);
      if (cuda_err != cudaSuccess) {
        return cb::Error(
            "unable to allocate memory of " + std::to_string(alloc_size) +
                " bytes on gpu for output " + output.first + " : " +
                std::string(cudaGetErrorString(cuda_err)),
            pa::GENERIC_ERROR);
      }
      shared_memory_regions_[region_name] =
          std::pair<uint8_t*, size_t>(output_shm_ptr, alloc_size);

      cudaIpcMemHandle_t cuda_handle;
      RETURN_IF_ERROR(CreateCUDAIPCHandle(&cuda_handle, (void*)output_shm_ptr));
      // Using GPU with device id 0
      RETURN_IF_ERROR(backend_->RegisterCudaSharedMemory(
          region_name, cuda_handle, alloc_size));
#endif  // TRITON_ENABLE_GPU
    }
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
                "The shape tensors should be identical in a batch (mismatch in "
                "size)",
                pa::GENERIC_ERROR);
          }

          for (size_t data_idx = 0; data_idx < batch1_bytesize; data_idx++) {
            if (*(data_ptr + data_idx) != *(data_ptrs.back() + data_idx)) {
              return cb::Error(
                  "The shape tensors should be identical in a batch (mismatch "
                  "in content)",
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
        if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
          std::string shm_key("/" + region_name);
          int shm_fd_ip;
          RETURN_IF_ERROR(backend_->CreateSharedMemoryRegion(
              shm_key, alloc_size, &shm_fd_ip));
          RETURN_IF_ERROR(backend_->MapSharedMemory(
              shm_fd_ip, 0, alloc_size, (void**)&input_shm_ptr));
          shared_memory_regions_[region_name] =
              std::pair<uint8_t*, size_t>(input_shm_ptr, alloc_size);

          // Populate the region with data
          size_t count = 0;
          size_t offset = 0;
          size_t max_count = input.second.is_shape_tensor_ ? 1 : batch_size_;
          while (count < max_count) {
            memcpy(input_shm_ptr + offset, data_ptrs[count], byte_size[count]);
            offset += byte_size[count];
            count++;
          }

          // Register the region with triton
          RETURN_IF_ERROR(backend_->RegisterSystemSharedMemory(
              region_name, shm_key, alloc_size));
        } else {
#ifdef TRITON_ENABLE_GPU
          cudaError_t cuda_err = cudaMalloc((void**)&input_shm_ptr, alloc_size);
          if (cuda_err != cudaSuccess) {
            return cb::Error(
                "unable to allocate memory of " + std::to_string(alloc_size) +
                    "bytes on gpu for input " + region_name + " : " +
                    std::string(cudaGetErrorString(cuda_err)),
                pa::GENERIC_ERROR);
          }

          shared_memory_regions_[region_name] =
              std::pair<uint8_t*, size_t>(input_shm_ptr, alloc_size);

          // Populate the region with data
          size_t count = 0;
          size_t offset = 0;
          size_t max_count = input.second.is_shape_tensor_ ? 1 : batch_size_;
          while (count < max_count) {
            cudaError_t cuda_err = cudaMemcpy(
                (void*)(input_shm_ptr + offset), (void*)data_ptrs[count],
                byte_size[count], cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
              return cb::Error(
                  "Failed to copy data to cuda shared memory for " +
                      region_name + " : " +
                      std::string(cudaGetErrorString(cuda_err)),
                  pa::GENERIC_ERROR);
            }
            offset += byte_size[count];
            count++;
          }

          cudaIpcMemHandle_t cuda_handle;
          RETURN_IF_ERROR(
              CreateCUDAIPCHandle(&cuda_handle, (void*)input_shm_ptr));

          // Register the region with triton
          RETURN_IF_ERROR(backend_->RegisterCudaSharedMemory(
              region_name, cuda_handle, alloc_size));
#endif  // TRITON_ENABLE_GPU
        }
      }
    }
  }
  return cb::Error::Success;
}

cb::Error
LoadManager::PrepareInfer(InferContext* ctx)
{
  // Initialize inputs
  for (const auto& input : *(parser_->Inputs())) {
    const uint8_t* data_ptr{nullptr};
    size_t batch1_bytesize;
    // Set input shape before getting the input data
    std::vector<int64_t> shape;
    RETURN_IF_ERROR(data_loader_->GetInputShape(input.second, 0, 0, &shape));
    if (shape.empty() && (backend_->Kind() == cb::BackendKind::TRITON)) {
      return cb::Error("unable to set shape for the input", pa::GENERIC_ERROR);
    }

    if ((parser_->MaxBatchSize() != 0) && (!input.second.is_shape_tensor_)) {
      shape.insert(shape.begin(), (int64_t)batch_size_);
    }

    cb::InferInput* infer_input;
    RETURN_IF_ERROR(cb::InferInput::Create(
        &infer_input, backend_->Kind(), input.first, shape,
        input.second.datatype_));
    ctx->inputs_.push_back(infer_input);

    data_ptr = nullptr;
    RETURN_IF_ERROR(data_loader_->GetInputData(
        input.second, 0, 0, &data_ptr, &batch1_bytesize));

    // Add optional input to request if data was found
    if (data_ptr != nullptr) {
      ctx->valid_inputs_.push_back(infer_input);
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
        &requested_output, backend_->Kind(), output.first));
    ctx->outputs_.push_back(requested_output);
  }
  RETURN_IF_ERROR(
      UpdateValidationOutputs(ctx->outputs_, 0, 0, ctx->expected_outputs_));

  return cb::Error::Success;
}

cb::Error
LoadManager::PrepareSharedMemoryInfer(InferContext* ctx)
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
    RETURN_IF_ERROR(cb::InferInput::Create(
        &infer_input, backend_->Kind(), input.first, shape,
        input.second.datatype_));
    ctx->inputs_.push_back(infer_input);

    // FIXME: TMA-765 - Shared memory mode does not support optional inputs,
    // currently, and will be implemented in the associated story.
    ctx->valid_inputs_.push_back(infer_input);

    RETURN_IF_ERROR(infer_input->SetSharedMemory(
        region_name, shared_memory_regions_[region_name].second));
  }

  for (const auto& output : *(parser_->Outputs())) {
    std::string region_name(TensorToRegionName(output.first));

    cb::InferRequestedOutput* requested_output;
    RETURN_IF_ERROR(cb::InferRequestedOutput::Create(
        &requested_output, backend_->Kind(), output.first));
    ctx->outputs_.push_back(requested_output);

    RETURN_IF_ERROR(requested_output->SetSharedMemory(
        region_name, shared_memory_regions_[region_name].second));
  }

  return cb::Error::Success;
}

cb::Error
LoadManager::UpdateInputs(
    const std::vector<cb::InferInput*>& inputs,
    std::vector<cb::InferInput*>& valid_inputs, int stream_index,
    int step_index)
{
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

  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    RETURN_IF_ERROR(SetInputs(inputs, valid_inputs, stream_index, step_index));
  } else {
    RETURN_IF_ERROR(SetInputsSharedMemory(inputs, stream_index, step_index));
  }

  return cb::Error::Success;
}

cb::Error
LoadManager::UpdateValidationOutputs(
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
LoadManager::ValidateOutputs(
    const InferContext& ctx, const cb::InferResult* result_ptr)
{
  // Validate output if set
  if (!ctx.expected_outputs_.empty()) {
    for (size_t i = 0; i < ctx.outputs_.size(); ++i) {
      const uint8_t* buf = nullptr;
      size_t byte_size = 0;
      result_ptr->RawData(ctx.outputs_[i]->Name(), &buf, &byte_size);
      for (const auto& expected : ctx.expected_outputs_[i]) {
        if (byte_size < expected.second) {
          return cb::Error(
              "Output size doesn't match expected size", pa::GENERIC_ERROR);
        } else if (memcmp(buf, expected.first, expected.second) != 0) {
          return cb::Error(
              "Output doesn't match expected output", pa::GENERIC_ERROR);
        } else {
          buf += expected.second;
          byte_size -= expected.second;
        }
      }
      if (byte_size != 0) {
        return cb::Error(
            "Output size doesn't match expected size", pa::GENERIC_ERROR);
      }
    }
  }
  return cb::Error::Success;
}

cb::Error
LoadManager::SetInputs(
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
    // some inferences in the batch provided data for an optional input and some
    // inferences did not, this is an invalid case and an error is thrown.
    if (missing_data_cnt == 0) {
      valid_inputs.push_back(input);
    } else if (missing_data_cnt > 0 && missing_data_cnt < batch_size_) {
      return cb::Error(
          "For batch sizes larger than 1, the same set of inputs must be "
          "specified for each batch. You cannot use different set of optional "
          "inputs for each individual batch.");
    }
  }
  return cb::Error::Success;
}

cb::Error
LoadManager::SetInputsSharedMemory(
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
        region_name, shared_memory_regions_[region_name].second));
  }
  return cb::Error::Success;
}

void
LoadManager::SetInferSequenceOptions(
    const uint32_t seq_id, std::unique_ptr<cb::InferOptions>& options)
{
  options->sequence_start_ = (sequence_stat_[seq_id]->remaining_queries_ == 0);

  // New sequence must be intialized before setting the id.
  if (options->sequence_start_) {
    InitNewSequence(seq_id);
  }
  options->sequence_id_ = sequence_stat_[seq_id]->seq_id_;
  options->sequence_end_ = (sequence_stat_[seq_id]->remaining_queries_ == 1);
}

void
LoadManager::InitNewSequence(int sequence_id)
{
  sequence_stat_[sequence_id]->seq_id_ =
      next_seq_id_++ % sequence_id_range_ + start_sequence_id_;
  if (!using_json_data_) {
    size_t new_length = GetRandomLength(0.2);
    sequence_stat_[sequence_id]->remaining_queries_ =
        new_length == 0 ? 1 : new_length;
  } else {
    // Selecting next available data stream based on uniform distribution.
    sequence_stat_[sequence_id]->data_stream_id_ =
        distribution_(rng_generator_);
    sequence_stat_[sequence_id]->remaining_queries_ =
        data_loader_->GetTotalSteps(
            sequence_stat_[sequence_id]->data_stream_id_);
  }
}

size_t
LoadManager::GetRandomLength(double offset_ratio)
{
  int random_offset = ((2.0 * rand() / double(RAND_MAX)) - 1.0) * offset_ratio *
                      sequence_length_;
  if (int(sequence_length_) + random_offset <= 0) {
    return 1;
  }
  return sequence_length_ + random_offset;
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
}

}}  // namespace triton::perfanalyzer
