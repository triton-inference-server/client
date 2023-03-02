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

#include "infer_data_manager.h"
#include "infer_data_manager_shm.h"
#include "mock_client_backend.h"

namespace triton { namespace perfanalyzer {


class MockInferDataManagerShm : public InferDataManagerShm {
 public:
  MockInferDataManagerShm(
      const int32_t batch_size, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
      : InferDataManagerShm(
            batch_size, shared_memory_type, output_shm_size, parser, factory,
            data_loader)
  {
  }

  // Mocked version of the CopySharedMemory method in loadmanager.
  // This is strictly for testing to mock out the memcpy calls
  //
  cb::Error CopySharedMemory(
      uint8_t* input_shm_ptr, std::vector<const uint8_t*>& data_ptrs,
      std::vector<size_t>& byte_size, bool is_shape_tensor,
      std::string& region_name) override
  {
    return cb::Error::Success;
  }

  cb::Error CreateInferInput(
      cb::InferInput** infer_input, const cb::BackendKind kind,
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype) override
  {
    *infer_input = new cb::MockInferInput(kind, name, dims, datatype);
    return cb::Error::Success;
  }
};


class MockInferDataManager : public InferDataManager {
 public:
  MockInferDataManager(
      const int32_t batch_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
      : InferDataManager(batch_size, parser, factory, data_loader)
  {
  }

  MockInferDataManager(bool use_mock_update_infer_data = false)
      : use_mock_update_infer_data_(use_mock_update_infer_data)
  {
  }

  cb::Error CreateInferInput(
      cb::InferInput** infer_input, const cb::BackendKind kind,
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype) override
  {
    *infer_input = new cb::MockInferInput(kind, name, dims, datatype);
    return cb::Error::Success;
  }

  cb::Error UpdateInferData(
      int stream_index, int step_index, InferData& infer_data) override
  {
    if (use_mock_update_infer_data_) {
      update_infer_data_step_index_values_.push_back(step_index);
      return cb::Error::Success;
    } else {
      return InferDataManager::UpdateInferData(
          stream_index, step_index, infer_data);
    }
  }

  std::vector<int> update_infer_data_step_index_values_{};
  const bool use_mock_update_infer_data_{false};
};

class MockInferDataManagerFactory {
 public:
  static std::shared_ptr<IInferDataManager> CreateMockInferDataManager(
      const int32_t batch_size, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
  {
    if (shared_memory_type == SharedMemoryType::NO_SHARED_MEMORY) {
      return std::make_shared<MockInferDataManager>(
          batch_size, parser, factory, data_loader);
    } else {
      return std::make_shared<MockInferDataManagerShm>(
          batch_size, shared_memory_type, output_shm_size, parser, factory,
          data_loader);
    }
  }
};

}}  // namespace triton::perfanalyzer
