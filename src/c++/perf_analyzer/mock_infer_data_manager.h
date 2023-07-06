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

#include "gmock/gmock.h"
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
  // Tracks the mapping of shared memory label to data
  //
  cb::Error CopySharedMemory(
      uint8_t* input_shm_ptr, const std::vector<DataLoaderData>& input_datas,
      bool is_shape_tensor, std::string& region_name) override
  {
    std::vector<int32_t> vals;

    for (size_t i = 0; i < input_datas.size(); i++) {
      int32_t val = *reinterpret_cast<const int32_t*>(input_datas[i].data_ptr);
      vals.push_back(val);
    }
    mocked_shared_memory_regions.insert(std::make_pair(region_name, vals));
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

  // Tracks the mapping of shared memory label to data
  std::map<std::string, std::vector<int32_t>> mocked_shared_memory_regions;
};


class MockInferDataManager : public InferDataManager {
 public:
  MockInferDataManager() { SetupMocks(); }

  MockInferDataManager(
      const size_t max_threads, const int32_t batch_size,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
      : InferDataManager(max_threads, batch_size, parser, factory, data_loader)
  {
    SetupMocks();
  }

  void SetupMocks()
  {
    ON_CALL(
        *this, UpdateInferData(testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(
            [this](
                size_t thread_id, int stream_index, int step_index,
                InferData& infer_data) -> cb::Error {
              return this->InferDataManager::UpdateInferData(
                  thread_id, stream_index, step_index, infer_data);
            });
  }

  MOCK_METHOD(
      cb::Error, UpdateInferData, (size_t, int, int, InferData&), (override));

  cb::Error CreateInferInput(
      cb::InferInput** infer_input, const cb::BackendKind kind,
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype) override
  {
    *infer_input = new cb::MockInferInput(kind, name, dims, datatype);
    return cb::Error::Success;
  }
};

class MockInferDataManagerFactory {
 public:
  static std::shared_ptr<IInferDataManager> CreateMockInferDataManager(
      const size_t max_threads, const int32_t batch_size,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
  {
    if (shared_memory_type == SharedMemoryType::NO_SHARED_MEMORY) {
      return std::make_shared<testing::NiceMock<MockInferDataManager>>(
          max_threads, batch_size, parser, factory, data_loader);
    } else {
      return std::make_shared<testing::NiceMock<MockInferDataManagerShm>>(
          batch_size, shared_memory_type, output_shm_size, parser, factory,
          data_loader);
    }
  }
};

}}  // namespace triton::perfanalyzer
