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

#include "data_loader.h"
#include "iinfer_data_manager.h"
#include "infer_data_manager.h"
#include "infer_data_manager_shm.h"
#include "model_parser.h"

namespace triton { namespace perfanalyzer {

class InferDataManagerFactory {
 public:
  static std::shared_ptr<IInferDataManager> CreateInferDataManager(
      const int32_t batch_size, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
  {
    if (shared_memory_type == SharedMemoryType::NO_SHARED_MEMORY) {
      return CreateInferDataManagerNoShm(
          batch_size, parser, factory, data_loader);
    } else {
      return CreateInferDataManagerShm(
          batch_size, shared_memory_type, output_shm_size, parser, factory,
          data_loader);
    }
  }

 private:
  static std::shared_ptr<IInferDataManager> CreateInferDataManagerNoShm(
      const int32_t batch_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
  {
    return std::make_shared<InferDataManager>(
        batch_size, parser, factory, data_loader);
  }

  static std::shared_ptr<IInferDataManager> CreateInferDataManagerShm(
      const int32_t batch_size, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
  {
    return std::make_shared<InferDataManagerShm>(
        batch_size, shared_memory_type, output_shm_size, parser, factory,
        data_loader);
  }
};

}}  // namespace triton::perfanalyzer