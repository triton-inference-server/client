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

#include "client_backend/client_backend.h"
#include "constants.h"
#include "data_loader.h"
#include "iinfer_data_manager.h"
#include "infer_data.h"
#include "model_parser.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

/// Manages infer data to prepare an inference request and the resulting
/// inference output from triton server
class InferDataManagerBase : public IInferDataManager {
 public:
  InferDataManagerBase(
      const int32_t batch_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::shared_ptr<DataLoader>& data_loader)
      : batch_size_(batch_size), parser_(parser), factory_(factory),
        data_loader_(data_loader), backend_kind_(factory->Kind())
  {
  }

  /// Updates the input data to use for inference request
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \param infer_data The target InferData object
  /// \return cb::Error object indicating success or failure.
  cb::Error UpdateInputs(
      int stream_index, int step_index, InferData& infer_data) override;

  /// Updates the expected output data to use for inference request. Empty
  /// vector will be returned if there is no expected output associated to the
  /// step.
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \param infer_data The target InferData object
  /// \return cb::Error object indicating success or failure.
  cb::Error UpdateValidationOutputs(
      int stream_index, int step_index, InferData& infer_data) override;

 protected:
  size_t batch_size_;
  std::shared_ptr<ModelParser> parser_;
  std::shared_ptr<cb::ClientBackendFactory> factory_;
  std::shared_ptr<DataLoader> data_loader_;
  std::unique_ptr<cb::ClientBackend> backend_;
  cb::BackendKind backend_kind_;

  // FIXME
  virtual cb::Error SetInputs(
      const int stream_index, const int step_index, InferData& infer_data) = 0;

  /// Creates inference input object
  /// \param infer_input Output parameter storing newly created inference input
  /// \param kind Backend kind
  /// \param name Name of inference input
  /// \param dims Shape of inference input
  /// \param datatype Data type of inference input
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error CreateInferInput(
      cb::InferInput** infer_input, const cb::BackendKind kind,
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype);
};

}}  // namespace triton::perfanalyzer
