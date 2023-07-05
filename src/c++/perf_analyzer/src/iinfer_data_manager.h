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
#include "infer_data.h"
#include "model_parser.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

/// Interface for classes that manage infer data preparation for inference
///
class IInferDataManager {
 public:
  /// Initialize this object. Must be called before any other functions
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error Init() = 0;

  /// Populate the target InferData object with input and output objects
  /// according to the model's shape
  /// \param infer_data The target InferData object.
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error InitInferData(InferData& infer_data) = 0;

  /// Updates the input and expected output data in the target infer_data for an
  /// inference request
  /// \param thread_id The ID of the calling thread
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \param infer_data The target InferData object
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error UpdateInferData(
      size_t thread_id, int stream_index, int step_index,
      InferData& infer_data) = 0;
};

}}  // namespace triton::perfanalyzer
