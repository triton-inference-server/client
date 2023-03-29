// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "model_parser.h"

namespace triton { namespace perfanalyzer {

class MockModelParser : public ModelParser {
 public:
  MockModelParser() : ModelParser(clientbackend::BackendKind::TRITON) {}

  MockModelParser(
      bool is_sequence_model, bool is_decoupled_model,
      size_t max_batch_size = 64)
      : ModelParser(clientbackend::BackendKind::TRITON)
  {
    if (is_sequence_model) {
      scheduler_type_ = ModelParser::SEQUENCE;
    }
    is_decoupled_ = is_decoupled_model;
    max_batch_size_ = max_batch_size;
  }

  // Expose private function
  cb::Error GetInt(const rapidjson::Value& value, int64_t* integer_value)
  {
    return ModelParser::GetInt(value, integer_value);
  }

  // Expose private function
  cb::Error DetermineComposingModelMap(
      const std::vector<cb::ModelIdentifier>& bls_composing_models,
      const rapidjson::Document& config,
      std::unique_ptr<cb::ClientBackend>& backend)
  {
    return ModelParser::DetermineComposingModelMap(
        bls_composing_models, config, backend);
  }

  // Expose private function
  cb::Error DetermineSchedulerType(
      const rapidjson::Document& config,
      std::unique_ptr<cb::ClientBackend>& backend)
  {
    return ModelParser::DetermineSchedulerType(config, backend);
  }

  std::shared_ptr<ComposingModelMap>& composing_models_map_{
      ModelParser::composing_models_map_};
  std::shared_ptr<ModelTensorMap>& inputs_{ModelParser::inputs_};
};

}}  // namespace triton::perfanalyzer
