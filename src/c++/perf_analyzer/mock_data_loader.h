// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

namespace triton { namespace perfanalyzer {

class MockDataLoader : public DataLoader {
 public:
  MockDataLoader() = default;

  cb::Error ReadDataFromStr(
      const std::shared_ptr<ModelTensorMap>& inputs,
      const std::shared_ptr<ModelTensorMap>& outputs, const std::string& str)
  {
    rapidjson::Document d{};
    const unsigned int parseFlags = rapidjson::kParseNanAndInfFlag;
    d.Parse<parseFlags>(str.c_str());
    if (d.HasParseError()) {
      std::cerr << "cb::Error  : " << d.GetParseError() << '\n'
                << "Offset : " << d.GetErrorOffset() << '\n';
      return cb::Error(
          "failed to parse the specified json file for reading provided data",
          pa::GENERIC_ERROR);
    }

    if (!d.HasMember("data")) {
      return cb::Error(
          "The json file doesn't contain data field", pa::GENERIC_ERROR);
    }

    const rapidjson::Value& streams = d["data"];

    // Validation data is optional, once provided, it must align with 'data'
    const rapidjson::Value* out_streams = nullptr;
    if (d.HasMember("validation_data")) {
      out_streams = &d["validation_data"];
      if (out_streams->Size() != streams.Size()) {
        return cb::Error(
            "The 'validation_data' field doesn't align with 'data' field in "
            "the "
            "json file",
            pa::GENERIC_ERROR);
      }
    }

    int count = streams.Size();

    data_stream_cnt_ += count;
    int offset = step_num_.size();
    for (size_t i = offset; i < data_stream_cnt_; i++) {
      const rapidjson::Value& steps = streams[i - offset];
      const rapidjson::Value* output_steps =
          (out_streams == nullptr) ? nullptr : &(*out_streams)[i - offset];
      if (steps.IsArray()) {
        step_num_.push_back(steps.Size());
        for (size_t k = 0; k < step_num_[i]; k++) {
          RETURN_IF_ERROR(ReadTensorData(steps[k], inputs, i, k, true));
        }

        if (output_steps != nullptr) {
          if (!output_steps->IsArray() ||
              (output_steps->Size() != steps.Size())) {
            return cb::Error(
                "The 'validation_data' field doesn't align with 'data' field "
                "in "
                "the json file",
                pa::GENERIC_ERROR);
          }
          for (size_t k = 0; k < step_num_[i]; k++) {
            RETURN_IF_ERROR(
                ReadTensorData((*output_steps)[k], outputs, i, k, false));
          }
        }
      } else {
        // There is no nesting of tensors, hence, will interpret streams as
        // steps and add the tensors to a single stream '0'.
        int offset = 0;
        if (step_num_.empty()) {
          step_num_.push_back(count);
        } else {
          offset = step_num_[0];
          step_num_[0] += (count);
        }
        data_stream_cnt_ = 1;
        for (size_t k = offset; k < step_num_[0]; k++) {
          RETURN_IF_ERROR(
              ReadTensorData(streams[k - offset], inputs, 0, k, true));
        }

        if (out_streams != nullptr) {
          for (size_t k = offset; k < step_num_[0]; k++) {
            RETURN_IF_ERROR(ReadTensorData(
                (*out_streams)[k - offset], outputs, 0, k, false));
          }
        }
        break;
      }
    }

    max_non_sequence_step_id_ = std::max(1, (int)(step_num_[0] / batch_size_));
    return cb::Error::Success;
  };
};

}}  // namespace triton::perfanalyzer
