// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gmock/gmock.h"

namespace triton { namespace perfanalyzer {

/// Mock DataLoader class used for testing to allow JSON data to be read
/// from string, rather than file.
///
class NaggyMockDataLoader : public DataLoader {
 public:
  NaggyMockDataLoader() { SetupMocks(); }
  NaggyMockDataLoader(size_t batch_size) : DataLoader(batch_size)
  {
    SetupMocks();
  }

  void SetupMocks()
  {
    ON_CALL(*this, GetTotalSteps(testing::_))
        .WillByDefault([this](size_t stream_id) -> size_t {
          return this->DataLoader::GetTotalSteps(stream_id);
        });
  }

  MOCK_METHOD(size_t, GetTotalSteps, (size_t), (override));


  cb::Error ReadDataFromJSON(
      const std::shared_ptr<ModelTensorMap>& inputs,
      const std::shared_ptr<ModelTensorMap>& outputs,
      const std::string& json_file) override
  {
    return ReadDataFromStr(json_file, inputs, outputs);
  }

  cb::Error ReadDataFromStr(
      const std::string& str, const std::shared_ptr<ModelTensorMap>& inputs,
      const std::shared_ptr<ModelTensorMap>& outputs)
  {
    rapidjson::Document d{};
    const unsigned int parseFlags = rapidjson::kParseNanAndInfFlag;
    d.Parse<parseFlags>(str.c_str());

    return ParseData(d, inputs, outputs);
  };

  std::vector<size_t>& step_num_{DataLoader::step_num_};
  size_t& data_stream_cnt_{DataLoader::data_stream_cnt_};
};

// Non-naggy version of Mock Data Loader (won't warn when using default gmock
// mocked function)
using MockDataLoader = testing::NiceMock<NaggyMockDataLoader>;

}}  // namespace triton::perfanalyzer
