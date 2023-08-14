// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS"" AND ANY
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
#include "profile_data_exporter.h"

namespace triton { namespace perfanalyzer {

class NaggyMockProfileDataExporter : public ProfileDataExporter {
 public:
  NaggyMockProfileDataExporter()
  {
    ON_CALL(*this, ConvertToJson(testing::_, testing::_))
        .WillByDefault([this](
                           const std::vector<Experiment>& raw_experiments,
                           std::string& raw_version) {
          return this->ProfileDataExporter::ConvertToJson(
              raw_experiments, raw_version);
        });

    ON_CALL(*this, OutputToFile(testing::_))
        .WillByDefault([this](std::string& file_path) {
          this->ProfileDataExporter::OutputToFile(file_path);
        });

    ON_CALL(*this, AddExperiment(testing::_, testing::_, testing::_))
        .WillByDefault([this](
                           rapidjson::Value& entry,
                           rapidjson::Value& experiment,
                           const Experiment& raw_experiment) {
          this->ProfileDataExporter::AddExperiment(
              entry, experiment, raw_experiment);
        });
  }

  MOCK_METHOD(
      void, ConvertToJson,
      (const std::vector<Experiment>& raw_experiments,
       std::string& raw_version));
  MOCK_METHOD(
      void, AddExperiment,
      (rapidjson::Value & entry, rapidjson::Value& experiment,
       const Experiment& raw_experiment));
  MOCK_METHOD(void, OutputToFile, (std::string & file_path));

  rapidjson::Document& document_{ProfileDataExporter::document_};
};

using MockProfileDataExporter = testing::NiceMock<NaggyMockProfileDataExporter>;

}}  // namespace triton::perfanalyzer
