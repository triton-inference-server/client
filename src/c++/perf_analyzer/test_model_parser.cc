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

#include <rapidjson/document.h>
#include <cstdint>
#include "client_backend/client_backend.h"
#include "constants.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_model_parser.h"

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

TEST_CASE("ModelParser: testing the GetInt function")
{
  int64_t integer_value{0};
  MockModelParser mmp;

  SUBCASE("valid string")
  {
    rapidjson::Value value("100");
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 100);
  }

  SUBCASE("invalid string, alphabet")
  {
    rapidjson::Value value("abc");
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(result.Message() == "unable to convert 'abc' to integer");
    CHECK(integer_value == 0);
  }

  SUBCASE("invalid string, number out of range")
  {
    rapidjson::Value value("9223372036854775808");
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(
        result.Message() ==
        "unable to convert '9223372036854775808' to integer");
    CHECK(integer_value == 0);
  }

  SUBCASE("valid int, lowest Int64")
  {
    rapidjson::Value value(2147483648);
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 2147483648);
  }

  SUBCASE("valid int, highest Int32")
  {
    rapidjson::Value value(2147483647);
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 2147483647);
  }

  SUBCASE("invalid floating point")
  {
    rapidjson::Value value(100.1);
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(result.Message() == "failed to parse the integer value");
    CHECK(integer_value == 0);
  }
}

TEST_CASE(
    "ModelParser: determining scheduler type" *
    doctest::description(
        "This test confirms the behavior and all side-effects "
        "of DetermineSchedulerType(). This includes setting the "
        "value of scheduler_type_ and composing_models_map_"))
{
  const char no_batching[] = R"({})";
  const char seq_batching[] = R"({ "sequence_batching":{} })";
  const char dyn_batching[] = R"({ "dynamic_batching":{} })";
  const char ensemble[] = R"({
    "name": "EnsembleModel",
    "platform": "ensemble",
    "ensemble_scheduling": {
      "step": [{
          "model_name": "ModelA",
          "model_version": -1
        },
        {
          "model_name": "ModelB",
          "model_version": -1
        }
      ]
    }
  })";

  const char nested_ensemble[] = R"({
    "name": "ModelA",
    "platform": "ensemble",
    "ensemble_scheduling": {
      "step": [{
          "model_name": "ModelC",
          "model_version": -1
        },
        {
          "model_name": "ModelD",
          "model_version": -1
        }
      ]
    }
  })";


  std::string model_version = "";

  std::shared_ptr<cb::MockClientStats> stats =
      std::make_shared<cb::MockClientStats>();
  std::unique_ptr<cb::MockClientBackend> mock_backend =
      std::make_unique<cb::MockClientBackend>(stats);


  rapidjson::Document config;
  ModelParser::ModelSchedulerType expected_type;
  ComposingModelMap expected_composing_model_map;

  auto SetJsonPtrNoSeq = [](rapidjson::Document* model_config) {
    model_config->Parse(R"({ "platform":"none" })");
    return cb::Error::Success;
  };

  auto SetJsonPtrYesSeq = [](rapidjson::Document* model_config) {
    model_config->Parse(R"({ "sequence_batching":{}, "platform":"none" })");
    return cb::Error::Success;
  };

  auto SetJsonPtrNestedEnsemble =
      [&nested_ensemble](rapidjson::Document* model_config) {
        model_config->Parse(nested_ensemble);
        return cb::Error::Success;
      };

  SUBCASE("No batching")
  {
    config.Parse(no_batching);
    expected_type = ModelParser::ModelSchedulerType::NONE;
  }
  SUBCASE("Sequence batching")
  {
    config.Parse(seq_batching);
    expected_type = ModelParser::ModelSchedulerType::SEQUENCE;
  }
  SUBCASE("Dynamic batching")
  {
    config.Parse(dyn_batching);
    expected_type = ModelParser::ModelSchedulerType::DYNAMIC;
  }
  SUBCASE("Ensemble")
  {
    config.Parse(ensemble);

    expected_composing_model_map["EnsembleModel"].emplace("ModelA", "");
    expected_composing_model_map["EnsembleModel"].emplace("ModelB", "");

    SUBCASE("no sequences")
    {
      EXPECT_CALL(
          *mock_backend, ModelConfig(testing::_, testing::_, testing::_))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq));

      expected_type = ModelParser::ModelSchedulerType::ENSEMBLE;
    }
    SUBCASE("yes sequences")
    {
      EXPECT_CALL(
          *mock_backend, ModelConfig(testing::_, testing::_, testing::_))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrYesSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrYesSeq));

      expected_type = ModelParser::ModelSchedulerType::ENSEMBLE_SEQUENCE;
    }
  }
  SUBCASE("Nested Ensemble")
  {
    config.Parse(ensemble);

    expected_composing_model_map["EnsembleModel"].emplace("ModelA", "");
    expected_composing_model_map["EnsembleModel"].emplace("ModelB", "");
    expected_composing_model_map["ModelA"].emplace("ModelC", "");
    expected_composing_model_map["ModelA"].emplace("ModelD", "");

    SUBCASE("no sequences")
    {
      EXPECT_CALL(
          *mock_backend, ModelConfig(testing::_, testing::_, testing::_))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNestedEnsemble))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNestedEnsemble))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq));

      expected_type = ModelParser::ModelSchedulerType::ENSEMBLE;
    }
    SUBCASE("yes sequences")
    {
      EXPECT_CALL(
          *mock_backend, ModelConfig(testing::_, testing::_, testing::_))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNestedEnsemble))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrYesSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNestedEnsemble))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrYesSeq))
          .WillOnce(testing::WithArg<0>(SetJsonPtrNoSeq));

      expected_type = ModelParser::ModelSchedulerType::ENSEMBLE_SEQUENCE;
    }
  }

  std::unique_ptr<cb::ClientBackend> backend = std::move(mock_backend);

  MockModelParser mmp;
  mmp.DetermineSchedulerType(config, model_version, backend);

  auto actual_type = mmp.SchedulerType();
  CHECK(actual_type == expected_type);

  auto actual_composing_model_map = *mmp.GetComposingModelMap().get();
  CHECK(actual_composing_model_map == expected_composing_model_map);

  // Destruct gmock objects to determine gmock-related test failure
  backend.reset();
}

}}  // namespace triton::perfanalyzer
