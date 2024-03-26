// Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "doctest.h"
#include "mock_profile_data_exporter.h"
#include "profile_data_exporter.h"

namespace triton { namespace perfanalyzer {

TEST_CASE("profile_data_exporter: ConvertToJson")
{
  using std::chrono::nanoseconds;
  using std::chrono::system_clock;
  using std::chrono::time_point;

  MockProfileDataExporter exporter{};

  InferenceLoadMode infer_mode{4, 0.0};
  uint64_t sequence_id{1};

  auto clock_epoch{time_point<system_clock>()};
  auto request_timestamp{clock_epoch + nanoseconds(1)};
  auto response_timestamp1{clock_epoch + nanoseconds(2)};
  auto response_timestamp2{clock_epoch + nanoseconds(3)};

  // Request inputs
  std::vector<std::string> in_bufs{"123", "456", "true"};
  RequestRecord::RequestInput request_input{
      {"in_key1",
       {reinterpret_cast<const uint8_t*>(in_bufs[0].data()), in_bufs[0].size(),
        "BYTES"}},
      {"in_key2",
       {reinterpret_cast<const uint8_t*>(in_bufs[1].data()), in_bufs[1].size(),
        "INT32"}},
      {"in_key3",
       {reinterpret_cast<const uint8_t*>(in_bufs[2].data()), in_bufs[2].size(),
        "BOOL"}},
  };


  // Response outputs
  std::vector<std::string> out_bufs{"abc", "def", "ghi", "jkl"};
  RequestRecord::ResponseOutput response_output1{
      {"out_key1",
       {reinterpret_cast<const uint8_t*>(out_bufs[0].data()),
        out_bufs[0].size()}},
      {"out_key2",
       {reinterpret_cast<const uint8_t*>(out_bufs[1].data()),
        out_bufs[1].size()}}};
  RequestRecord::ResponseOutput response_output2{
      {"out_key3",
       {reinterpret_cast<const uint8_t*>(out_bufs[2].data()),
        out_bufs[2].size()}},
      {"out_key4",
       {reinterpret_cast<const uint8_t*>(out_bufs[3].data()),
        out_bufs[3].size()}}};

  RequestRecord request_record{
      request_timestamp,
      std::vector<time_point<system_clock>>{
          response_timestamp1, response_timestamp2},
      {request_input},
      {response_output1, response_output2},
      0,
      false,
      sequence_id,
      false};
  std::vector<RequestRecord> requests{request_record};
  std::vector<uint64_t> window_boundaries{1, 5, 6};

  Experiment experiment;
  experiment.mode = infer_mode;
  experiment.requests = requests;
  experiment.window_boundaries = window_boundaries;
  std::vector<Experiment> experiments{experiment};

  std::string version{"1.2.3"};

  exporter.ConvertToJson(experiments, version);

  std::string json{R"(
      {
        "experiments" : [
          {
            "experiment" : {
              "mode" : "concurrency",
              "value" : 4
            },
            "requests" : [
              {
                "timestamp" : 1,
                "sequence_id" : 1,
                "request_inputs" : {"in_key1":"123","in_key2":456,"in_key3":true},
                "response_timestamps" : [ 2, 3 ],
                "response_outputs" : [ {"out_key1":"abc","out_key2":"def"}, {"out_key3":"ghi","out_key4":"jkl"} ]
              }
            ],
            "window_boundaries" : [ 1, 5, 6 ]
          }
        ],
        "version" : "1.2.3"
      }
      )"};

  rapidjson::Document expected_document;
  expected_document.Parse(json.c_str());

  // FIXME (TMA-1339): Look into the testing the order of things in the json
  const rapidjson::Value& expected_experiment{
      expected_document["experiments"][0]["experiment"]};
  const rapidjson::Value& expected_request{
      expected_document["experiments"][0]["requests"][0]};
  const rapidjson::Value& expected_windows{
      expected_document["experiments"][0]["window_boundaries"]};
  const rapidjson::Value& expected_version{expected_document["version"]};

  const rapidjson::Value& actual_experiment{
      exporter.document_["experiments"][0]["experiment"]};
  const rapidjson::Value& actual_request{
      exporter.document_["experiments"][0]["requests"][0]};
  const rapidjson::Value& actual_windows{
      exporter.document_["experiments"][0]["window_boundaries"]};
  const rapidjson::Value& actual_version{exporter.document_["version"]};

  CHECK(actual_experiment["mode"] == expected_experiment["mode"]);
  CHECK(actual_experiment["value"] == expected_experiment["value"]);

  CHECK(actual_request["timestamp"] == expected_request["timestamp"]);
  CHECK(actual_request["sequence_id"] == expected_request["sequence_id"]);
  CHECK(
      actual_request["response_timestamps"][0] ==
      expected_request["response_timestamps"][0]);
  CHECK(
      actual_request["response_timestamps"][1] ==
      expected_request["response_timestamps"][1]);
  CHECK(
      actual_request["response_outputs"][0] ==
      expected_request["response_outputs"][0]);
  CHECK(
      actual_request["response_outputs"][1] ==
      expected_request["response_outputs"][1]);

  CHECK(actual_windows[0] == expected_windows[0]);
  CHECK(actual_windows[1] == expected_windows[1]);
  CHECK(actual_windows[2] == expected_windows[2]);

  CHECK(actual_version == expected_version);
}

TEST_CASE("profile_data_exporter: AddExperiment")
{
  MockProfileDataExporter exporter{};

  Experiment raw_experiment;
  rapidjson::Value entry(rapidjson::kObjectType);
  rapidjson::Value experiment(rapidjson::kObjectType);

  SUBCASE("Concurrency mode")
  {
    InferenceLoadMode infer_mode{15, 0.0};
    raw_experiment.mode = infer_mode;

    exporter.AddExperiment(entry, experiment, raw_experiment);
    CHECK(entry.HasMember("experiment"));
    CHECK(entry["experiment"]["mode"] == "concurrency");
    CHECK(entry["experiment"]["value"] == 15);
  }

  SUBCASE("Request rate mode")
  {
    InferenceLoadMode infer_mode{0, 23.5};
    raw_experiment.mode = infer_mode;

    exporter.AddExperiment(entry, experiment, raw_experiment);
    CHECK(entry.HasMember("experiment"));
    CHECK(entry["experiment"]["mode"] == "request_rate");
    CHECK(entry["experiment"]["value"] == 23.5);
  }
}

TEST_CASE("profile_data_exporter: OutputToFile")
{
  MockProfileDataExporter exporter{};
  std::string file_path;

  SUBCASE("Empty file path")
  {
    file_path = "";
    CHECK_THROWS_WITH_AS(
        exporter.OutputToFile(file_path),
        "failed to open file for outputting raw profile data",
        PerfAnalyzerException);
  }

  SUBCASE("With file path")
  {
    file_path = "/tmp/test-" + GetRandomString(4) + ".json";
    CHECK_NOTHROW(exporter.OutputToFile(file_path));
    CHECK(IsFile(file_path));

    std::remove(file_path.c_str());
    CHECK(!IsFile(file_path));
  }
}

}}  // namespace triton::perfanalyzer
