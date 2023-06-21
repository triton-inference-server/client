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

#include "data_loader.h"
#include "doctest.h"
#include "mock_data_loader.h"

namespace triton { namespace perfanalyzer {

/// Helper class for testing the DataLoader
///
class TestDataLoader {
 public:
  // Static function to create a generic ModelTensor
  //
  static ModelTensor CreateTensor(std::string name)
  {
    ModelTensor t;
    t.name_ = name;
    t.datatype_ = "INT32";
    t.shape_ = {1};
    t.is_shape_tensor_ = false;
    t.is_optional_ = false;
    return t;
  }
};

TEST_CASE("dataloader: no data")
{
  MockDataLoader dataloader;
  CHECK(dataloader.GetDataStreamsCount() == 0);
  cb::Error status = dataloader.ValidateIndexes(0, 0);
  CHECK(status.IsOk() == false);
}

TEST_CASE("dataloader: ValidateIndexes")
{
  MockDataLoader dataloader;

  // Pretend we loaded 2 streams, one with 1 step, one with 3 steps
  dataloader.data_stream_cnt_ = 2;
  dataloader.step_num_.push_back(1);
  dataloader.step_num_.push_back(3);

  CHECK_EQ(dataloader.GetDataStreamsCount(), 2);

  // Step in range for stream 0
  cb::Error status = dataloader.ValidateIndexes(0, 0);
  CHECK(status.IsOk() == true);

  // Step out of range for stream 0
  status = dataloader.ValidateIndexes(0, 1);
  CHECK(status.IsOk() == false);

  // Step in range for stream 1
  status = dataloader.ValidateIndexes(1, 2);
  CHECK(status.IsOk() == true);

  // Step out of range for stream 1
  status = dataloader.ValidateIndexes(1, 3);
  CHECK(status.IsOk() == false);

  // Stream out of range
  status = dataloader.ValidateIndexes(2, 0);
  CHECK(status.IsOk() == false);
}

TEST_CASE("dataloader: GetTotalSteps")
{
  MockDataLoader dataloader;

  // Pretend we loaded 2 streams, one with 1 step, one with 3 steps
  dataloader.data_stream_cnt_ = 2;
  dataloader.step_num_.push_back(1);
  dataloader.step_num_.push_back(3);

  CHECK_EQ(dataloader.GetTotalSteps(0), 1);
  CHECK_EQ(dataloader.GetTotalSteps(1), 3);

  // It will return 0 if out of range
  CHECK_EQ(dataloader.GetTotalSteps(2), 0);
}

TEST_CASE("dataloader: ParseData: Bad Json")
{
  std::string json_str{"bad json text"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
  CHECK(status.IsOk() == false);
  CHECK_EQ(
      status.Message(),
      "failed to parse the specified json file for reading provided data");
}

TEST_CASE("dataloader: ParseData: Misc error cases")
{
  std::string expected_message;
  std::string json_str;

  SUBCASE("No data")
  {
    json_str = R"({ "notdata" : 5})";
    expected_message = "The json file doesn't contain data field";
  }
  SUBCASE("Not string b64")
  {
    json_str = R"({"data": [{ "INPUT1": {"b64": 5} }]})";
    expected_message =
        "the value of b64 field should be of type string ( Location stream id: "
        "0, step id: 0)";
  }
  SUBCASE("Not b64 or array")
  {
    json_str = R"({"data": [{ "INPUT1": {"not_b64": "AAAAAQ=="} }]})";
    expected_message =
        "missing content field. ( Location stream id: 0, step id: 0)";
  }
  SUBCASE("Malformed input (boolean type)")
  {
    json_str = R"({"data": [{ "INPUT1": null }]})";
    expected_message = "Input data file is malformed.";
  }

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();
  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
  CHECK(status.IsOk() == false);
  CHECK_EQ(status.Message(), expected_message);
}

// FIXME TMA-1210
// TEST_CASE(
//    "dataloader: ParseData: Mismatch Shape" *
//    doctest::description(
//        "When the size of the provided Input is not in line with the Tensor's
//        " "shape, then an error should be thrown"))
//{
//  std::string json_str{R"({"data": [{ "INPUT1": [1,2] }]})"};
//
//
//  MockDataLoader dataloader;
//  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
//  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
//  input1.shape_ = {3};
//  inputs->insert(std::make_pair(input1.name_, input1));
//
//  std::shared_ptr<ModelTensorMap> outputs =
//  std::make_shared<ModelTensorMap>();
//
//  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
//  CHECK(status.IsOk() == false);
//  CHECK_EQ(status.Message(), "TKG");
//}


TEST_CASE(
    "dataloader: ParseData: Mismatch Input and Output" *
    doctest::description(
        "When the size of the provided Input and validation Output data are "
        "different, then an error should be thrown"))
{
  std::string json_str{R"({
   "data": [
     { "INPUT1": [1] },
     { "INPUT1": [2] },
     { "INPUT1": [3] }
   ],
   "validation_data": [
     { "OUTPUT1": [7] }
   ]})"};


  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
  CHECK(status.IsOk() == false);
  CHECK_EQ(
      status.Message(),
      "The 'validation_data' field doesn't align with 'data' field in the json "
      "file");
}

TEST_CASE("dataloader: ParseData: Valid Data")
{
  std::string json_str;

  SUBCASE("Normal json")
  {
    json_str = R"({
   "data": [
     { "INPUT1": [1] },
     { "INPUT1": [2] },
     { "INPUT1": [3] }
   ],
   "validation_data": [
    { "OUTPUT1": [4] },
    { "OUTPUT1": [5] },
    { "OUTPUT1": [6] }
   ]})";
  }
  SUBCASE("b64 json")
  {
    // Note that these encoded values decode to the numbers 1,2,3,4,5,6, which
    // is the same data as the normal json case above
    json_str = R"({
   "data": [
     { "INPUT1": {"b64": "AAAAAQ=="} },
     { "INPUT1": {"b64": "AgAAAA=="} },
     { "INPUT1": {"b64": "AwAAAA=="} }
   ],
   "validation_data": [
    { "OUTPUT1": {"b64": "BAAAAA=="} },
    { "OUTPUT1": {"b64": "BQAAAA=="} },
    { "OUTPUT1": {"b64": "BgAAAA=="} }
   ]})";
  }

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  ModelTensor output1 = TestDataLoader::CreateTensor("OUTPUT1");

  inputs->insert(std::make_pair(input1.name_, input1));
  outputs->insert(std::make_pair(output1.name_, output1));

  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
  REQUIRE(status.IsOk());
  CHECK_EQ(dataloader.GetDataStreamsCount(), 1);
  CHECK_EQ(dataloader.GetTotalSteps(0), 3);

  // Confirm the correct data is in the dataloader
  //
  const uint8_t* data_ptr{nullptr};
  size_t batch1_size;
  std::vector<int64_t> shape;

  dataloader.GetInputShape(input1, 0, 1, &shape);
  CHECK_EQ(shape.size(), 1);
  CHECK_EQ(shape[0], 1);

  dataloader.GetInputData(input1, 0, 1, &data_ptr, &batch1_size);
  auto input_data = *reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(input_data, 2);
  CHECK_EQ(batch1_size, 4);

  dataloader.GetOutputData("OUTPUT1", 0, 2, &data_ptr, &batch1_size);
  auto output_data = *reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(output_data, 6);
  CHECK_EQ(batch1_size, 4);
}

TEST_CASE("dataloader: ParseData: Multiple Streams Invalid Cases")
{
  // Mismatch because one stream with wrong number of steps
  std::string mismatch_case1a{R"({
   "data": [ { "INPUT1": [1,2] } ],
   "validation_data": [ { "OUTPUT1": [4] }, { "OUTPUT1": [5] } ]
   })"};
  std::string mismatch_case1b{R"({
   "data": [ { "INPUT1": [1,2] }, { "INPUT1": [2,3] } ],
   "validation_data": [ { "OUTPUT1": [4] } ]
   })"};

  // Mismatch because wrong number of streams (3 output streams for 2 input
  // streams)
  std::string mismatch_case2{R"({
   "data": [
     [ { "INPUT1": [1,2] }, { "INPUT1": [2,3] } ],
     [ { "INPUT1": [10,11] } ]
   ],
   "validation_data": [
    [ { "OUTPUT1": [4] }, { "OUTPUT1": [5] } ],
    [ { "OUTPUT1": [40] } ],
    [ { "OUTPUT1": [60] } ]
   ]})"};

  // Mismatch because same number of streams but wrong number of steps
  std::string mismatch_case3a{R"({
   "data": [
     [ { "INPUT1": [1,2] } ],
     [ { "INPUT1": [10,11] } ]
   ],
   "validation_data": [
    [ { "OUTPUT1": [4] }, { "OUTPUT1": [5] } ],
    [ { "OUTPUT1": [40] } ]
   ]})"};
  std::string mismatch_case3b{R"({
   "data": [
     [ { "INPUT1": [1,2] } ],
     [ { "INPUT1": [10,11] } ]
   ],
   "validation_data": [
    [ { "OUTPUT1": [4] } ],
    [ { "OUTPUT1": [40] }, { "OUTPUT1": [50] } ]
   ]})"};

  auto test_lambda = [&](std::string json_data) {
    std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
    std::shared_ptr<ModelTensorMap> outputs =
        std::make_shared<ModelTensorMap>();

    ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
    ModelTensor output1 = TestDataLoader::CreateTensor("OUTPUT1");
    input1.shape_ = {2};
    inputs->insert(std::make_pair(input1.name_, input1));
    outputs->insert(std::make_pair(output1.name_, output1));

    MockDataLoader dataloader;
    cb::Error status = dataloader.ReadDataFromStr(json_data, inputs, outputs);
    CHECK(status.IsOk() == false);
    CHECK_EQ(
        status.Message(),
        "The 'validation_data' field doesn't align with 'data' field in the "
        "json file");
  };

  test_lambda(mismatch_case1a);
  test_lambda(mismatch_case1b);
  test_lambda(mismatch_case2);
  test_lambda(mismatch_case3a);
  test_lambda(mismatch_case3b);
}

TEST_CASE("dataloader: ParseData: Multiple Streams Valid")
{
  std::string json_str{R"({
   "data": [
     [ { "INPUT1": [1,2] }, { "INPUT1": [2,3] }],
     [ { "INPUT1": [10,11] } ]
   ],
   "validation_data": [
    [ { "OUTPUT1": [4] }, { "OUTPUT1": [5] } ],
    [ { "OUTPUT1": [40] } ]
   ]
   })"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  ModelTensor output1 = TestDataLoader::CreateTensor("OUTPUT1");
  input1.shape_ = {2};
  inputs->insert(std::make_pair(input1.name_, input1));
  outputs->insert(std::make_pair(output1.name_, output1));

  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
  REQUIRE(status.IsOk());
  CHECK_EQ(dataloader.GetDataStreamsCount(), 2);
  CHECK_EQ(dataloader.GetTotalSteps(0), 2);
  CHECK_EQ(dataloader.GetTotalSteps(1), 1);

  // Confirm the correct data is in the dataloader
  //
  const uint8_t* data_ptr{nullptr};
  size_t batch1_size;

  dataloader.GetInputData(input1, 0, 1, &data_ptr, &batch1_size);

  const int32_t* input_data = reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(input_data[0], 2);
  CHECK_EQ(input_data[1], 3);
  // 2 elements of int32 data is 8 bytes
  CHECK_EQ(batch1_size, 8);

  dataloader.GetOutputData("OUTPUT1", 1, 0, &data_ptr, &batch1_size);
  const int32_t* output_data = reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(output_data[0], 40);
  CHECK_EQ(batch1_size, 4);
}


TEST_CASE(
    "dataloader: ParseData: Missing Shape" *
    doctest::description(
        "When a tensor's shape is dynamic (-1), then it needs to be provided "
        "via --shape option (which is not visable to this testing), or via a "
        "shape option in the json. If not, an error is thrown"))
{
  std::string json_str{R"({"data": [{ "INPUT1": [1,2,3] } ]})"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  input1.shape_ = {-1};

  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
  CHECK_EQ(status.IsOk(), false);
  CHECK_EQ(
      status.Message(),
      "The variable-sized tensor \"INPUT1\" is missing shape, see --shape "
      "option.");
}

// FIXME TMA-1210
// TEST_CASE(
//    "dataloader: ParseData: Supplied Shape is wrong" *
//    doctest::description("Supply the dynamic shape for an input, but have it "
//                         "mismatch the size/shape of the supplied data"))
//{
//  std::string json_str{R"({"data": [{
//     "INPUT1": { "shape": [4], "content": [1,2,3] }
//    }]})"};
//
//  MockDataLoader dataloader;
//  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
//  std::shared_ptr<ModelTensorMap> outputs =
//  std::make_shared<ModelTensorMap>();
//
//  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
//  input1.shape_ = {-1};
//
//  inputs->insert(std::make_pair(input1.name_, input1));
//
//  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
//  CHECK_EQ(status.IsOk(), false);
//  CHECK_EQ(status.Message(), "FIXME");
//}
//
// FIXME TMA-1210 - what about the case of mismatch in the number of dimentions?
// Supplying [1,2,3] is ok for [-1,2,-1], but not for [-1,-1] or [2,-1,-1]
//


TEST_CASE(
    "dataloader: ParseData: Supplied Shape is valid" *
    doctest::description("Supply the dynamic shape for an input"))
{
  std::string json_str{R"({"data": [{
     "INPUT1": { "shape": [3,2], "content": [1,2,3,4,5,6] }
    }]})"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  input1.shape_ = {-1, -1};

  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
  REQUIRE(status.IsOk());

  std::vector<int64_t> shape;
  dataloader.GetInputShape(input1, 0, 0, &shape);
  CHECK_EQ(shape.size(), 2);
  CHECK_EQ(shape[0], 3);
  CHECK_EQ(shape[1], 2);
}


TEST_CASE(
    "dataloader: ParseData: Supplied Shape is zero" *
    doctest::description(
        "Zero is a legal shape value and should be handled correctly"))
{
  std::string json_str{R"({"data": [{
     "INPUT1": { "shape": [0,2], "content": [] }
    }]})"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  input1.shape_ = {-1, 2};

  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status = dataloader.ReadDataFromStr(json_str, inputs, outputs);
  REQUIRE(status.IsOk());

  std::vector<int64_t> shape;
  dataloader.GetInputShape(input1, 0, 0, &shape);
  CHECK_EQ(shape.size(), 2);
  CHECK_EQ(shape[0], 0);
  CHECK_EQ(shape[1], 2);

  const uint8_t* data_ptr{nullptr};
  size_t batch1_size;
  status = dataloader.GetInputData(input1, 0, 0, &data_ptr, &batch1_size);
  REQUIRE(status.IsOk());

  const int32_t* input_data = reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(data_ptr, nullptr);
  CHECK_EQ(batch1_size, 0);
}

// FIXME TMA 1211
// TEST_CASE(
//    "dataloader: ParseData: Multiple Calls" *
//    doctest::description(
//        "ParseData can be called multiple times. The data should "
//        "accumulate (as opposed to only the last call being valid)"))
//{
//  std::string json_str1{R"({
//  "data": [
//    { "INPUT1": [1, 2] },
//    { "INPUT1": [2, 3] }
//  ],
//  "validation_data": [
//   { "OUTPUT1": [4, 5, 6] },
//   { "OUTPUT1": [5, 6, 7] }
//  ]})"};
//
//  std::string json_str2{R"({
//  "data": [
//    [{ "INPUT1": [10, 20] }],
//    [{ "INPUT1": [100, 200] }]
//  ]})"};
//
//  std::string json_str3{R"({
//  "data": [
//    { "INPUT1": [1000, 2000] },
//    { "INPUT1": [2000, 3000] },
//    { "INPUT1": [3000, 4000] }
//  ]})"};
//
//
//  MockDataLoader dataloader;
//  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
//  std::shared_ptr<ModelTensorMap> outputs =
//  std::make_shared<ModelTensorMap>();
//
//  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
//  ModelTensor output1 = TestDataLoader::CreateTensor("OUTPUT1");
//
//  inputs->insert(std::make_pair(input1.name_, input1));
//  outputs->insert(std::make_pair(output1.name_, output1));
//
//  cb::Error status = dataloader.ReadDataFromStr(json_str1, inputs, outputs);
//  CHECK(status.IsOk());
//  CHECK_EQ(dataloader.GetDataStreamsCount(), 1);
//  CHECK_EQ(dataloader.GetTotalSteps(0), 2);
//
//  status = dataloader.ReadDataFromStr(json_str2, inputs, outputs);
//  CHECK(status.IsOk());
//  CHECK_EQ(dataloader.GetDataStreamsCount(), 3);
//  CHECK_EQ(dataloader.GetTotalSteps(0), 2);
//  CHECK_EQ(dataloader.GetTotalSteps(1), 1);
//  CHECK_EQ(dataloader.GetTotalSteps(2), 1);
//
//  status = dataloader.ReadDataFromStr(json_str3, inputs, outputs);
//  CHECK(status.IsOk());
//  CHECK_EQ(dataloader.GetDataStreamsCount(), 4);
//  CHECK_EQ(dataloader.GetTotalSteps(0), 2);
//  CHECK_EQ(dataloader.GetTotalSteps(1), 1);
//  CHECK_EQ(dataloader.GetTotalSteps(2), 1);
//  CHECK_EQ(dataloader.GetTotalSteps(3), 3);
//}

TEST_CASE(
    "dataloader: GenerateData: Is Shape Tensor" *
    doctest::description("It is illegal to generate data for any Tensor with "
                         "is_shape_tensor=True"))
{
  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  input1.is_shape_tensor_ = true;
  inputs->insert(std::make_pair(input1.name_, input1));

  bool zero_input = true;
  size_t string_length = 5;
  std::string string_data = "FOOBAR";
  cb::Error status =
      dataloader.GenerateData(inputs, zero_input, string_length, string_data);
  CHECK(status.IsOk() == false);
  CHECK_EQ(
      status.Message(),
      "can not generate data for shape tensor 'INPUT1', user-provided data is "
      "needed.");
}


TEST_CASE(
    "dataloader: GenerateData: Non-BYTES" *
    doctest::description(
        "Calling GenerateData for non-BYTES datatype should result in a single "
        "stream with one step. If the zero input flag is set, all of that data "
        "will be 0. Else it will be random"))
{
  bool zero_input;
  size_t string_length = 5;
  std::string string_data = "FOOBAR";

  SUBCASE("zero_input true") { zero_input = true; }
  SUBCASE("zero_input false") { zero_input = false; }
  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  input1.shape_ = {3};
  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status =
      dataloader.GenerateData(inputs, zero_input, string_length, string_data);
  REQUIRE(status.IsOk());
  CHECK_EQ(dataloader.GetDataStreamsCount(), 1);
  CHECK_EQ(dataloader.GetTotalSteps(0), 1);
  const uint8_t* data_ptr{nullptr};
  size_t batch1_size;

  status = dataloader.GetInputData(input1, 0, 0, &data_ptr, &batch1_size);
  REQUIRE(status.IsOk());

  const int32_t* input_data = reinterpret_cast<const int32_t*>(data_ptr);
  if (zero_input) {
    CHECK_EQ(input_data[0], 0);
    CHECK_EQ(input_data[1], 0);
    CHECK_EQ(input_data[2], 0);
  } else {
    CHECK_NE(input_data[0], 0);
    CHECK_NE(input_data[1], 0);
    CHECK_NE(input_data[2], 0);
  }
  // 3 elements of int32 data is 12 bytes
  CHECK_EQ(batch1_size, 12);
}

TEST_CASE(
    "dataloader: GenerateData: BYTES" *
    doctest::description(
        "Calling GenerateData for BYTES datatype should result in a single "
        "stream with one step. The zero-input flag is ignored. If string_data "
        "is not null, it will be used. Else it will be a random string of "
        "length string_length"))
{
  bool zero_input = false;
  size_t string_length = 5;
  std::string string_data;

  SUBCASE("valid string_data") { string_data = "FOOBAR"; }
  SUBCASE("empty string_data") { string_data = ""; }

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  input1.datatype_ = "BYTES";
  input1.shape_ = {3};
  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status =
      dataloader.GenerateData(inputs, zero_input, string_length, string_data);
  REQUIRE(status.IsOk());
  CHECK_EQ(dataloader.GetDataStreamsCount(), 1);
  CHECK_EQ(dataloader.GetTotalSteps(0), 1);

  const uint8_t* data_ptr{nullptr};
  size_t batch1_size;

  status = dataloader.GetInputData(input1, 0, 0, &data_ptr, &batch1_size);
  REQUIRE(status.IsOk());


  // For string data, the result should be a 32-bit number indicating the data
  // length, and then 1 byte per letter
  //
  // For "FOOBAR", the length would be 10 bytes:
  //    4 bytes to indicate the string length (the number 6)
  //    1 byte for each letter
  //
  // For empty string, the string length would instead be the value in
  // string_length (5 in this case), and the characters would be random for
  // each entry in the batch. Thus, the data length would be 9 bytes
  //
  // For a shape of [3], this data would be repeated 3 times

  if (string_data.empty()) {
    // 3 elements of 9 bytes is 27
    CHECK_EQ(batch1_size, 27);

    const char* char_data = reinterpret_cast<const char*>(data_ptr);

    // Check all 3 entries in the "batch" of shape [3]
    for (size_t i = 0; i < 3; i++) {
      size_t start_index = 9 * i;

      // The first 4 bytes are an int32 indicating the number of characters
      const int32_t* int32_data =
          reinterpret_cast<const int32_t*>(&char_data[start_index]);
      CHECK_EQ(int32_data[0], 5);

      // All of the characters should be in the specified character_set
      for (size_t j = start_index + 4; j < start_index + 9; j++) {
        CHECK_NE(character_set.find(char_data[j]), std::string::npos);
      }
    }

  } else {
    // 3 elements of 10 bytes is 30
    CHECK_EQ(batch1_size, 30);

    const int32_t* int32_data = reinterpret_cast<const int32_t*>(data_ptr);
    const char* char_data = reinterpret_cast<const char*>(data_ptr);
    CHECK_EQ(int32_data[0], 6);
    CHECK_EQ(char_data[4], 'F');
    CHECK_EQ(char_data[5], 'O');
    CHECK_EQ(char_data[6], 'O');
    CHECK_EQ(char_data[7], 'B');
    CHECK_EQ(char_data[8], 'A');
    CHECK_EQ(char_data[9], 'R');

    // The data would repeat two more times for shape of [3]
    for (size_t i = 10; i < 30; i++) {
      CHECK_EQ(char_data[i - 10], char_data[i]);
    }
  }
}

TEST_CASE("dataloader: GenerateData: Dynamic shape")
{
  bool zero_input = false;
  size_t string_length = 5;
  std::string string_data;

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  input1.shape_ = {-1};

  std::string expected_message =
      "input INPUT1 contains dynamic shape, provide shapes to send along with "
      "the request";

  SUBCASE("BYTES") { input1.datatype_ = "BYTES"; }
  SUBCASE("non-BYTES") { input1.datatype_ = "INT32"; }

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();
  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status =
      dataloader.GenerateData(inputs, zero_input, string_length, string_data);
  REQUIRE(status.IsOk() == false);
  CHECK_EQ(status.Message(), expected_message);
}

TEST_CASE(
    "dataloader: ReadDataFromDir: Error reading input file" *
    doctest::description(
        "When there is an error reading an input data file, the error should "
        "bubble up to the return value of ReadDataFromDir"))
{
  MockDataLoader dataloader;

  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");

  std::string dir{"fake/path"};

  SUBCASE("BYTES (string) data") { input1.datatype_ = "BYTES"; }
  SUBCASE("Raw Binary data") { input1.datatype_ = "INT32"; }

  inputs->insert(std::make_pair(input1.name_, input1));
  cb::Error status = dataloader.ReadDataFromDir(inputs, outputs, dir);
  CHECK(status.IsOk() == false);
}

TEST_CASE(
    "dataloader: ReadDataFromDir: Error reading output file" *
    doctest::description(
        "When there is an error reading an output data file, an error is NOT "
        "raised from ReadDataFromDir, and instead GetOutputData will return "
        "nullptr with a batch1_size of 0"))
{
  MockDataLoader dataloader;

  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor output1 = TestDataLoader::CreateTensor("OUTPUT1");

  std::string dir{"fake/path"};

  SUBCASE("BYTES (string) data") { output1.datatype_ = "BYTES"; }
  SUBCASE("Raw Binary data") { output1.datatype_ = "INT32"; }

  outputs->insert(std::make_pair(output1.name_, output1));
  cb::Error status = dataloader.ReadDataFromDir(inputs, outputs, dir);
  CHECK(status.IsOk() == true);

  const uint8_t* data_ptr{nullptr};
  size_t batch1_size;

  dataloader.GetOutputData("OUTPUT1", 0, 0, &data_ptr, &batch1_size);
  CHECK(data_ptr == nullptr);
  CHECK(batch1_size == 0);
}

TEST_CASE(
    "dataloader: ReadDataFromDir: Mismatching Input Data" *
    doctest::description("Successfully reading input files but having a "
                         "mismatch will result in an error being thrown"))
{
  MockDataLoader dataloader;

  std::string datatype;
  std::string expected_error_message;

  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  ModelTensor output1 = TestDataLoader::CreateTensor("OUTPUT1");

  std::string dir{"mocked_out"};

  SUBCASE("BYTES (string) data")
  {
    datatype = "BYTES";
    std::vector<std::string> string_data;

    SUBCASE("Dynamic shape")
    {
      input1.shape_ = {-1};
      expected_error_message =
          "input INPUT1 contains dynamic shape, provide shapes to send along "
          "with the request";
    }
    SUBCASE("Supplied shape")
    {
      input1.shape_ = {1};
      string_data = {"InStr", "ExtraStr"};

      expected_error_message =
          "provided data for input INPUT1 has 2 elements, expect 1";
    }

    EXPECT_CALL(dataloader, ReadTextFile(testing::_, testing::_))
        .WillOnce(testing::DoAll(
            testing::SetArgPointee<1>(string_data),
            testing::Return(cb::Error::Success)));
  }
  SUBCASE("Raw Binary data")
  {
    datatype = "INT32";
    std::vector<char> char_data;

    SUBCASE("Dynamic shape")
    {
      input1.shape_ = {-1};
      expected_error_message =
          "input INPUT1 contains dynamic shape, provide shapes to send along "
          "with the request";
    }
    SUBCASE("Supplied shape")
    {
      // An INT32 of shape {1} will be 4 bytes. However, we are supplying 5
      // bytes via char_data.
      input1.shape_ = {1};
      char_data = {'0', '0', '0', '7', '5'};
      expected_error_message =
          "provided data for input INPUT1 has byte size 5, expect 4";
    }

    EXPECT_CALL(dataloader, ReadFile(testing::_, testing::_))
        .WillOnce(testing::DoAll(
            testing::SetArgPointee<1>(char_data),
            testing::Return(cb::Error::Success)));
  }

  input1.datatype_ = datatype;
  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status = dataloader.ReadDataFromDir(inputs, outputs, dir);
  REQUIRE(status.IsOk() == false);
  CHECK(status.Message() == expected_error_message);
}

// FIXME TMA-1210 -- the output data is not being ignored here and no error is
// thrown, despite the mismatch
// TEST_CASE(
//    "dataloader: ReadDataFromDir: Mismatching Output Data" *
//    doctest::description("Successfully reading output files but having a "
//                         "mismatch will result in the data being ignored"))
//{
//  MockDataLoader dataloader;
//
//  std::string datatype;
//
//  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
//  std::shared_ptr<ModelTensorMap> outputs =
//  std::make_shared<ModelTensorMap>();
//
//  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
//  ModelTensor output1 = TestDataLoader::CreateTensor("OUTPUT1");
//
//  std::string dir{"mocked_out"};
//
//  std::vector<char> char_data{'0', '0', '0', '7', '5'};
//
//  std::vector<std::string> string_data{"InStr", "ExtraStr"};
//
//  SUBCASE("BYTES (string) data")
//  {
//    datatype = "BYTES";
//    EXPECT_CALL(dataloader, ReadTextFile(testing::_, testing::_))
//        .WillOnce(testing::DoAll(
//            testing::SetArgPointee<1>(string_data),
//            testing::Return(cb::Error::Success)));
//
//    SUBCASE("Dynamic shape") { output1.shape_ = {-1}; }
//    SUBCASE("Supplied shape") { output1.shape_ = {1}; }
//  }
//  SUBCASE("Raw Binary data")
//  {
//    datatype = "INT32";
//    EXPECT_CALL(dataloader, ReadFile(testing::_, testing::_))
//        .WillOnce(testing::DoAll(
//            testing::SetArgPointee<1>(char_data),
//            testing::Return(cb::Error::Success)));
//
//    SUBCASE("Dynamic shape") { input1.shape_ = {-1}; }
//    SUBCASE("Supplied shape") { input1.shape_ = {1}; }
//  }
//
//  output1.datatype_ = datatype;
//  outputs->insert(std::make_pair(output1.name_, output1));
//
//  cb::Error status = dataloader.ReadDataFromDir(inputs, outputs, dir);
//  REQUIRE(status.IsOk() == true);
//
//  // Confirm that the data is not in the dataloader
//  const uint8_t* data_ptr{nullptr};
//  size_t batch1_size;
//
//  dataloader.GetOutputData("OUTPUT1", 0, 0, &data_ptr, &batch1_size);
//  CHECK(data_ptr == nullptr);
//  CHECK(batch1_size == 0);
//}

TEST_CASE(
    "dataloader: ReadDataFromDir: Valid Data" *
    doctest::description("Successfully reading files will always result in a "
                         "single stream with a single step"))
{
  MockDataLoader dataloader;

  std::string datatype;

  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::CreateTensor("INPUT1");
  ModelTensor output1 = TestDataLoader::CreateTensor("OUTPUT1");

  std::string dir{"mocked_out"};

  std::vector<char> input_char_data{'0', '0', '0', '7'};
  std::vector<char> output_char_data{'0', '0', '0', '3'};

  std::vector<std::string> input_string_data{"InStr"};
  std::vector<std::string> output_string_data{"OutStr"};

  std::vector<char> expected_input;
  std::vector<char> expected_output;

  SUBCASE("BYTES (string) data")
  {
    datatype = "BYTES";

    expected_input = {'\5', '\0', '\0', '\0', 'I', 'n', 'S', 't', 'r'};
    expected_output = {'\6', '\0', '\0', '\0', 'O', 'u', 't', 'S', 't', 'r'};

    EXPECT_CALL(dataloader, ReadTextFile(testing::_, testing::_))
        .WillOnce(testing::DoAll(
            testing::SetArgPointee<1>(input_string_data),
            testing::Return(cb::Error::Success)))
        .WillOnce(testing::DoAll(
            testing::SetArgPointee<1>(output_string_data),
            testing::Return(cb::Error::Success)));
  }
  SUBCASE("Raw Binary data")
  {
    datatype = "INT32";

    expected_input = input_char_data;
    expected_output = output_char_data;

    EXPECT_CALL(dataloader, ReadFile(testing::_, testing::_))
        .WillOnce(testing::DoAll(
            testing::SetArgPointee<1>(input_char_data),
            testing::Return(cb::Error::Success)))
        .WillOnce(testing::DoAll(
            testing::SetArgPointee<1>(output_char_data),
            testing::Return(cb::Error::Success)));
  }

  input1.datatype_ = datatype;
  output1.datatype_ = datatype;

  inputs->insert(std::make_pair(input1.name_, input1));
  outputs->insert(std::make_pair(output1.name_, output1));

  cb::Error status = dataloader.ReadDataFromDir(inputs, outputs, dir);
  REQUIRE(status.IsOk() == true);
  CHECK_EQ(dataloader.GetDataStreamsCount(), 1);
  CHECK_EQ(dataloader.GetTotalSteps(0), 1);

  // Validate input and output data
  const uint8_t* data_ptr{nullptr};
  size_t batch1_size;

  dataloader.GetInputData(input1, 0, 0, &data_ptr, &batch1_size);
  const char* input_data = reinterpret_cast<const char*>(data_ptr);
  REQUIRE(batch1_size == expected_input.size());
  for (size_t i = 0; i < batch1_size; i++) {
    CHECK(input_data[i] == expected_input[i]);
  }

  dataloader.GetOutputData("OUTPUT1", 0, 0, &data_ptr, &batch1_size);
  const char* output_data = reinterpret_cast<const char*>(data_ptr);
  REQUIRE(batch1_size == expected_output.size());
  for (size_t i = 0; i < batch1_size; i++) {
    CHECK(output_data[i] == expected_output[i]);
  }
}

}}  // namespace triton::perfanalyzer
