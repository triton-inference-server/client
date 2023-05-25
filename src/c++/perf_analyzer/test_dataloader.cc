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


// FIXME TKG things to test:
// TMA-1214 ReadFromDir
// TMA-1215 String data
// TMA-1216 Bytes data


/// Helper class for testing the DataLoader
///
class TestDataLoader {
 public:
  // Static function to create a generic ModelTensor
  //
  static ModelTensor GetTensor(std::string name)
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
  CHECK_FALSE(status.IsOk());
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

TEST_CASE("dataloader: ReadDataFromJSON: Bad Json")
{
  std::string json_str{"bad json text"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
  CHECK_FALSE(status.IsOk());
  CHECK_EQ(
      status.Message(),
      "failed to parse the specified json file for reading provided data");
}

TEST_CASE(
    "dataloader: ReadDataFromJSON: No Data" *
    doctest::description("If there is no field called data in the json, an "
                         "error should be thrown"))
{
  std::string json_str{R"({ "notdata" : 5})"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
  CHECK_FALSE(status.IsOk());
  CHECK_EQ(status.Message(), "The json file doesn't contain data field");
}

// FIXME TMA-1210
// TEST_CASE(
//    "dataloader: ReadDataFromJSON: Mismatch Shape" *
//    doctest::description(
//        "If the size of the provided Input is not in line with the Tensor's "
//        "shape, then an error should be thrown"))
//{
//  std::string json_str{R"({"data": [{ "INPUT1": [1,2] }]})"};
//
//
//  MockDataLoader dataloader;
//  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
//  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
//  input1.shape_ = {3};
//  inputs->insert(std::make_pair(input1.name_, input1));
//
//  std::shared_ptr<ModelTensorMap> outputs =
//  std::make_shared<ModelTensorMap>();
//
//  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
//  CHECK_FALSE(status.IsOk());
//  CHECK_EQ(status.Message(), "TKG");
//}


TEST_CASE(
    "dataloader: ReadDataFromJSON: Mismatch Input and Output" *
    doctest::description(
        "If the size of the provided Input and validation Output data are "
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

  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
  CHECK_FALSE(status.IsOk());
  CHECK_EQ(
      status.Message(),
      "The 'validation_data' field doesn't align with 'data' field in the json "
      "file");
}

TEST_CASE("dataloader: ReadDataFromJSON: Valid Data")
{
  std::string json_str{R"({
   "data": [
     { "INPUT1": [1] },
     { "INPUT1": [2] },
     { "INPUT1": [3] }
   ],
   "validation_data": [
    { "OUTPUT1": [4] },
    { "OUTPUT1": [5] },
    { "OUTPUT1": [6] }
   ]})"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
  ModelTensor output1 = TestDataLoader::GetTensor("OUTPUT1");

  inputs->insert(std::make_pair(input1.name_, input1));
  outputs->insert(std::make_pair(output1.name_, output1));

  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
  REQUIRE(status.IsOk());
  CHECK_EQ(dataloader.GetDataStreamsCount(), 1);
  CHECK_EQ(dataloader.GetTotalSteps(0), 3);

  // Confirm the correct data is in the dataloader
  //
  const uint8_t* data_ptr{nullptr};
  size_t batch_size;
  std::vector<int64_t> shape;

  dataloader.GetInputShape(input1, 0, 1, &shape);
  CHECK_EQ(shape.size(), 1);
  CHECK_EQ(shape[0], 1);

  dataloader.GetInputData(input1, 0, 1, &data_ptr, &batch_size);
  auto data = *reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(data, 2);
  CHECK_EQ(batch_size, 4);

  dataloader.GetOutputData("OUTPUT1", 0, 2, &data_ptr, &batch_size);
  auto data2 = *reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(data2, 6);
  CHECK_EQ(batch_size, 4);
}

TEST_CASE("dataloader: ReadDataFromJSON: Multiple Streams Invalid Cases")
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

    ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
    ModelTensor output1 = TestDataLoader::GetTensor("OUTPUT1");
    input1.shape_ = {2};
    inputs->insert(std::make_pair(input1.name_, input1));
    outputs->insert(std::make_pair(output1.name_, output1));

    MockDataLoader dataloader;
    cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_data);
    CHECK_FALSE(status.IsOk());
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

TEST_CASE("dataloader: ReadDataFromJSON: Multiple Streams Valid")
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

  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
  ModelTensor output1 = TestDataLoader::GetTensor("OUTPUT1");
  input1.shape_ = {2};
  inputs->insert(std::make_pair(input1.name_, input1));
  outputs->insert(std::make_pair(output1.name_, output1));

  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
  REQUIRE(status.IsOk());
  CHECK_EQ(dataloader.GetDataStreamsCount(), 2);
  CHECK_EQ(dataloader.GetTotalSteps(0), 2);
  CHECK_EQ(dataloader.GetTotalSteps(1), 1);

  // Confirm the correct data is in the dataloader
  //
  const uint8_t* data_ptr{nullptr};
  size_t batch_size;

  dataloader.GetInputData(input1, 0, 1, &data_ptr, &batch_size);

  const int32_t* data = reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(data[0], 2);
  CHECK_EQ(data[1], 3);
  // 2 elements of int32 data is 8 bytes
  CHECK_EQ(batch_size, 8);

  dataloader.GetOutputData("OUTPUT1", 1, 0, &data_ptr, &batch_size);
  const int32_t* data2 = reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(data2[0], 40);
  CHECK_EQ(batch_size, 4);
}


TEST_CASE(
    "dataloader: ReadDataFromJSON: Missing Shape" *
    doctest::description(
        "If a tensor's shape is dynamic (-1), then it needs to be provided via "
        "--shape option (which is not visable to this testing), or via a shape "
        "option in the json. If not, an error is thrown"))
{
  std::string json_str{R"({"data": [{ "INPUT1": [1,2,3] } ]})"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
  input1.shape_ = {-1};

  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
  CHECK_EQ(status.IsOk(), false);
  CHECK_EQ(
      status.Message(),
      "The variable-sized tensor \"INPUT1\" is missing shape, see --shape "
      "option.");
}

// FIXME TMA-1210
// TEST_CASE(
//    "dataloader: ReadDataFromJSON: Supplied Shape is wrong" *
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
//  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
//  input1.shape_ = {-1};
//
//  inputs->insert(std::make_pair(input1.name_, input1));
//
//  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
//  CHECK_EQ(status.IsOk(), false);
//  CHECK_EQ(status.Message(), "FIXME");
//}


TEST_CASE(
    "dataloader: ReadDataFromJSON: Supplied Shape is valid" *
    doctest::description("Supply the dynamic shape for an input"))
{
  std::string json_str{R"({"data": [{
     "INPUT1": { "shape": [3,2], "content": [1,2,3,4,5,6] }
    }]})"};

  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
  input1.shape_ = {-1};

  inputs->insert(std::make_pair(input1.name_, input1));

  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str);
  REQUIRE(status.IsOk());

  std::vector<int64_t> shape;
  dataloader.GetInputShape(input1, 0, 0, &shape);
  CHECK_EQ(shape.size(), 2);
  CHECK_EQ(shape[0], 3);
  CHECK_EQ(shape[1], 2);
}

// FIXME TMA 1211
// TEST_CASE(
//    "dataloader: ReadDataFromJSON: Multiple Calls" *
//    doctest::description(
//        "ReadDataFromJson can be called multiple times. The data should "
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
//  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
//  ModelTensor output1 = TestDataLoader::GetTensor("OUTPUT1");
//
//  inputs->insert(std::make_pair(input1.name_, input1));
//  outputs->insert(std::make_pair(output1.name_, output1));
//
//  cb::Error status = dataloader.ReadDataFromJSON(inputs, outputs, json_str1);
//  CHECK(status.IsOk());
//  CHECK_EQ(dataloader.GetDataStreamsCount(), 1);
//  CHECK_EQ(dataloader.GetTotalSteps(0), 2);
//
//  status = dataloader.ReadDataFromJSON(inputs, outputs, json_str2);
//  CHECK(status.IsOk());
//  CHECK_EQ(dataloader.GetDataStreamsCount(), 3);
//  CHECK_EQ(dataloader.GetTotalSteps(0), 2);
//  CHECK_EQ(dataloader.GetTotalSteps(1), 1);
//  CHECK_EQ(dataloader.GetTotalSteps(2), 1);
//
//  status = dataloader.ReadDataFromJSON(inputs, outputs, json_str3);
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

  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
  input1.is_shape_tensor_ = true;
  inputs->insert(std::make_pair(input1.name_, input1));

  bool zero_input = true;
  size_t string_length = 5;
  std::string string_data = "FOOBAR";
  cb::Error status =
      dataloader.GenerateData(inputs, zero_input, string_length, string_data);
  CHECK_FALSE(status.IsOk());
  CHECK_EQ(
      status.Message(),
      "can not generate data for shape tensor 'INPUT1', user-provided data is "
      "needed.");
}


TEST_CASE(
    "dataloader: GenerateData: Zero Input" *
    doctest::description(
        "Calling GenerateData with the zero_input flag set to true should "
        "result in a single stream with one step with all data having the "
        "value of 0"))
{
  MockDataLoader dataloader;
  std::shared_ptr<ModelTensorMap> inputs = std::make_shared<ModelTensorMap>();
  std::shared_ptr<ModelTensorMap> outputs = std::make_shared<ModelTensorMap>();

  ModelTensor input1 = TestDataLoader::GetTensor("INPUT1");
  input1.shape_ = {3};
  inputs->insert(std::make_pair(input1.name_, input1));

  bool zero_input = true;
  size_t string_length = 5;
  std::string string_data = "FOOBAR";
  CHECK_EQ(dataloader.GetDataStreamsCount(), 0);
  cb::Error status =
      dataloader.GenerateData(inputs, zero_input, string_length, string_data);
  REQUIRE(status.IsOk());
  CHECK_EQ(dataloader.GetDataStreamsCount(), 1);

  const uint8_t* data_ptr{nullptr};
  size_t batch_size;

  status = dataloader.GetInputData(input1, 0, 0, &data_ptr, &batch_size);
  REQUIRE(status.IsOk());

  const int32_t* data = reinterpret_cast<const int32_t*>(data_ptr);
  CHECK_EQ(data[0], 0);
  CHECK_EQ(data[1], 0);
  CHECK_EQ(data[2], 0);
  // 3 elements of int32 data is 12 bytes
  CHECK_EQ(batch_size, 12);
}

}}  // namespace triton::perfanalyzer
