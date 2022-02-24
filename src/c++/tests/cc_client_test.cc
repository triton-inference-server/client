// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gtest/gtest.h"

#include "grpc_client.h"
#include "http_client.h"

namespace tc = triton::client;

namespace {

// This test must be run with a running Triton server,
// check L0_grpc in server repo for the setup.
class GrpcClientTest : public ::testing::Test {
 public:
  GrpcClientTest()
      : url_("localhost:8001"),
        model_name_("onnx_int32_int32_int32"), shape_{1, 16}, dtype_("INT32")
  {
  }

  void SetUp() override
  {
    auto err = tc::InferenceServerGrpcClient::Create(&client_, url_);
    ASSERT_TRUE(err.IsOk())
        << "failed to create GRPC client: " << err.Message();

    // Initialize 3 sets of inputs, each with 16 elements
    for (size_t i = 0; i < 3; ++i) {
      input_data_.emplace_back();
      for (size_t j = 0; j < 16; ++j) {
        input_data_.back().emplace_back(i * 16 + j);
      }
    }
  }

  std::string url_;
  std::string model_name_;
  std::unique_ptr<tc::InferenceServerGrpcClient> client_;
  std::vector<std::vector<int32_t>> input_data_;
  std::vector<int64_t> shape_;
  std::string dtype_;
};

TEST_F(GrpcClientTest, InferMulti)
{
  tc::Error err = tc::Error::Success;
  // Create 3 sets of 'options', 'inputs', 'outputs', technically
  // only InferInput can not be reused for requests that are sent
  // concurrently, here use distinct objects for all 'options',
  // 'inputs', and 'outputs' for simplicity.
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  for (size_t i = 0; i < 3; ++i) {
    options.emplace_back(model_name_);
    // Not swap
    options.back().model_version_ = "1";
    inputs.emplace_back();
    auto& curr_inputs = inputs.back();
    curr_inputs.emplace_back();
    err = tc::InferInput::Create(&curr_inputs.back(), "INPUT0", shape_, dtype_);
    ASSERT_TRUE(err.IsOk())
        << "failed to create inference input: " << err.Message();
    const auto& input_0 = input_data_[i % input_data_.size()];
    err = curr_inputs.back()->AppendRaw(
        reinterpret_cast<const uint8_t*>(input_0.data()),
        input_0.size() * sizeof(int32_t));
    ASSERT_TRUE(err.IsOk()) << "failed to set input data: " << err.Message();
    curr_inputs.emplace_back();
    err = tc::InferInput::Create(&curr_inputs.back(), "INPUT1", shape_, dtype_);
    ASSERT_TRUE(err.IsOk())
        << "failed to create inference input: " << err.Message();
    const auto& input_1 = input_data_[(i + 1) % input_data_.size()];
    err = curr_inputs.back()->AppendRaw(
        reinterpret_cast<const uint8_t*>(input_1.data()),
        input_1.size() * sizeof(int32_t));
    ASSERT_TRUE(err.IsOk()) << "failed to set input data: " << err.Message();

    tc::InferRequestedOutput* output;
    outputs.emplace_back();
    err = tc::InferRequestedOutput::Create(&output, "OUTPUT0");
    ASSERT_TRUE(err.IsOk())
        << "failed to create inference output: " << err.Message();
    outputs.back().emplace_back(output);
    err = tc::InferRequestedOutput::Create(&output, "OUTPUT1");
    ASSERT_TRUE(err.IsOk())
        << "failed to create inference output: " << err.Message();
    outputs.back().emplace_back(output);

    expected_outputs.emplace_back();
    {
      auto& expected = expected_outputs.back()["OUTPUT0"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] + input_1[i]);
      }
    }
    {
      auto& expected = expected_outputs.back()["OUTPUT1"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] - input_1[i]);
      }
    }
  }

  std::vector<tc::InferResult*> results;
  err = client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  ASSERT_EQ(results.size(), inputs.size()) << "unexpected number of results";
  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_TRUE(results[i]->RequestStatus().IsOk());
    for (const auto& expected : expected_outputs[i]) {
      const uint8_t* buf = nullptr;
      size_t byte_size = 0;
      err = results[i]->RawData(expected.first, &buf, &byte_size);
      ASSERT_TRUE(err.IsOk()) << "failed to retrieve output '" << expected.first
                              << "': " << err.Message();
      ASSERT_EQ(byte_size, (expected.second.size() * sizeof(int32_t)));
      EXPECT_EQ(memcmp(buf, expected.second.data(), byte_size), 0);
    }
  }
}

TEST_F(GrpcClientTest, InferMultiDifferentOutputs)
{
  EXPECT_FALSE(true) << "Not implemented";
}

TEST_F(GrpcClientTest, InferMultiDifferentOptions)
{
  EXPECT_FALSE(true) << "Not implemented";
}

TEST_F(GrpcClientTest, InferMultiOneOption)
{
  EXPECT_FALSE(true) << "Not implemented";
}

TEST_F(GrpcClientTest, InferMultiOneOutput)
{
  EXPECT_FALSE(true) << "Not implemented";
}

TEST_F(GrpcClientTest, InferMultiNoOutput)
{
  EXPECT_FALSE(true) << "Not implemented";
}

TEST_F(GrpcClientTest, InferMultiMismatchOptions)
{
  EXPECT_FALSE(true) << "Not implemented";
}

TEST_F(GrpcClientTest, InferMultiMismatchOutputs)
{
  EXPECT_FALSE(true) << "Not implemented";
}

// [WIP] AsyncInferMulti

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}