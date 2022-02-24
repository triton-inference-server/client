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
template <typename ClientType>
class ClientTest : public ::testing::Test {
 public:
  ClientTest()
      : model_name_("onnx_int32_int32_int32"), shape_{1, 16}, dtype_("INT32")
  {
  }

  void SetUp() override
  {
    std::string url;
    if (std::is_same<ClientType, tc::InferenceServerGrpcClient>::value) {
      url = "localhost:8001";
    } else if (std::is_same<ClientType, tc::InferenceServerHttpClient>::value) {
      url = "localhost:8000";
    } else {
      ASSERT_TRUE(false) << "Unrecognized client class type '"
                         << typeid(ClientType).name() << "'";
    }
    auto err = ClientType::Create(&this->client_, url);
    ASSERT_TRUE(err.IsOk())
        << "failed to create GRPC client: " << err.Message();

    // Initialize 3 sets of inputs, each with 16 elements
    for (size_t i = 0; i < 3; ++i) {
      this->input_data_.emplace_back();
      for (size_t j = 0; j < 16; ++j) {
        this->input_data_.back().emplace_back(i * 16 + j);
      }
    }
  }

  tc::Error PrepareInputs(
      const std::vector<int32_t>& input_0, const std::vector<int32_t>& input_1,
      std::vector<tc::InferInput*>* inputs)
  {
    inputs->emplace_back();
    auto err = tc::InferInput::Create(
        &inputs->back(), "INPUT0", this->shape_, this->dtype_);
    if (!err.IsOk()) {
      return err;
    }
    err = inputs->back()->AppendRaw(
        reinterpret_cast<const uint8_t*>(input_0.data()),
        input_0.size() * sizeof(int32_t));
    if (!err.IsOk()) {
      return err;
    }
    inputs->emplace_back();
    err = tc::InferInput::Create(
        &inputs->back(), "INPUT1", this->shape_, this->dtype_);
    if (!err.IsOk()) {
      return err;
    }
    err = inputs->back()->AppendRaw(
        reinterpret_cast<const uint8_t*>(input_1.data()),
        input_1.size() * sizeof(int32_t));
    if (!err.IsOk()) {
      return err;
    }
    return tc::Error::Success;
  }

  void ValidateOutput(
      const std::vector<tc::InferResult*>& results,
      const std::vector<std::map<std::string, std::vector<int32_t>>>&
          expected_outputs)
  {
    ASSERT_EQ(results.size(), expected_outputs.size())
        << "unexpected number of results";
    for (size_t i = 0; i < results.size(); ++i) {
      ASSERT_TRUE(results[i]->RequestStatus().IsOk());
      for (const auto& expected : expected_outputs[i]) {
        const uint8_t* buf = nullptr;
        size_t byte_size = 0;
        auto err = results[i]->RawData(expected.first, &buf, &byte_size);
        ASSERT_TRUE(err.IsOk())
            << "failed to retrieve output '" << expected.first
            << "' for result " << i << ": " << err.Message();
        ASSERT_EQ(byte_size, (expected.second.size() * sizeof(int32_t)));
        EXPECT_EQ(memcmp(buf, expected.second.data(), byte_size), 0);
      }
    }
  }

  std::string model_name_;
  std::unique_ptr<ClientType> client_;
  std::vector<std::vector<int32_t>> input_data_;
  std::vector<int64_t> shape_;
  std::string dtype_;
};


TYPED_TEST_SUITE_P(ClientTest);

TYPED_TEST_P(ClientTest, InferMulti)
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
    options.emplace_back(this->model_name_);
    // Not swap
    options.back().model_version_ = "1";

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

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
  err = this->client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, InferMultiDifferentOutputs)
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
    options.emplace_back(this->model_name_);
    // Not swap
    options.back().model_version_ = "1";

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

    // Explicitly request different output for different request
    // 0: request 0
    // 1: request 1
    // 2: no request (both will be returned)
    tc::InferRequestedOutput* output;
    outputs.emplace_back();
    expected_outputs.emplace_back();
    if (i != 1) {
      if (i != 2) {
        err = tc::InferRequestedOutput::Create(&output, "OUTPUT0");
        ASSERT_TRUE(err.IsOk())
            << "failed to create inference output: " << err.Message();
        outputs.back().emplace_back(output);
      }

      auto& expected = expected_outputs.back()["OUTPUT0"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] + input_1[i]);
      }
    }
    if (i != 0) {
      if (i != 2) {
        err = tc::InferRequestedOutput::Create(&output, "OUTPUT1");
        ASSERT_TRUE(err.IsOk())
            << "failed to create inference output: " << err.Message();
        outputs.back().emplace_back(output);
      }

      auto& expected = expected_outputs.back()["OUTPUT1"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] - input_1[i]);
      }
    }
  }

  std::vector<tc::InferResult*> results;
  err = this->client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, InferMultiDifferentOptions)
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
    options.emplace_back(this->model_name_);
    // output will be different based on version
    // v1 : not swap
    // v2 : swap
    // v3 : swap
    size_t version = (i % 3) + 1;
    options.back().model_version_ = std::to_string(version);

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

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
      auto& expected =
          expected_outputs.back()[version == 1 ? "OUTPUT0" : "OUTPUT1"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] + input_1[i]);
      }
    }
    {
      auto& expected =
          expected_outputs.back()[version == 1 ? "OUTPUT1" : "OUTPUT0"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] - input_1[i]);
      }
    }
  }

  std::vector<tc::InferResult*> results;
  err = this->client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, InferMultiOneOption)
{
  // Create only 1 sets of 'options'.
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  options.emplace_back(this->model_name_);
  // Not swap
  options.back().model_version_ = "1";
  for (size_t i = 0; i < 3; ++i) {
    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

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
  err = this->client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, InferMultiOneOutput)
{
  // Create only 1 sets of 'outputs', but combine with different 'options'
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  for (size_t i = 0; i < 3; ++i) {
    options.emplace_back(this->model_name_);
    // output will be different based on version
    // v1 : not swap
    // v2 : swap
    // v3 : swap
    size_t version = (i % 3) + 1;
    options.back().model_version_ = std::to_string(version);

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

    tc::InferRequestedOutput* output;
    outputs.emplace_back();
    err = tc::InferRequestedOutput::Create(&output, "OUTPUT0");
    ASSERT_TRUE(err.IsOk())
        << "failed to create inference output: " << err.Message();
    outputs.back().emplace_back(output);

    expected_outputs.emplace_back();
    {
      auto& expected = expected_outputs.back()["OUTPUT0"];
      if (version == 1) {
        for (size_t i = 0; i < 16; ++i) {
          expected.emplace_back(input_0[i] + input_1[i]);
        }
      } else {
        for (size_t i = 0; i < 16; ++i) {
          expected.emplace_back(input_0[i] - input_1[i]);
        }
      }
    }
  }

  std::vector<tc::InferResult*> results;
  err = this->client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}
TYPED_TEST_P(ClientTest, InferMultiNoOutput)
{
  // Not specifying 'outputs' at all, but combine with different 'options'
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  for (size_t i = 0; i < 3; ++i) {
    options.emplace_back(this->model_name_);
    // output will be different based on version
    // v1 : not swap
    // v2 : swap
    // v3 : swap
    size_t version = (i % 3) + 1;
    options.back().model_version_ = std::to_string(version);

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

    expected_outputs.emplace_back();
    {
      auto& expected =
          expected_outputs.back()[version == 1 ? "OUTPUT0" : "OUTPUT1"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] + input_1[i]);
      }
    }
    {
      auto& expected =
          expected_outputs.back()[version == 1 ? "OUTPUT1" : "OUTPUT0"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] - input_1[i]);
      }
    }
  }

  std::vector<tc::InferResult*> results;
  err = this->client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, InferMultiMismatchOptions)
{
  // Create mismatch number of 'options'.
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  options.emplace_back(this->model_name_);
  options.emplace_back(this->model_name_);
  for (size_t i = 0; i < 3; ++i) {
    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

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
  }

  std::vector<tc::InferResult*> results;
  err = this->client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_FALSE(err.IsOk()) << "Expect InferMulti() to fail";
}

TYPED_TEST_P(ClientTest, InferMultiMismatchOutputs)
{
  // Create mismatch number of 'outputs'.
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  for (size_t i = 0; i < 3; ++i) {
    options.emplace_back(this->model_name_);
    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

    if (i != 2) {
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
    }
  }

  std::vector<tc::InferResult*> results;
  err = this->client_->InferMulti(&results, options, inputs, outputs);
  ASSERT_FALSE(err.IsOk()) << "Expect InferMulti() to fail";
}

TYPED_TEST_P(ClientTest, AsyncInferMulti)
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
    options.emplace_back(this->model_name_);
    // Not swap
    options.back().model_version_ = "1";

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

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
  std::condition_variable cv;
  std::mutex mu;
  err = this->client_->AsyncInferMulti(
      [&results, &cv, &mu](std::vector<tc::InferResult*> res) {
        {
          std::lock_guard<std::mutex> lk(mu);
          results.swap(res);
        }
        cv.notify_one();
      },
      options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [this, &results] { return !results.empty(); });
  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, AsyncInferMultiDifferentOutputs)
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
    options.emplace_back(this->model_name_);
    // Not swap
    options.back().model_version_ = "1";

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

    // Explicitly request different output for different request
    // 0: request 0
    // 1: request 1
    // 2: no request (both will be returned)
    tc::InferRequestedOutput* output;
    outputs.emplace_back();
    expected_outputs.emplace_back();
    if (i != 1) {
      if (i != 2) {
        err = tc::InferRequestedOutput::Create(&output, "OUTPUT0");
        ASSERT_TRUE(err.IsOk())
            << "failed to create inference output: " << err.Message();
        outputs.back().emplace_back(output);
      }

      auto& expected = expected_outputs.back()["OUTPUT0"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] + input_1[i]);
      }
    }
    if (i != 0) {
      if (i != 2) {
        err = tc::InferRequestedOutput::Create(&output, "OUTPUT1");
        ASSERT_TRUE(err.IsOk())
            << "failed to create inference output: " << err.Message();
        outputs.back().emplace_back(output);
      }

      auto& expected = expected_outputs.back()["OUTPUT1"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] - input_1[i]);
      }
    }
  }

  std::vector<tc::InferResult*> results;
  std::condition_variable cv;
  std::mutex mu;
  err = this->client_->AsyncInferMulti(
      [&results, &cv, &mu](std::vector<tc::InferResult*> res) {
        {
          std::lock_guard<std::mutex> lk(mu);
          results.swap(res);
        }
        cv.notify_one();
      },
      options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [this, &results] { return !results.empty(); });
  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, AsyncInferMultiDifferentOptions)
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
    options.emplace_back(this->model_name_);
    // output will be different based on version
    // v1 : not swap
    // v2 : swap
    // v3 : swap
    size_t version = (i % 3) + 1;
    options.back().model_version_ = std::to_string(version);

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

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
      auto& expected =
          expected_outputs.back()[version == 1 ? "OUTPUT0" : "OUTPUT1"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] + input_1[i]);
      }
    }
    {
      auto& expected =
          expected_outputs.back()[version == 1 ? "OUTPUT1" : "OUTPUT0"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] - input_1[i]);
      }
    }
  }

  std::vector<tc::InferResult*> results;
  std::condition_variable cv;
  std::mutex mu;
  err = this->client_->AsyncInferMulti(
      [&results, &cv, &mu](std::vector<tc::InferResult*> res) {
        {
          std::lock_guard<std::mutex> lk(mu);
          results.swap(res);
        }
        cv.notify_one();
      },
      options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [this, &results] { return !results.empty(); });
  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, AsyncInferMultiOneOption)
{
  // Create only 1 sets of 'options'.
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  options.emplace_back(this->model_name_);
  // Not swap
  options.back().model_version_ = "1";
  for (size_t i = 0; i < 3; ++i) {
    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

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
  std::condition_variable cv;
  std::mutex mu;
  err = this->client_->AsyncInferMulti(
      [&results, &cv, &mu](std::vector<tc::InferResult*> res) {
        {
          std::lock_guard<std::mutex> lk(mu);
          results.swap(res);
        }
        cv.notify_one();
      },
      options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [this, &results] { return !results.empty(); });
  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, AsyncInferMultiOneOutput)
{
  // Create only 1 sets of 'outputs', but combine with different 'options'
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  for (size_t i = 0; i < 3; ++i) {
    options.emplace_back(this->model_name_);
    // output will be different based on version
    // v1 : not swap
    // v2 : swap
    // v3 : swap
    size_t version = (i % 3) + 1;
    options.back().model_version_ = std::to_string(version);

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

    tc::InferRequestedOutput* output;
    outputs.emplace_back();
    err = tc::InferRequestedOutput::Create(&output, "OUTPUT0");
    ASSERT_TRUE(err.IsOk())
        << "failed to create inference output: " << err.Message();
    outputs.back().emplace_back(output);

    expected_outputs.emplace_back();
    {
      auto& expected = expected_outputs.back()["OUTPUT0"];
      if (version == 1) {
        for (size_t i = 0; i < 16; ++i) {
          expected.emplace_back(input_0[i] + input_1[i]);
        }
      } else {
        for (size_t i = 0; i < 16; ++i) {
          expected.emplace_back(input_0[i] - input_1[i]);
        }
      }
    }
  }

  std::vector<tc::InferResult*> results;
  std::condition_variable cv;
  std::mutex mu;
  err = this->client_->AsyncInferMulti(
      [&results, &cv, &mu](std::vector<tc::InferResult*> res) {
        {
          std::lock_guard<std::mutex> lk(mu);
          results.swap(res);
        }
        cv.notify_one();
      },
      options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [this, &results] { return !results.empty(); });
  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, AsyncInferMultiNoOutput)
{
  // Not specifying 'outputs' at all, but combine with different 'options'
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  for (size_t i = 0; i < 3; ++i) {
    options.emplace_back(this->model_name_);
    // output will be different based on version
    // v1 : not swap
    // v2 : swap
    // v3 : swap
    size_t version = (i % 3) + 1;
    options.back().model_version_ = std::to_string(version);

    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

    expected_outputs.emplace_back();
    {
      auto& expected =
          expected_outputs.back()[version == 1 ? "OUTPUT0" : "OUTPUT1"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] + input_1[i]);
      }
    }
    {
      auto& expected =
          expected_outputs.back()[version == 1 ? "OUTPUT1" : "OUTPUT0"];
      for (size_t i = 0; i < 16; ++i) {
        expected.emplace_back(input_0[i] - input_1[i]);
      }
    }
  }

  std::vector<tc::InferResult*> results;
  std::condition_variable cv;
  std::mutex mu;
  err = this->client_->AsyncInferMulti(
      [&results, &cv, &mu](std::vector<tc::InferResult*> res) {
        {
          std::lock_guard<std::mutex> lk(mu);
          results.swap(res);
        }
        cv.notify_one();
      },
      options, inputs, outputs);
  ASSERT_TRUE(err.IsOk()) << "failed to perform multiple inferences: "
                          << err.Message();

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [this, &results] { return !results.empty(); });
  EXPECT_NO_FATAL_FAILURE(this->ValidateOutput(results, expected_outputs));
}

TYPED_TEST_P(ClientTest, AsyncInferMultiMismatchOptions)
{
  // Create mismatch number of 'options'.
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  options.emplace_back(this->model_name_);
  options.emplace_back(this->model_name_);
  for (size_t i = 0; i < 3; ++i) {
    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

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
  }

  std::vector<tc::InferResult*> results;
  std::condition_variable cv;
  std::mutex mu;
  err = this->client_->AsyncInferMulti(
      [&results, &cv, &mu](std::vector<tc::InferResult*> res) {
        {
          std::lock_guard<std::mutex> lk(mu);
          results.swap(res);
        }
        cv.notify_one();
      },
      options, inputs, outputs);
  ASSERT_FALSE(err.IsOk()) << "Expect AsyncInferMulti() to fail";
}

TYPED_TEST_P(ClientTest, AsyncInferMultiMismatchOutputs)
{
  // Create mismatch number of 'outputs'.
  tc::Error err = tc::Error::Success;
  std::vector<tc::InferOptions> options;
  std::vector<std::vector<tc::InferInput*>> inputs;
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs;

  std::vector<std::map<std::string, std::vector<int32_t>>> expected_outputs;
  for (size_t i = 0; i < 3; ++i) {
    options.emplace_back(this->model_name_);
    const auto& input_0 = this->input_data_[i % this->input_data_.size()];
    const auto& input_1 = this->input_data_[(i + 1) % this->input_data_.size()];
    inputs.emplace_back();
    err = this->PrepareInputs(input_0, input_1, &inputs.back());

    if (i != 2) {
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
    }
  }

  std::vector<tc::InferResult*> results;
  std::condition_variable cv;
  std::mutex mu;
  err = this->client_->AsyncInferMulti(
      [&results, &cv, &mu](std::vector<tc::InferResult*> res) {
        {
          std::lock_guard<std::mutex> lk(mu);
          results.swap(res);
        }
        cv.notify_one();
      },
      options, inputs, outputs);
  ASSERT_FALSE(err.IsOk()) << "Expect AsyncInferMulti() to fail";
}

REGISTER_TYPED_TEST_SUITE_P(
    ClientTest, InferMulti, InferMultiDifferentOutputs,
    InferMultiDifferentOptions, InferMultiOneOption, InferMultiOneOutput,
    InferMultiNoOutput, InferMultiMismatchOptions, InferMultiMismatchOutputs,
    AsyncInferMulti, AsyncInferMultiDifferentOutputs,
    AsyncInferMultiDifferentOptions, AsyncInferMultiOneOption,
    AsyncInferMultiOneOutput, AsyncInferMultiNoOutput,
    AsyncInferMultiMismatchOptions, AsyncInferMultiMismatchOutputs);

INSTANTIATE_TYPED_TEST_SUITE_P(GRPC, ClientTest, tc::InferenceServerGrpcClient);
INSTANTIATE_TYPED_TEST_SUITE_P(HTTP, ClientTest, tc::InferenceServerHttpClient);

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}