// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gmock/gmock.h"
#include "grpc_client.h"
#include "gtest/gtest.h"
#include "http_client.h"
#include "shm_utils.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  do {                                                             \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  } while (false)

#define FAIL_IF_SUCCESS(X, MSG, ERR_MSG)                       \
  do {                                                         \
    tc::Error err = (X);                                       \
    ASSERT_FALSE(err.IsOk()) << "error: " << (MSG) << ": ";    \
    ASSERT_THAT(err.Message(), ::testing::HasSubstr(ERR_MSG)); \
  } while (false)

namespace {

template <typename ClientType>
class ClientInputTest : public ::testing::Test {
 public:
  ClientInputTest() : shape_{1, 16} {}

  void SetUp() override
  {
    std::string url;
    std::string client_type;
    if (std::is_same<ClientType, tc::InferenceServerGrpcClient>::value) {
      url = "localhost:8001";
      client_type = "GRPC";
    } else if (std::is_same<ClientType, tc::InferenceServerHttpClient>::value) {
      url = "localhost:8000";
      client_type = "HTTP";
    } else {
      ASSERT_TRUE(false) << "Unrecognized client class type '"
                         << typeid(ClientType).name() << "'";
    }
    auto err = ClientType::Create(&this->client_, url);
    ASSERT_TRUE(err.IsOk())
        << "failed to create " << client_type << " client: " << err.Message();

    // Initialize vector input_data_
    for (size_t i = 0; i < 16; ++i) {
      this->input_data_.emplace_back(i);
    }
  }

  std::unique_ptr<ClientType> client_;
  std::vector<int32_t> input_data_;
  std::vector<int64_t> shape_;
};

TYPED_TEST_SUITE_P(ClientInputTest);

TYPED_TEST_P(ClientInputTest, AppendRaw)
{
  // Initialize the inputs with the data.
  tc::InferInput* input0;
  tc::InferInput* input1;

  FAIL_IF_ERR(
      tc::InferInput::Create(&input0, "INPUT0", this->shape_, "INT32"),
      "unable to get INPUT0");
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);
  FAIL_IF_ERR(
      tc::InferInput::Create(&input1, "INPUT1", this->shape_, "INT32"),
      "unable to get INPUT1");
  std::shared_ptr<tc::InferInput> input1_ptr;
  input1_ptr.reset(input1);

  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&(this->input_data_[0])),
          this->input_data_.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");
  FAIL_IF_ERR(
      input1_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&(this->input_data_[0])),
          this->input_data_.size() * sizeof(int32_t)),
      "unable to set data for INPUT1");

  // The inference settings. Will be using default for now.
  tc::InferOptions options("simple");
  options.model_version_ = "";

  std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
  tc::InferResult* results;

  // Test 1
  inputs[1]->SetShape({1, 15});
  FAIL_IF_SUCCESS(
      this->client_->Infer(&results, options, inputs),
      "expect error with inference request",
      "input 'INPUT1' got unexpected byte size 64, expected 60");

  // Test 2
  inputs[0]->SetShape({2, 8});
  inputs[1]->SetShape({2, 8});
  // Assert the request reaches the server
  FAIL_IF_SUCCESS(
      this->client_->Infer(&results, options, inputs),
      "expect error with inference request",
      "unexpected shape for input 'INPUT1' for model 'simple'. Expected "
      "[-1,16], got [2,8]");
}

TYPED_TEST_P(ClientInputTest, SetSharedMemory)
{
  // Unregistering all shared memory regions for a clean
  // start.
  FAIL_IF_ERR(
      this->client_->UnregisterSystemSharedMemory(),
      "unable to unregister all system shared memory regions");
  FAIL_IF_ERR(
      this->client_->UnregisterCudaSharedMemory(),
      "unable to unregister all cuda shared memory regions");

  // Initialize the inputs with the data.
  tc::InferInput* input0;
  tc::InferInput* input1;
  size_t input_byte_size = 64;

  FAIL_IF_ERR(
      tc::InferInput::Create(&input0, "INPUT0", this->shape_, "INT32"),
      "unable to get INPUT0");
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);
  FAIL_IF_ERR(
      tc::InferInput::Create(&input1, "INPUT1", this->shape_, "INT32"),
      "unable to get INPUT1");
  std::shared_ptr<tc::InferInput> input1_ptr;
  input1_ptr.reset(input1);

  // Create Input0 and Input1 in Shared Memory. Initialize Input0 to unique
  // integers and Input1 to all ones.
  std::string shm_key = "/input_simple";
  int shm_fd_ip, *input0_shm;
  FAIL_IF_ERR(
      tc::CreateSharedMemoryRegion(shm_key, input_byte_size * 2, &shm_fd_ip),
      "");
  FAIL_IF_ERR(
      tc::MapSharedMemory(
          shm_fd_ip, 0, input_byte_size * 2, (void**)&input0_shm),
      "");
  FAIL_IF_ERR(tc::CloseSharedMemory(shm_fd_ip), "");
  int* input1_shm = (int*)(input0_shm + 16);
  for (size_t i = 0; i < 16; ++i) {
    *(input0_shm + i) = i;
    *(input1_shm + i) = 1;
  }

  FAIL_IF_ERR(
      this->client_->RegisterSystemSharedMemory(
          "input_data", shm_key, input_byte_size * 2),
      "failed to register input shared memory region");

  FAIL_IF_ERR(
      input0_ptr->SetSharedMemory(
          "input_data", input_byte_size, 0 /* offset */),
      "unable to set shared memory for INPUT0");
  FAIL_IF_ERR(
      input1_ptr->SetSharedMemory(
          "input_data", input_byte_size, input_byte_size /* offset */),
      "unable to set shared memory for INPUT1");

  // The inference settings. Will be using default for now.
  tc::InferOptions options("simple");
  options.model_version_ = "";

  std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
  tc::InferResult* results;

  // Test 1
  inputs[1]->SetShape({1, 15});
  FAIL_IF_SUCCESS(
      this->client_->Infer(&results, options, inputs),
      "expect error with inference request",
      ("input 'INPUT1' got unexpected byte size " +
       std::to_string(input_byte_size) + ", expected " +
       std::to_string(input_byte_size - sizeof(int))));

  // Test 2
  inputs[0]->SetShape({2, 8});
  inputs[1]->SetShape({2, 8});
  // Assert the request reaches the server
  FAIL_IF_SUCCESS(
      this->client_->Infer(&results, options, inputs),
      "expect error with inference request",
      "unexpected shape for input 'INPUT1' for model 'simple'. Expected "
      "[-1,16], got [2,8]");

  // Get shared memory regions active/registered within triton
  using ClientType = TypeParam;
  if constexpr (std::is_same<
                    ClientType, tc::InferenceServerGrpcClient>::value) {
    inference::SystemSharedMemoryStatusResponse shm_status;
    FAIL_IF_ERR(
        this->client_->SystemSharedMemoryStatus(&shm_status),
        "failed to get shared memory status");
    std::cout << "Shared Memory Status:\n" << shm_status.DebugString() << "\n";
  } else {
    std::string shm_status;
    FAIL_IF_ERR(
        this->client_->SystemSharedMemoryStatus(&shm_status),
        "failed to get shared memory status");
    std::cout << "Shared Memory Status:\n" << shm_status << "\n";
  }

  // Unregister shared memory
  FAIL_IF_ERR(
      this->client_->UnregisterSystemSharedMemory("input_data"),
      "unable to unregister shared memory input region");

  // Cleanup shared memory
  FAIL_IF_ERR(tc::UnmapSharedMemory(input0_shm, input_byte_size * 2), "");
  FAIL_IF_ERR(tc::UnlinkSharedMemoryRegion("/input_simple"), "");
}

REGISTER_TYPED_TEST_SUITE_P(ClientInputTest, AppendRaw, SetSharedMemory);

INSTANTIATE_TYPED_TEST_SUITE_P(
    GRPC, ClientInputTest, tc::InferenceServerGrpcClient);
INSTANTIATE_TYPED_TEST_SUITE_P(
    HTTP, ClientInputTest, tc::InferenceServerHttpClient);

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
