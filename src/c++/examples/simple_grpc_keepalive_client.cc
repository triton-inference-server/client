// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <getopt.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include "grpc_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<tc::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 16)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype),
      "unable to get datatype for '" + name + "'");
  // Validate datatype
  if (datatype.compare("INT32") != 0) {
    std::cerr << "error: received incorrect datatype for '" << name
              << "': " << datatype << std::endl;
    exit(1);
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << "\t--grpc-keepalive-time <milliseconds>" << std::endl;
  std::cerr << "\t--grpc-keepalive-timeout <milliseconds>" << std::endl;
  std::cerr << "\t--grpc-keepalive-permit-without-calls" << std::endl;
  std::cerr << "\t--grpc-http2-max-pings-without-data <number of pings>"
            << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8001");
  tc::Headers http_headers;
  uint32_t client_timeout = 0;
  bool use_ssl = false;
  tc::SslOptions ssl_options;
  tc::KeepAliveOptions keepalive_options;
  // GRPC KeepAlive: https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
  int keepalive_time_ms = INT_MAX;
  int keepalive_timeout_ms = 20000;
  bool keepalive_permit_without_calls = false;
  int http2_max_pings_without_data = 2;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"grpc-keepalive-time", 1, 0, 0},
      {"grpc-keepalive-timeout", 1, 0, 1},
      {"grpc-keepalive-permit-without-calls", 0, 0, 2},
      {"grpc-http2-max-pings-without-data", 1, 0, 3}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(argc, argv, "vu:t:H:C:", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 0:
        keepalive_options.keepalive_time_ms = std::stoi(optarg);
        break;
      case 1:
        keepalive_options.keepalive_timeout_ms = std::stoi(optarg);
        break;
      case 2:
        keepalive_options.keepalive_permit_without_calls = true;
        break;
      case 3:
        keepalive_options.http2_max_pings_without_data = std::stoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";
  std::string model_version = "";

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  FAIL_IF_ERR(
      tc::InferenceServerGrpcClient::Create(
          &client, url, verbose, use_ssl, ssl_options, keepalive_options),
      "unable to create grpc client");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<int32_t> input0_data(16);
  std::vector<int32_t> input1_data(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = i;
    input1_data[i] = 1;
  }

  std::vector<int64_t> shape{1, 16};

  // Initialize the inputs with the data.
  tc::InferInput* input0;
  tc::InferInput* input1;

  FAIL_IF_ERR(
      tc::InferInput::Create(&input0, "INPUT0", shape, "INT32"),
      "unable to get INPUT0");
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);
  FAIL_IF_ERR(
      tc::InferInput::Create(&input1, "INPUT1", shape, "INT32"),
      "unable to get INPUT1");
  std::shared_ptr<tc::InferInput> input1_ptr;
  input1_ptr.reset(input1);

  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");
  FAIL_IF_ERR(
      input1_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input1_data[0]),
          input1_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT1");

  // Generate the outputs to be requested.
  tc::InferRequestedOutput* output0;
  tc::InferRequestedOutput* output1;

  FAIL_IF_ERR(
      tc::InferRequestedOutput::Create(&output0, "OUTPUT0"),
      "unable to get 'OUTPUT0'");
  std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
  output0_ptr.reset(output0);
  FAIL_IF_ERR(
      tc::InferRequestedOutput::Create(&output1, "OUTPUT1"),
      "unable to get 'OUTPUT1'");
  std::shared_ptr<tc::InferRequestedOutput> output1_ptr;
  output1_ptr.reset(output1);


  // The inference settings. Will be using default for now.
  tc::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;

  std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
  std::vector<const tc::InferRequestedOutput*> outputs = {output0_ptr.get(),
                                                          output1_ptr.get()};

  tc::InferResult* results;
  FAIL_IF_ERR(
      client->Infer(&results, options, inputs, outputs, http_headers),
      "unable to run model");
  std::shared_ptr<tc::InferResult> results_ptr;
  results_ptr.reset(results);

  // Validate the results...
  ValidateShapeAndDatatype("OUTPUT0", results_ptr);
  ValidateShapeAndDatatype("OUTPUT1", results_ptr);

  // Get pointers to the result returned...
  int32_t* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(
          "OUTPUT0", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get result data for 'OUTPUT0'");
  if (output0_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
              << output0_byte_size << std::endl;
    exit(1);
  }

  int32_t* output1_data;
  size_t output1_byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(
          "OUTPUT1", (const uint8_t**)&output1_data, &output1_byte_size),
      "unable to get result data for 'OUTPUT1'");
  if (output1_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT1': "
              << output1_byte_size << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input1_data[i] << " = "
              << *(output0_data + i) << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              << *(output1_data + i) << std::endl;

    if ((input0_data[i] + input1_data[i]) != *(output0_data + i)) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_data[i] - input1_data[i]) != *(output1_data + i)) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  // Get full response
  std::cout << results_ptr->DebugString() << std::endl;
  std::cout << "PASS : KeepAlive" << std::endl;

  return 0;
}
