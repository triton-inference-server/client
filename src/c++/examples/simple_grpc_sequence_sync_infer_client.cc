// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <unistd.h>
#include <condition_variable>
#include <iostream>
#include <string>
#include <vector>
#include "grpc_client.h"

namespace tc = triton::client;

using ResultList = std::vector<std::shared_ptr<tc::InferResult>>;

// Global mutex to synchronize the threads
std::mutex mutex_;
std::condition_variable cv_;

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
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service and its gRPC port>"
            << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;
  std::cerr << "\t-o <offset for sequence ID>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "For -o, the client will use sequence ID <1 + 2 * offset> "
            << "and <2 + 2 * offset>. Default offset is 0." << std::endl;

  exit(1);
}

void
SyncSend(
    const std::unique_ptr<tc::InferenceServerGrpcClient>& client,
    tc::InferOptions& options, int32_t value, std::vector<int32_t>& result_data,
    tc::Headers& http_headers)
{
  // Initialize the inputs with the data.
  tc::InferInput* input;
  std::vector<int64_t> shape{1, 1};
  FAIL_IF_ERR(
      tc::InferInput::Create(&input, "INPUT", shape, "INT32"),
      "unable to create 'INPUT'");
  std::shared_ptr<tc::InferInput> ivalue(input);
  FAIL_IF_ERR(ivalue->Reset(), "unable to reset 'INPUT'");
  FAIL_IF_ERR(
      ivalue->AppendRaw(reinterpret_cast<uint8_t*>(&value), sizeof(int32_t)),
      "unable to set data for 'INPUT'");

  std::vector<tc::InferInput*> inputs = {ivalue.get()};

  tc::InferRequestedOutput* output;
  FAIL_IF_ERR(
      tc::InferRequestedOutput::Create(&output, "OUTPUT"),
      "unable to get 'OUTPUT'");
  std::shared_ptr<const tc::InferRequestedOutput> routput;
  routput.reset(output);

  std::vector<const tc::InferRequestedOutput*> outputs = {routput.get()};

  tc::InferResult* result;
  // Send inference request to the inference server.
  FAIL_IF_ERR(
      client->Infer(&result, options, inputs, outputs, http_headers),
      "unable to run model");
  std::shared_ptr<tc::InferResult> this_result(result);

  // Get pointers to the result returned...
  int32_t* output_data;
  size_t output_byte_size;
  FAIL_IF_ERR(
      this_result->RawData(
          "OUTPUT", (const uint8_t**)&output_data, &output_byte_size),
      "unable to get result data for 'OUTPUT'");
  if (output_byte_size != 4) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT': "
              << output_byte_size << std::endl;
    exit(1);
  }

  result_data.push_back(*output_data);
}

void
SyncSend(
    const std::unique_ptr<tc::InferenceServerGrpcClient>& client,
    const std::string& model_name, int32_t value, const uint64_t sequence_id,
    bool start_of_sequence, bool end_of_sequence,
    std::vector<int32_t>& result_data, tc::Headers& http_headers)
{
  tc::InferOptions options(model_name);
  options.sequence_id_ = sequence_id;
  options.sequence_start_ = start_of_sequence;
  options.sequence_end_ = end_of_sequence;

  SyncSend(client, options, value, result_data, http_headers);
}

void
SyncSend(
    const std::unique_ptr<tc::InferenceServerGrpcClient>& client,
    const std::string& model_name, int32_t value,
    const std::string& sequence_id, bool start_of_sequence,
    bool end_of_sequence, std::vector<int32_t>& result_data,
    tc::Headers& http_headers)
{
  tc::InferOptions options(model_name);
  options.sequence_id_str_ = sequence_id;
  options.sequence_start_ = start_of_sequence;
  options.sequence_end_ = end_of_sequence;

  SyncSend(client, options, value, result_data, http_headers);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool dyna_sequence = false;
  std::string url("localhost:8001");
  int sequence_id_offset = 0;
  tc::Headers http_headers;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vdu:H:o:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'd':
        dyna_sequence = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case 'o':
        sequence_id_offset = std::stoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  tc::Error err;

  // We use the custom "sequence" model which takes 1 input value. The
  // output is the accumulated value of the inputs. See
  // src/custom/sequence.
  std::string int_model_name =
      dyna_sequence ? "simple_dyna_sequence" : "simple_sequence";
  std::string string_model_name =
      dyna_sequence ? "simple_string_dyna_sequence" : "simple_sequence";

  const uint64_t int_sequence_id0 = 1 + sequence_id_offset * 2;
  const uint64_t int_sequence_id1 = 2 + sequence_id_offset * 2;

  // For string sequence IDs, the dyna backend requires that the
  // sequence id be decodable into an integer, otherwise we'll use
  // a test string sequence id and a model that doesn't require corrid
  // control.
  const std::string string_sequence_id0 =
      dyna_sequence ? std::to_string(3 + sequence_id_offset * 2) : "SEQ-3";

  std::cout << "sequence ID " << int_sequence_id0 << " : "
            << "sequence ID " << int_sequence_id1 << " : "
            << "sequence ID " << string_sequence_id0 << std::endl;

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  FAIL_IF_ERR(
      tc::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create grpc client");

  // Now send the inference sequences..
  //
  std::vector<int32_t> values{11, 7, 5, 3, 2, 0, 1};
  std::vector<int32_t> int_result0_data;
  std::vector<int32_t> int_result1_data;
  std::vector<int32_t> string_result0_data;

  // Send requests, first reset accumulator for the sequence.
  SyncSend(
      client, int_model_name, 0, int_sequence_id0, true /* start-of-sequence */,
      false /* end-of-sequence */, int_result0_data, http_headers);
  SyncSend(
      client, int_model_name, 100, int_sequence_id1,
      true /* start-of-sequence */, false /* end-of-sequence */,
      int_result1_data, http_headers);
  SyncSend(
      client, string_model_name, 20, string_sequence_id0,
      true /* start-of-sequence */, false /* end-of-sequence */,
      string_result0_data, http_headers);

  // Now send a sequence of values...
  for (int32_t v : values) {
    SyncSend(
        client, int_model_name, v, int_sequence_id0,
        false /* start-of-sequence */, (v == 1) /* end-of-sequence */,
        int_result0_data, http_headers);
    SyncSend(
        client, int_model_name, -v, int_sequence_id1,
        false /* start-of-sequence */, (v == 1) /* end-of-sequence */,
        int_result1_data, http_headers);
    SyncSend(
        client, string_model_name, -v, string_sequence_id0,
        false /* start-of-sequence */, (v == 1) /* end-of-sequence */,
        string_result0_data, http_headers);
  }

  for (size_t i = 0; i < int_result0_data.size(); i++) {
    int32_t int_seq0_expected = (i == 0) ? 1 : values[i - 1];
    int32_t int_seq1_expected = (i == 0) ? 101 : values[i - 1] * -1;
    int32_t string_seq0_expected;

    // For string sequence ID case we are testing two different backends
    if ((i == 0) && dyna_sequence) {
      string_seq0_expected = 20;
    } else if ((i == 0) && !dyna_sequence) {
      string_seq0_expected = 21;
    } else if ((i != 0) && dyna_sequence) {
      string_seq0_expected = values[i - 1] * -1 + string_result0_data[i - 1];
    } else {
      string_seq0_expected = values[i - 1] * -1;
    }

    // The dyna_sequence custom backend adds the sequence ID to
    // the last request in a sequence.
    if (dyna_sequence && (i != 0) && (values[i - 1] == 1)) {
      int_seq0_expected += int_sequence_id0;
      int_seq1_expected += int_sequence_id1;
      string_seq0_expected += std::stoi(string_sequence_id0);
    }

    std::cout << "[" << i << "] " << int_result0_data[i] << " : "
              << int_result1_data[i] << " : " << string_result0_data[i]
              << std::endl;

    if ((int_seq0_expected != int_result0_data[i]) ||
        (int_seq1_expected != int_result1_data[i]) ||
        (string_seq0_expected != string_result0_data[i])) {
      std::cout << "[ expected ] " << int_seq0_expected << " : "
                << int_seq1_expected << " : " << string_seq0_expected
                << std::endl;
      return 1;
    }
  }

  return 0;
}
