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

#include <getopt.h>
#include <unistd.h>
#include <algorithm>
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

// Create a InferenceServerGrpcClient instance
std::unique_ptr<tc::InferenceServerGrpcClient> client;

namespace {

//  Helper function to make sure the trace setting is properly initialized /
//  reset before actually running the test case.
void
CheckServerInitialState(const std::string& model_name)
{
  std::string initial_settings =
      "settings{key:\"log_frequency\"value{value:\"0\"}}settings{key:\"trace_"
      "count\"value{value:\"-1\"}}settings{key:\"trace_file\"value{value:"
      "\"global_unittest.log\"}}settings{key:\"trace_level\"value{value:"
      "\"TIMESTAMPS\"}}settings{key:\"trace_rate\"value{value:\"1\"}}";


  inference::TraceSettingResponse trace_settings;
  FAIL_IF_ERR(
      client->GetTraceSettings(&trace_settings, model_name),
      "unable to get trace settings");
  std::string str = trace_settings.DebugString();
  str.erase(remove(str.begin(), str.end(), ' '), str.end());
  str.erase(remove(str.begin(), str.end(), '\n'), str.end());
  if (str.compare(initial_settings) != 0) {
    std::cerr << "error: trace settings is not properly initialized for model'"
              << model_name << "'" << std::endl;
    exit(1);
  }

  FAIL_IF_ERR(
      client->GetTraceSettings(&trace_settings, ""),
      "unable to get trace settings");
  str = trace_settings.DebugString();
  str.erase(remove(str.begin(), str.end(), ' '), str.end());
  str.erase(remove(str.begin(), str.end(), '\n'), str.end());
  if (str.compare(initial_settings) != 0) {
    std::cerr << "error: default trace settings is not properly initialized'"
              << std::endl;
    exit(1);
  }
}

// Clear all the trace settings to initial state.
void
TearDown(const std::string model_name)
{
  std::map<std::string, std::vector<std::string>> clear_settings = {
      {"trace_file", {}},
      {"trace_level", {}},
      {"trace_rate", {}},
      {"trace_count", {}},
      {"log_frequency", {}}};

  FAIL_IF_ERR(
      client->UpdateTraceSettings(model_name, clear_settings),
      "unable to update trace settings");
  FAIL_IF_ERR(
      client->UpdateTraceSettings("", clear_settings),
      "unable to update trace settings");
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
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

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:H:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
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
      case '?':
        Usage(argv);
        break;
    }
  }

  std::string model_name = "simple";
  inference::TraceSettingResponse trace_settings;

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  FAIL_IF_ERR(
      tc::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create grpc client");

  {
    CheckServerInitialState(model_name);
    // Model trace settings will be the same as global trace settings since no
    // update has been made.
    std::string initial_settings =
        "settings{key:\"log_frequency\"value{value:\"0\"}}settings{key:\"trace_"
        "count\"value{value:\"-1\"}}settings{key:\"trace_file\"value{value:"
        "\"global_unittest.log\"}}settings{key:\"trace_level\"value{value:"
        "\"TIMESTAMPS\"}}settings{key:\"trace_rate\"value{value:\"1\"}}";

    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings, model_name),
        "unable to get trace settings");
    std::string str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(initial_settings) != 0) {
      std::cerr
          << "error: trace settings is not properly initialized for model'"
          << model_name << "'" << std::endl;
      exit(1);
    }

    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings, ""),
        "unable to get trace settings");
    str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(initial_settings) != 0) {
      std::cerr << "error: default trace settings is not properly initialized'"
                << std::endl;
      exit(1);
    }
  }

  {
    // Update model and global trace settings in order, and expect the global
    // trace settings will only reflect to the model setting fields that haven't
    // been specified.
    TearDown(model_name);
    CheckServerInitialState(model_name);

    std::string expected_first_model_settings =
        "settings{key:\"log_frequency\"value{value:\"0\"}}settings{key:\"trace_"
        "count\"value{value:\"-1\"}}settings{key:\"trace_file\"value{value:"
        "\"model.log\"}}settings{key:\"trace_level\"value{value:"
        "\"TIMESTAMPS\"}}settings{key:\"trace_rate\"value{value:\"1\"}}";
    std::string expected_second_model_settings =
        "settings{key:\"log_frequency\"value{value:\"0\"}}settings{key:\"trace_"
        "count\"value{value:\"-1\"}}settings{key:\"trace_file\"value{value:"
        "\"model.log\"}}settings{key:\"trace_level\"value{value:"
        "\"TIMESTAMPS\"value:\"TENSORS\"}}settings{key:\"trace_rate\"value{"
        "value:"
        "\"1\"}}";
    std::string expected_global_settings =
        "settings{key:\"log_frequency\"value{value:\"0\"}}settings{key:\"trace_"
        "count\"value{value:\"-1\"}}settings{key:\"trace_file\"value{value:"
        "\"another.log\"}}settings{key:\"trace_level\"value{value:"
        "\"TIMESTAMPS\"value:\"TENSORS\"}}settings{key:\"trace_rate\"value{"
        "value:"
        "\"1\"}}";

    std::map<std::string, std::vector<std::string>> model_update_settings = {
        {"trace_file", {"model.log"}}};
    std::map<std::string, std::vector<std::string>> global_update_settings = {
        {"trace_file", {"another.log"}},
        {"trace_level", {"TIMESTAMPS", "TENSORS"}}};

    FAIL_IF_ERR(
        client->UpdateTraceSettings(model_name, model_update_settings),
        "unable to update trace settings");
    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings, model_name),
        "unable to get trace settings");
    std::string str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(expected_first_model_settings) != 0) {
      std::cerr << "error: Unexpected updated model trace settings"
                << std::endl;
      exit(1);
    }
    // Note that 'trace_level' may be mismatch due to the order of the levels
    // listed, currently we assume the order is the same for simplicity. But the
    // order shouldn't be enforced and this checking needs to be improved when
    // this kind of failure is reported
    FAIL_IF_ERR(
        client->UpdateTraceSettings("", global_update_settings),
        "unable to update trace settings");
    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings),
        "unable to get trace settings");
    str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(expected_global_settings) != 0) {
      std::cerr << "error: Unexpected updated global trace settings"
                << std::endl;
      exit(1);
    }

    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings, model_name),
        "unable to get trace settings");
    str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(expected_second_model_settings) != 0) {
      std::cerr << "error: Unexpected model trace settings after global update "
                << std::endl;
      exit(1);
    }
  }

  {
    // Clear global and model trace settings in order, and expect the default /
    // global trace settings are propagated properly.
    TearDown(model_name);
    CheckServerInitialState(model_name);

    // First set up the model / global trace setting that: model 'simple' has
    // 'trace_rate' and 'log_frequency' specified global has 'trace_level',
    // 'trace_count' and 'trace_rate' specified
    std::map<std::string, std::vector<std::string>> model_update_settings = {
        {"trace_rate", {"12"}}, {"log_frequency", {"34"}}};
    std::map<std::string, std::vector<std::string>> global_update_settings = {
        {"trace_rate", {"56"}},
        {"trace_count", {"78"}},
        {"trace_level", {"OFF"}}};
    FAIL_IF_ERR(
        client->UpdateTraceSettings(model_name, model_update_settings),
        "unable to update trace settings");
    FAIL_IF_ERR(
        client->UpdateTraceSettings("", global_update_settings),
        "unable to update trace settings");

    std::string expected_global_settings =
        "settings{key:\"log_frequency\"value{value:\"0\"}}settings{key:\"trace_"
        "count\"value{value:\"-1\"}}settings{key:\"trace_file\"value{value:"
        "\"global_unittest.log\"}}settings{key:\"trace_level\"value{value:"
        "\"OFF\"}}settings{key:\"trace_rate\"value{value:\"1\"}}";
    std::string expected_first_model_settings =
        "settings{key:\"log_frequency\"value{value:\"34\"}}settings{key:"
        "\"trace_"
        "count\"value{value:\"-1\"}}settings{key:\"trace_file\"value{value:"
        "\"global_unittest.log\"}}settings{key:\"trace_level\"value{value:"
        "\"OFF\"}}settings{key:\"trace_rate\"value{value:\"12\"}}";
    std::string expected_second_model_settings =
        "settings{key:\"log_frequency\"value{value:\"34\"}}settings{key:"
        "\"trace_"
        "count\"value{value:\"-1\"}}settings{key:\"trace_file\"value{value:"
        "\"global_unittest.log\"}}settings{key:\"trace_level\"value{value:"
        "\"OFF\"}}settings{key:\"trace_rate\"value{value:\"1\"}}";
    std::map<std::string, std::vector<std::string>> global_clear_settings = {
        {"trace_rate", {}}, {"trace_count", {}}};
    std::map<std::string, std::vector<std::string>> model_clear_settings = {
        {"trace_rate", {}}, {"trace_level", {}}};

    // Clear global
    FAIL_IF_ERR(
        client->UpdateTraceSettings("", global_clear_settings),
        "unable to update trace settings");
    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings),
        "unable to get trace settings");
    std::string str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(expected_global_settings) != 0) {
      std::cerr << "error: Unexpected updated global trace settings"
                << std::endl;
      exit(1);
    }
    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings, model_name),
        "unable to get trace settings");
    str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(expected_first_model_settings) != 0) {
      std::cerr << "error: Unexpected model trace settings after global clear"
                << std::endl;
      exit(1);
    }
    // Clear model
    FAIL_IF_ERR(
        client->UpdateTraceSettings(model_name, model_clear_settings),
        "unable to update trace settings");
    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings, model_name),
        "unable to get trace settings");
    str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(expected_second_model_settings) != 0) {
      std::cerr << "error: Unexpected model trace settings after model clear"
                << std::endl;
      exit(1);
    }
    FAIL_IF_ERR(
        client->GetTraceSettings(&trace_settings),
        "unable to get trace settings");
    str = trace_settings.DebugString();
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    if (str.compare(expected_global_settings) != 0) {
      std::cerr << "error: Unexpected global trace settings after model clear"
                << std::endl;
      exit(1);
    }
  }

  std::cout << "PASS : GRPC_Trace" << std::endl;

  return 0;
}
