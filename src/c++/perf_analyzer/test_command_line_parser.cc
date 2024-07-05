// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//
#include <getopt.h>

#include <array>

#include "command_line_parser.h"
#include "doctest.h"
#include "perf_analyzer_exception.h"

namespace triton { namespace perfanalyzer {

inline void
CHECK_STRING(const char* name, const std::string& str, const std::string& val)
{
  CHECK_MESSAGE(
      !str.compare(val), name, " expecting '", val, "', found '", str, "'");
}

inline void
CHECK_STRING(std::string act, std::string exp)
{
  CHECK_MESSAGE(
      !act.compare(exp), "Expecting: '", exp, "', Found: '", act, "'");
}

std::string
CreateUsageMessage(const std::string& option_name, const std::string& msg)
{
  return "Failed to parse " + option_name + ". " + msg;
}

// Performs a doc test check against all the individual parameters
// in a PAParams object.
//
// /param act actual object under test
// /param exp expected value for object
//
inline void
CHECK_PARAMS(PAParamsPtr act, PAParamsPtr exp)
{
  CHECK(act->verbose == exp->verbose);
  CHECK(act->streaming == exp->streaming);
  CHECK(act->extra_verbose == exp->extra_verbose);
  CHECK(act->max_threads == exp->max_threads);
  CHECK(act->max_threads_specified == exp->max_threads_specified);
  CHECK(act->sequence_length == exp->sequence_length);
  CHECK(act->percentile == exp->percentile);
  REQUIRE(act->user_data.size() == exp->user_data.size());
  for (size_t i = 0; i < act->user_data.size(); i++) {
    CHECK_STRING(act->user_data[i], exp->user_data[i]);
  }
  CHECK(act->input_shapes.size() == exp->input_shapes.size());
  for (auto act_shape : act->input_shapes) {
    auto exp_shape = exp->input_shapes.find(act_shape.first);
    REQUIRE_MESSAGE(
        exp_shape != exp->input_shapes.end(),
        "Unexpected input_shape: ", act_shape.first);
    REQUIRE(act_shape.second.size() == exp_shape->second.size());
    for (size_t i = 0; i < act_shape.second.size(); i++) {
      CHECK_MESSAGE(
          act_shape.second[i] == exp_shape->second[i],
          "Unexpected shape value for: ", act_shape.first, "[", i, "]");
    }
  }
  CHECK(act->measurement_window_ms == exp->measurement_window_ms);
  CHECK(act->using_concurrency_range == exp->using_concurrency_range);
  CHECK(act->concurrency_range.start == exp->concurrency_range.start);
  CHECK(act->concurrency_range.end == exp->concurrency_range.end);
  CHECK(act->concurrency_range.step == exp->concurrency_range.step);
  CHECK(act->latency_threshold_ms == exp->latency_threshold_ms);
  CHECK(act->stability_threshold == doctest::Approx(act->stability_threshold));
  CHECK(act->max_trials == exp->max_trials);
  CHECK(act->zero_input == exp->zero_input);
  CHECK(act->string_length == exp->string_length);
  CHECK_STRING(act->string_data, exp->string_data);
  CHECK(act->async == exp->async);
  CHECK(act->forced_sync == exp->forced_sync);
  CHECK(act->using_request_rate_range == exp->using_request_rate_range);
  CHECK(
      act->request_rate_range[0] ==
      doctest::Approx(exp->request_rate_range[0]));
  CHECK(
      act->request_rate_range[1] ==
      doctest::Approx(exp->request_rate_range[1]));
  CHECK(
      act->request_rate_range[2] ==
      doctest::Approx(exp->request_rate_range[2]));
  CHECK(act->num_of_sequences == exp->num_of_sequences);
  CHECK(act->search_mode == exp->search_mode);
  CHECK(act->request_distribution == exp->request_distribution);
  CHECK(act->using_custom_intervals == exp->using_custom_intervals);
  CHECK_STRING(act->request_intervals_file, exp->request_intervals_file);
  CHECK(act->shared_memory_type == exp->shared_memory_type);
  CHECK(act->output_shm_size == exp->output_shm_size);
  CHECK(act->kind == exp->kind);
  CHECK_STRING(act->model_signature_name, exp->model_signature_name);
  CHECK(act->using_grpc_compression == exp->using_grpc_compression);
  CHECK(act->compression_algorithm == exp->compression_algorithm);
  CHECK(act->measurement_mode == exp->measurement_mode);
  CHECK(act->measurement_request_count == exp->measurement_request_count);
  CHECK_STRING(act->triton_server_path, exp->triton_server_path);
  CHECK_STRING(act->model_repository_path, exp->model_repository_path);
  CHECK(act->start_sequence_id == exp->start_sequence_id);
  CHECK(act->sequence_id_range == exp->sequence_id_range);
  CHECK_STRING(
      act->ssl_options.ssl_grpc_certificate_chain_file,
      exp->ssl_options.ssl_grpc_certificate_chain_file);
  CHECK_STRING(
      act->ssl_options.ssl_grpc_private_key_file,
      exp->ssl_options.ssl_grpc_private_key_file);
  CHECK_STRING(
      act->ssl_options.ssl_grpc_root_certifications_file,
      exp->ssl_options.ssl_grpc_root_certifications_file);
  CHECK(act->ssl_options.ssl_grpc_use_ssl == exp->ssl_options.ssl_grpc_use_ssl);
  CHECK_STRING(
      act->ssl_options.ssl_https_ca_certificates_file,
      exp->ssl_options.ssl_https_ca_certificates_file);
  CHECK_STRING(
      act->ssl_options.ssl_https_client_certificate_file,
      exp->ssl_options.ssl_https_client_certificate_file);
  CHECK_STRING(
      act->ssl_options.ssl_https_client_certificate_type,
      exp->ssl_options.ssl_https_client_certificate_type);
  CHECK_STRING(
      act->ssl_options.ssl_https_private_key_file,
      exp->ssl_options.ssl_https_private_key_file);
  CHECK_STRING(
      act->ssl_options.ssl_https_private_key_type,
      exp->ssl_options.ssl_https_private_key_type);
  CHECK(
      act->ssl_options.ssl_https_verify_host ==
      exp->ssl_options.ssl_https_verify_host);
  CHECK(
      act->ssl_options.ssl_https_verify_peer ==
      exp->ssl_options.ssl_https_verify_peer);
  CHECK(act->verbose_csv == exp->verbose_csv);
  CHECK(act->enable_mpi == exp->enable_mpi);
  CHECK(act->trace_options.size() == exp->trace_options.size());
  CHECK(act->using_old_options == exp->using_old_options);
  CHECK(act->dynamic_concurrency_mode == exp->dynamic_concurrency_mode);
  CHECK(act->url_specified == exp->url_specified);
  CHECK_STRING(act->url, exp->url);
  CHECK_STRING(act->model_name, exp->model_name);
  CHECK_STRING(act->model_version, exp->model_version);
  CHECK(act->batch_size == exp->batch_size);
  CHECK(act->using_batch_size == exp->using_batch_size);
  CHECK(act->concurrent_request_count == exp->concurrent_request_count);
  CHECK(act->protocol == exp->protocol);
  CHECK(act->http_headers->size() == exp->http_headers->size());
  CHECK(act->max_concurrency == exp->max_concurrency);
  CHECK_STRING(act->filename, act->filename);
  CHECK(act->mpi_driver != nullptr);
  CHECK_STRING(act->memory_type, exp->memory_type);
  CHECK(
      act->is_using_periodic_concurrency_mode ==
      exp->is_using_periodic_concurrency_mode);
  CHECK(
      act->periodic_concurrency_range.start ==
      exp->periodic_concurrency_range.start);
  CHECK(
      act->periodic_concurrency_range.end ==
      exp->periodic_concurrency_range.end);
  CHECK(
      act->periodic_concurrency_range.step ==
      exp->periodic_concurrency_range.step);
  CHECK(act->request_period == exp->request_period);
  CHECK(act->request_parameters.size() == exp->request_parameters.size());
  for (auto act_param : act->request_parameters) {
    auto exp_param = exp->request_parameters.find(act_param.first);
    REQUIRE_MESSAGE(
        exp_param != exp->request_parameters.end(),
        "Unexpected parameter: ", act_param.first);

    CHECK(act_param.second.value == exp_param->second.value);
    CHECK(act_param.second.type == exp_param->second.type);
  }
}


#define CHECK_INT_OPTION(option_name, exp_val, msg)                          \
  SUBCASE("valid value")                                                     \
  {                                                                          \
    int argc = 5;                                                            \
    char* argv[argc] = {app_name, "-m", model_name, option_name, "2000"};    \
    CAPTURE(argv[3]);                                                        \
    CAPTURE(argv[4]);                                                        \
                                                                             \
    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));                         \
    CHECK(!parser.UsageCalled());                                            \
    CAPTURE(parser.GetUsageMessage());                                       \
                                                                             \
    exp_val = 2000;                                                          \
    CAPTURE(exp_val);                                                        \
  }                                                                          \
                                                                             \
  SUBCASE("negative value")                                                  \
  {                                                                          \
    int argc = 5;                                                            \
    char* argv[argc] = {app_name, "-m", model_name, option_name, "-2000"};   \
    CHECK_THROWS_WITH_AS(                                                    \
        act = parser.Parse(argc, argv), msg.c_str(), PerfAnalyzerException); \
                                                                             \
    check_params = false;                                                    \
  }                                                                          \
                                                                             \
  SUBCASE("floating point value")                                            \
  {                                                                          \
    int argc = 5;                                                            \
    char* argv[argc] = {app_name, "-m", model_name, option_name, "29.5"};    \
                                                                             \
    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));                         \
    CHECK(!parser.UsageCalled());                                            \
                                                                             \
    exp_val = 29;                                                            \
  }                                                                          \
                                                                             \
  SUBCASE("missing value")                                                   \
  {                                                                          \
    int argc = 4;                                                            \
    char* argv[argc] = {app_name, "-m", model_name, option_name};            \
                                                                             \
    CHECK_THROWS_WITH_AS(                                                    \
        act = parser.Parse(argc, argv), "", PerfAnalyzerException);          \
                                                                             \
    check_params = false;                                                    \
  }


TEST_CASE("Testing PerfAnalyzerParameters")
{
  PAParamsPtr params(new PerfAnalyzerParameters{});

  CHECK(params->verbose == false);
  CHECK(params->streaming == false);
  CHECK(params->extra_verbose == false);
  CHECK(params->max_threads == 4);
  CHECK(params->max_threads_specified == false);
  CHECK(params->sequence_length == 20);
  CHECK(params->percentile == -1);
  CHECK(params->request_count == 0);
  CHECK(params->user_data.size() == 0);
  CHECK_STRING("endpoint", params->endpoint, "");
  CHECK(params->input_shapes.size() == 0);
  CHECK(params->measurement_window_ms == 5000);
  CHECK(params->using_concurrency_range == false);
  CHECK(params->concurrency_range.start == 1);
  CHECK(params->concurrency_range.end == 1);
  CHECK(params->concurrency_range.step == 1);
  CHECK(params->latency_threshold_ms == NO_LIMIT);
  CHECK(params->stability_threshold == doctest::Approx(0.1));
  CHECK(params->max_trials == 10);
  CHECK(params->zero_input == false);
  CHECK(params->string_length == 128);
  CHECK_STRING("string_data", params->string_data, "");
  CHECK(params->async == false);
  CHECK(params->forced_sync == false);
  CHECK(params->using_request_rate_range == false);
  CHECK(params->request_rate_range[0] == doctest::Approx(1.0));
  CHECK(params->request_rate_range[1] == doctest::Approx(1.0));
  CHECK(params->request_rate_range[2] == doctest::Approx(1.0));
  CHECK(params->num_of_sequences == 4);
  CHECK(params->search_mode == SearchMode::LINEAR);
  CHECK(params->request_distribution == Distribution::CONSTANT);
  CHECK(params->using_custom_intervals == false);
  CHECK_STRING("request_intervals_file", params->request_intervals_file, "");
  CHECK(params->shared_memory_type == NO_SHARED_MEMORY);
  CHECK(params->output_shm_size == 102400);
  CHECK(params->kind == clientbackend::BackendKind::TRITON);
  CHECK_STRING(
      "model_signature_name", params->model_signature_name, "serving_default");
  CHECK(params->using_grpc_compression == false);
  CHECK(
      params->compression_algorithm ==
      clientbackend::GrpcCompressionAlgorithm::COMPRESS_NONE);
  CHECK(params->measurement_mode == MeasurementMode::TIME_WINDOWS);
  CHECK(params->measurement_request_count == 50);
  CHECK_STRING(
      "triton_server_path", params->triton_server_path, "/opt/tritonserver");
  CHECK_STRING("model_repository_path", params->model_repository_path, "");
  CHECK(params->start_sequence_id == 1);
  CHECK(params->sequence_id_range == UINT32_MAX);
  CHECK_STRING(
      "ssl_grpc_certificate_chain_file",
      params->ssl_options.ssl_grpc_certificate_chain_file, "");
  CHECK_STRING(
      "ssl_grpc_private_key_file",
      params->ssl_options.ssl_grpc_private_key_file, "");
  CHECK_STRING(
      "ssl_grpc_root_certifications_file",
      params->ssl_options.ssl_grpc_root_certifications_file, "");
  CHECK(params->ssl_options.ssl_grpc_use_ssl == false);
  CHECK_STRING(
      "ssl_https_ca_certificates_file",
      params->ssl_options.ssl_https_ca_certificates_file, "");
  CHECK_STRING(
      "ssl_https_client_certificate_file",
      params->ssl_options.ssl_https_client_certificate_file, "");
  CHECK_STRING(
      "ssl_https_client_certificate_type",
      params->ssl_options.ssl_https_client_certificate_type, "");
  CHECK_STRING(
      "ssl_https_private_key_file",
      params->ssl_options.ssl_https_private_key_file, "");
  CHECK_STRING(
      "ssl_https_private_key_type",
      params->ssl_options.ssl_https_private_key_type, "");
  CHECK(params->ssl_options.ssl_https_verify_host == 2);
  CHECK(params->ssl_options.ssl_https_verify_peer == 1);
  CHECK(params->verbose_csv == false);
  CHECK(params->enable_mpi == false);
  CHECK(params->trace_options.size() == 0);
  CHECK(params->using_old_options == false);
  CHECK(params->dynamic_concurrency_mode == false);
  CHECK(params->url_specified == false);
  CHECK_STRING("url", params->url, "localhost:8000");
  CHECK_STRING("model_name", params->model_name, "");
  CHECK_STRING("model_version", params->model_version, "");
  CHECK(params->batch_size == 1);
  CHECK(params->using_batch_size == false);
  CHECK(params->concurrent_request_count == 1);
  CHECK(params->protocol == clientbackend::ProtocolType::HTTP);
  CHECK(params->http_headers->size() == 0);
  CHECK(params->max_concurrency == 0);
  CHECK_STRING("filename", params->filename, "");
  CHECK(params->mpi_driver == nullptr);
  CHECK_STRING("memory_type", params->memory_type, "system");
}

// Test CLParser Class that captures the usage string but suppresses the output
//
class TestCLParser : public CLParser {
 public:
  std::string GetUsageMessage() const { return usage_message_; }
  bool UsageCalled() const { return usage_called_; }

 private:
  std::string usage_message_;
  bool usage_called_ = false;

  virtual void Usage(const std::string& msg = std::string())
  {
    throw PerfAnalyzerException(msg, GENERIC_ERROR);
  }
};

void
CheckValidRange(
    std::vector<char*>& args, char* option_name, TestCLParser& parser,
    PAParamsPtr& act, bool& using_range, Range<uint64_t>& range,
    PAParamsPtr& exp)
{
  SUBCASE("start:end provided")
  {
    exp->max_threads = 400;
    args.push_back(option_name);
    args.push_back("100:400");  // start:end

    int argc = args.size();
    char* argv[argc];
    std::copy(args.begin(), args.end(), argv);

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(!parser.UsageCalled());

    using_range = true;
    range.start = 100;
    range.end = 400;
  }

  SUBCASE("start:end:step provided")
  {
    exp->max_threads = 400;
    args.push_back(option_name);
    args.push_back("100:400:10");  // start:end:step

    int argc = args.size();
    char* argv[argc];
    std::copy(args.begin(), args.end(), argv);

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(!parser.UsageCalled());

    using_range = true;
    range.start = 100;
    range.end = 400;
    range.step = 10;
  }
}

void
CheckInvalidRange(
    std::vector<char*>& args, char* option_name, TestCLParser& parser,
    PAParamsPtr& act, bool& check_params)
{
  std::string expected_msg;

  SUBCASE("too many input values")
  {
    args.push_back(option_name);
    args.push_back("200:100:25:10");

    int argc = args.size();
    char* argv[argc];
    std::copy(args.begin(), args.end(), argv);

    expected_msg = CreateUsageMessage(
        option_name, "The value does not match <start:end:step>.");
    CHECK_THROWS_WITH_AS(
        act = parser.Parse(argc, argv), expected_msg.c_str(),
        PerfAnalyzerException);

    check_params = false;
  }

  SUBCASE("invalid start value")
  {
    args.push_back(option_name);
    args.push_back("bad:400:10");

    int argc = args.size();
    char* argv[argc];
    std::copy(args.begin(), args.end(), argv);

    expected_msg =
        CreateUsageMessage(option_name, "Invalid value provided: bad:400:10");
    CHECK_THROWS_WITH_AS(
        act = parser.Parse(argc, argv), expected_msg.c_str(),
        PerfAnalyzerException);

    check_params = false;
  }

  SUBCASE("invalid end value")
  {
    args.push_back(option_name);
    args.push_back("100:bad:10");

    int argc = args.size();
    char* argv[argc];
    std::copy(args.begin(), args.end(), argv);

    expected_msg =
        CreateUsageMessage(option_name, "Invalid value provided: 100:bad:10");
    CHECK_THROWS_WITH_AS(
        act = parser.Parse(argc, argv), expected_msg.c_str(),
        PerfAnalyzerException);

    check_params = false;
  }

  SUBCASE("invalid step value")
  {
    args.push_back(option_name);
    args.push_back("100:400:bad");

    int argc = args.size();
    char* argv[argc];
    std::copy(args.begin(), args.end(), argv);

    expected_msg =
        CreateUsageMessage(option_name, "Invalid value provided: 100:400:bad");
    CHECK_THROWS_WITH_AS(
        act = parser.Parse(argc, argv), expected_msg.c_str(),
        PerfAnalyzerException);

    check_params = false;
  }

  SUBCASE("no input values")
  {
    args.push_back(option_name);

    int argc = args.size();
    char* argv[argc];
    std::copy(args.begin(), args.end(), argv);

    // BUG (TMA-1307): Usage message does not contain error. Error statement
    // "option '--concurrency-range' requires an argument" written directly
    // to std::out
    //
    CHECK_THROWS_WITH_AS(
        act = parser.Parse(argc, argv), "", PerfAnalyzerException);

    check_params = false;
  }
}


TEST_CASE("Testing Command Line Parser")
{
  char* model_name = "my_model";
  char* app_name = "test_perf_analyzer";

  std::string expected_msg;
  std::vector<char*> args{app_name, "-m", model_name};

  opterr = 1;  // Enable error output for GetOpt library
  bool check_params = true;

  TestCLParser parser;  // Command Line parser under test
  PAParamsPtr act;      // Actual options parsed from parser
  PAParamsPtr exp{new PerfAnalyzerParameters()};  // Expected results

  // Most common defaults
  exp->model_name = model_name;  // model_name;
  exp->max_threads = 16;

  SUBCASE("with no parameters")
  {
    int argc = 1;
    char* argv[argc] = {app_name};

    expected_msg =
        CreateUsageMessage("-m (model name)", "The value must be specified.");
    CHECK_THROWS_WITH_AS(
        act = parser.Parse(argc, argv), expected_msg.c_str(),
        PerfAnalyzerException);

    check_params = false;
  }

  SUBCASE("with min parameters")
  {
    int argc = 3;
    char* argv[argc] = {app_name, "-m", model_name};

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    REQUIRE(!parser.UsageCalled());
  }

  SUBCASE("Option : --streaming")
  {
    SUBCASE("streaming option - without model")
    {
      int argc = 2;
      char* argv[argc] = {app_name, "--streaming"};

      expected_msg =
          CreateUsageMessage("-m (model name)", "The value must be specified.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("with model")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "-m", model_name, "--streaming"};

      // NOTE: This is not an informative error message, how do I specify a gRPC
      // protocol? Error output should list missing params.
      //
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv),
          "Streaming is only allowed with gRPC protocol.",
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("with model last")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "--streaming", "-m", model_name};

      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv),
          "Streaming is only allowed with gRPC protocol.",
          PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --max-threads")
  {
    SUBCASE("set to 1")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--max-threads", "1"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      REQUIRE(!parser.UsageCalled());

      exp->max_threads = 1;
      exp->max_threads_specified = true;
    }

    SUBCASE("set to max")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--max-threads", "65535"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      REQUIRE(!parser.UsageCalled());

      exp->max_threads = 65535;
      exp->max_threads_specified = true;
    }

    SUBCASE("missing value")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "-m", model_name, "--max-threads"};

      // NOTE: Empty message is not helpful
      //
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), "", PerfAnalyzerException);

      // BUG: Dumping string "option '--max-threads' requires an argument"
      // directly to std::out, instead of through usage()
      //
      check_params = false;
    }

    SUBCASE("bad value")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "-m", model_name, "--max-threads", "bad"};

      // NOTE: Empty message is not helpful
      //
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), "", PerfAnalyzerException);

      // BUG: Dumping string "option '--max-threads' requires an argument"
      // directly to std::out, instead of through usage()
      //
      check_params = false;
    }
  }

  SUBCASE("Option : --sequence-length")
  {
    SUBCASE("set to 2000")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--sequence-length", "2000"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->sequence_length = 2000;
    }

    SUBCASE("set to 0")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--sequence-length", "0"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->sequence_length = 20;
    }
  }

  SUBCASE("Option : --sequence-length-variation")
  {
    SUBCASE("non-negative")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--sequence-length-variation", "33.3"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->sequence_length_variation = 33.3;
    }

    SUBCASE("negative")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--sequence-length-variation", "-10"};

      expected_msg = CreateUsageMessage(
          "--sequence-length-variation", "The value must be >= 0.0.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --percentile")
  {
    SUBCASE("set to 25")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--percentile", "25"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->percentile = 25;
    }

    SUBCASE("set to 225 - overflow check")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--percentile", "225"};

      expected_msg = CreateUsageMessage(
          "--percentile",
          "The value must be -1 for not reporting or in range (0, 100).");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("set to -1 - use average latency")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--percentile", "-1"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->percentile = -1;
    }
  }

  SUBCASE("Option : --data-directory")
  {
    SUBCASE("set to `/usr/data`")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--data-directory", "/usr/data"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->user_data.push_back("/usr/data");
    }

    SUBCASE("call twice")
    {
      // QUESTION: Is this the expected behavior? There is not enough details in
      // in the output. It is marked as deprecated, what does that mean? Is it
      // used?
      //
      int argc = 7;
      char* argv[argc] = {app_name,           "-m",        model_name,
                          "--data-directory", "/usr/data", "--data-directory",
                          "/another/dir"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->user_data.push_back("/usr/data");
      exp->user_data.push_back("/another/dir");
    }
  }

  SUBCASE("Option : --sequence-id-range")
  {
    SUBCASE("One arg")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--sequence-id-range", "53"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->start_sequence_id = 53;
      exp->sequence_id_range = UINT32_MAX;
    }
    SUBCASE("Two args")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--sequence-id-range", "53:67"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->start_sequence_id = 53;
      exp->sequence_id_range = 14;
    }
    SUBCASE("Three args")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--sequence-id-range", "53:67:92"};

      expected_msg = CreateUsageMessage(
          "--sequence-id-range", "The value does not match <start:end>.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }
    SUBCASE("Not a number")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--sequence-id-range", "BAD"};

      expected_msg = CreateUsageMessage(
          "--sequence-id-range", "Invalid value provided: BAD");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;  // Usage message called
    }
    SUBCASE("Not a number 2")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--sequence-id-range", "53:BAD"};

      expected_msg = CreateUsageMessage(
          "--sequence-id-range", "Invalid value provided: 53:BAD");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;  // Usage message called
    }
  }


  SUBCASE("Option : --input-tensor-format")
  {
    SUBCASE("binary")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--input-tensor-format", "binary"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->input_tensor_format = cb::TensorFormat::BINARY;
    }
    SUBCASE("json")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--input-tensor-format", "json"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->input_tensor_format = cb::TensorFormat::JSON;
    }
    SUBCASE("invalid")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--input-tensor-format", "invalid"};

      expected_msg = CreateUsageMessage(
          "--input-tensor-format",
          "Unsupported type provided: 'invalid'. The available options are "
          "'binary' or 'json'.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }
  }


  SUBCASE("Option : --shape")
  {
    SUBCASE("expected input, single shape")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--shape", "input_name:1,2,3"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->input_shapes.emplace(
          std::string("input_name"), std::vector<int64_t>{1, 2, 3});
    }

    SUBCASE("expected input, multiple shapes")
    {
      int argc = 9;
      char* argv[argc] = {
          app_name,
          "-m",
          model_name,
          "--shape",
          "input_name:1,2,3",
          "--shape",
          "alpha:10,24",
          "--shape",
          "beta:10,200,34,15,9000"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->input_shapes.emplace(
          std::string("input_name"), std::vector<int64_t>{1, 2, 3});
      exp->input_shapes.emplace(
          std::string("alpha"), std::vector<int64_t>{10, 24});
      exp->input_shapes.emplace(
          std::string("beta"), std::vector<int64_t>{10, 200, 34, 15, 9000});
    }

    SUBCASE("using negative dims")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--shape", "input_name:-1,2,3"};

      expected_msg = CreateUsageMessage(
          "--shape", "The dimensions of input tensor must be > 0.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("equals sign, not colon")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--shape", "input_name=-1,2,3"};

      expected_msg = CreateUsageMessage(
          "--shape", "There must be a colon after input name.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("missing shape")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--shape", "input_name"};

      expected_msg = CreateUsageMessage(
          "--shape", "There must be a colon after input name.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("missing colon")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--shape", "input_name1,2,3"};

      expected_msg = CreateUsageMessage(
          "--shape", "There must be a colon after input name.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("bad shapes - a,b,c")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--shape", "input_name:a,b,c"};

      expected_msg = CreateUsageMessage(
          "--shape", "Invalid value provided: input_name:a,b,c");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;  // Usage message called
    }

    SUBCASE("bad shapes - [1,2,3]")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--shape", "input_name:[1,2,3]"};

      expected_msg = CreateUsageMessage(
          "--shape", "Invalid value provided: input_name:[1,2,3]");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;  // Usage message called
    }
  }

  SUBCASE("Option : --measurement-interval")
  {
    SUBCASE("set to 500")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "", "500"};

      SUBCASE("Long form")
      {
        argv[3] = "--measurement-interval";
      }

      SUBCASE("Short form")
      {
        argv[3] = "-p";
      }

      CAPTURE(argv[3]);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->measurement_window_ms = 500;
    }

    SUBCASE("set to -200")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "", "-200"};

      SUBCASE("Long form")
      {
        argv[3] = "--measurement-interval";
      }

      SUBCASE("Short form")
      {
        argv[3] = "-p";
      }

      CAPTURE(argv[3]);

      expected_msg = CreateUsageMessage(
          "--measurement-interval (-p)", "The value must be > 0 msec.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("set to non-numeric value")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "", "foobar"};

      SUBCASE("Long form")
      {
        argv[3] = "--measurement-interval";
        expected_msg = CreateUsageMessage(
            "--measurement-interval", "Invalid value provided: foobar");
      }

      SUBCASE("Short form")
      {
        argv[3] = "-p";
        expected_msg =
            CreateUsageMessage("-p", "Invalid value provided: foobar");
      }

      CAPTURE(argv[3]);

      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;  // Usage message called
    }
  }

  SUBCASE("Option : --concurrency-range")
  {
    char* option_name = "--concurrency-range";

    SUBCASE("start provided")
    {
      args.push_back(option_name);
      args.push_back("100");  // start

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->using_concurrency_range = true;
      exp->concurrency_range.start = 100;
      exp->max_threads = 16;
    }

    CheckValidRange(
        args, option_name, parser, act, exp->using_concurrency_range,
        exp->concurrency_range, exp);
    CheckInvalidRange(args, option_name, parser, act, check_params);

    SUBCASE("wrong separator")
    {
      args.push_back(option_name);
      args.push_back("100,400,10");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      // BUG (TMA-1307): Should detect this and through an error. User will
      // enter this and have no clue why the end and step sizes are not used
      // correctly.
      //

      check_params = false;
    }

    SUBCASE("invalid condition - end and latency threshold are 0")
    {
      args.push_back(option_name);
      args.push_back("100:0:25");
      args.push_back("--latency-threshold");
      args.push_back("0");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv),
          "The end of the search range and the latency limit can not be both 0 "
          "(or 0.0) simultaneously",
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("concurrency-range.end < 16")
    {
      args.push_back(option_name);
      args.push_back("10:10");  // start

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->using_concurrency_range = true;
      exp->concurrency_range.start = 10;
      exp->concurrency_range.end = 10;
      exp->max_threads = 16;
    }

    SUBCASE("concurrency-range.end == 16")
    {
      args.push_back(option_name);
      args.push_back("10:16");  // start

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->using_concurrency_range = true;
      exp->concurrency_range.start = 10;
      exp->concurrency_range.end = 16;
      exp->max_threads = 16;
    }

    SUBCASE("concurrency-range.end > 16")
    {
      args.push_back(option_name);
      args.push_back("10:40");  // start

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->using_concurrency_range = true;
      exp->concurrency_range.start = 10;
      exp->concurrency_range.end = 40;
      exp->max_threads = 40;
    }
  }

  SUBCASE("Option : --periodic-concurrency-range")
  {
    char* option_name = "--periodic-concurrency-range";

    // Add required args that specifies where to dump profiled data
    args.insert(
        args.end(), {"-i", "grpc", "--async", "--streaming",
                     "--profile-export-file", "profile.json"});
    exp->protocol = cb::ProtocolType::GRPC;
    exp->async = true;
    exp->streaming = true;
    exp->url = "localhost:8001";  // gRPC url

    SUBCASE("start provided")
    {
      args.push_back(option_name);
      args.push_back("100");  // start

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      expected_msg = CreateUsageMessage(
          option_name, "Both <start> and <end> values must be provided.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    exp->max_threads = 400;

    CheckValidRange(
        args, option_name, parser, act, exp->is_using_periodic_concurrency_mode,
        exp->periodic_concurrency_range, exp);

    CheckInvalidRange(args, option_name, parser, act, check_params);

    SUBCASE("more than one load mode")
    {
      args.push_back(option_name);
      args.push_back("100:400");
      args.push_back("--concurrency-range");
      args.push_back("10:40");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      expected_msg =
          "Cannot specify more then one inference load mode. Please choose "
          "only one of the following modes: --concurrency-range, "
          "--periodic-concurrency-range, --request-rate-range, or "
          "--request-intervals.";
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("no export file specified")
    {
      // Remove the export file args
      args.pop_back();
      args.pop_back();

      args.push_back(option_name);
      args.push_back("100:400");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      expected_msg =
          "Must provide --profile-export-file when using the "
          "--periodic-concurrency-range option.";
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("step is not factor of range size")
    {
      args.push_back(option_name);
      args.push_back("100:400:7");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      expected_msg = CreateUsageMessage(
          option_name,
          "The <step> value must be a factor of the range size (<end> - "
          "<start>).");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("step is zero")
    {
      args.push_back(option_name);
      args.push_back("10:400:0");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      expected_msg =
          CreateUsageMessage(option_name, "The <step> value must be > 0.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --request-period")
  {
    expected_msg =
        CreateUsageMessage("--request-period", "The value must be > 0");
    CHECK_INT_OPTION("--request-period", exp->request_period, expected_msg);

    SUBCASE("set to 0")
    {
      args.push_back("--request-period");
      args.push_back("0");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --request-parameter")
  {
    char* option_name = "--request-parameter";

    // Add required args that specifies where to dump profiled data
    args.insert(args.end(), {"-i", "grpc", "--async", "--streaming"});
    exp->protocol = cb::ProtocolType::GRPC;
    exp->async = true;
    exp->streaming = true;
    exp->url = "localhost:8001";  // gRPC url

    SUBCASE("valid parameter")
    {
      args.push_back(option_name);
      args.push_back("max_tokens:256:int");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      cb::RequestParameter param;
      param.value = "256";
      param.type = "int";
      exp->request_parameters["max_tokens"] = param;
    }

    SUBCASE("missing type")
    {
      args.push_back(option_name);
      args.push_back("max_tokens:256");

      int argc = args.size();
      char* argv[argc];
      std::copy(args.begin(), args.end(), argv);

      expected_msg = CreateUsageMessage(
          option_name, "The value does not match <name:value:type>.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --latency-threshold")
  {
    expected_msg = CreateUsageMessage(
        "--latency-threshold (-l)", "The value must be >= 0 msecs.");
    CHECK_INT_OPTION(
        "--latency-threshold", exp->latency_threshold_ms, expected_msg);

    SUBCASE("set to 0")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--latency-threshold", "0"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());
    }
  }

  SUBCASE("Option : --stability-percentage")
  {
    SUBCASE("valid value")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--stability-percentage", "80"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->stability_threshold = .8f;
    }

    SUBCASE("set to 0")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--stability-percentage", "0"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());
    }

    SUBCASE("negative value")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--stability-percentage", "-20"};

      expected_msg = CreateUsageMessage(
          "--stability-percentage (-s)", "The value must be >= 0.0.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("floating point value")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--stability-percentage", "29.5"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->stability_threshold = .295f;
    }

    SUBCASE("missing value")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "-m", model_name, "--stability-percentage"};

      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), "", PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --max-trials")
  {
    expected_msg =
        CreateUsageMessage("--max-trials (-r)", "The value must be > 0.");
    CHECK_INT_OPTION("--max-trials", exp->max_trials, expected_msg);

    SUBCASE("set to 0")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--max-trials", "0"};

      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --request-count")
  {
    SUBCASE("valid value")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--request-count", "500"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->request_count = 500;
      exp->measurement_mode = MeasurementMode::COUNT_WINDOWS;
      exp->measurement_request_count = 500;
    }
    SUBCASE("negative value")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--request-count", "-2"};

      expected_msg =
          CreateUsageMessage("--request-count", "The value must be > 0.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);
      check_params = false;
    }
    SUBCASE("less than request rate")
    {
      int argc = 7;
      char* argv[argc] = {app_name,   "-m",
                          model_name, "--request-count",
                          "2",        "--request-rate-range",
                          "5"};

      expected_msg = "request-count can not be less than request-rate";
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);
      check_params = false;
    }
    SUBCASE("less than concurrency")
    {
      int argc = 7;
      char* argv[argc] = {app_name,   "-m",
                          model_name, "--request-count",
                          "2",        "--concurrency-range",
                          "5"};

      expected_msg = "request-count can not be less than concurrency";
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);
      check_params = false;
    }
    SUBCASE("multiple request rate")
    {
      int argc = 7;
      char* argv[argc] = {app_name,   "-m",
                          model_name, "--request-count",
                          "20",       "--request-rate-range",
                          "5:6:1"};

      expected_msg =
          "request-count not supported with multiple request-rate values in "
          "one run";
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);
      check_params = false;
    }
    SUBCASE("multiple concurrency")
    {
      int argc = 7;
      char* argv[argc] = {app_name,   "-m",
                          model_name, "--request-count",
                          "20",       "--concurrency-range",
                          "5:6:1"};

      expected_msg =
          "request-count not supported with multiple concurrency values in "
          "one run";
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);
      check_params = false;
    }

    SUBCASE("mode and count are overwritten with non-zero request-count")
    {
      int argc = 9;
      char* argv[argc] = {
          app_name,
          "-m",
          model_name,
          "--request-count",
          "2000",
          "--measurement-mode",
          "time_windows",
          "measurement-request-count",
          "30"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->request_count = 2000;
      exp->measurement_mode = MeasurementMode::COUNT_WINDOWS;
      exp->measurement_request_count = 2000;
    }
    SUBCASE("zero value (no override to measurement mode)")
    {
      int argc = 7;
      char* argv[argc] = {app_name,          "-m", model_name,
                          "--request-count", "0",  "--measurement-mode",
                          "time_windows"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->request_count = 0;
      exp->measurement_mode = MeasurementMode::TIME_WINDOWS;
    }
    SUBCASE("zero value (no override to measurement request count)")
    {
      int argc = 9;
      char* argv[argc] = {
          app_name,
          "-m",
          model_name,
          "--request-count",
          "0",
          "--measurement-mode",
          "count_windows",
          "--measurement-request-count",
          "50"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->request_count = 0;
      exp->measurement_mode = MeasurementMode::COUNT_WINDOWS;
      exp->measurement_request_count = 50;
    }
  }

  SUBCASE("Option : --collect-metrics")
  {
    SUBCASE("with --service-kind != triton")
    {
      int argc = 8;
      char* argv[argc] = {
          app_name,         "-m",        model_name, "--collect-metrics",
          "--service-kind", "tfserving", "-i",       "grpc"};

      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv),
          "Server-side metric collection is only supported with Triton client "
          "backend.",
          PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --metrics-url")
  {
    // missing --collect-metrics
    int argc = 5;
    char* argv[argc] = {
        app_name, "-m", model_name, "--metrics-url", "localhost:8002/metrics"};

    CHECK_THROWS_WITH_AS(
        act = parser.Parse(argc, argv),
        "Must specify --collect-metrics when using the --metrics-url option.",
        PerfAnalyzerException);

    check_params = false;
  }

  SUBCASE("Option : --metrics-interval")
  {
    SUBCASE("missing --collect-metrics")
    {
      int argc = 5;
      char* argv[argc] = {
          app_name, "-m", model_name, "--metrics-interval", "1000"};

      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv),
          "Must specify --collect-metrics when using the --metrics-interval "
          "option.",
          PerfAnalyzerException);

      check_params = false;
    }

    SUBCASE("metrics interval 0")
    {
      int argc = 6;
      char* argv[argc] = {
          app_name, "-m", model_name, "--collect-metrics", "--metrics-interval",
          "0"};

      expected_msg = CreateUsageMessage(
          "--metrics-interval", "The value must be > 0 msecs.");
      CHECK_THROWS_WITH_AS(
          act = parser.Parse(argc, argv), expected_msg.c_str(),
          PerfAnalyzerException);

      check_params = false;
    }
  }

  SUBCASE("Option : --bls-composing-models")
  {
    int argc = 5;

    SUBCASE("one model")
    {
      char* argv[argc] = {
          app_name, "-m", model_name, "--bls-composing-models", "a"};
      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(act->bls_composing_models.size() == 1);
      CHECK_STRING(act->bls_composing_models[0].first, "a");
      CHECK_STRING(act->bls_composing_models[0].second, "");
    }
    SUBCASE("lists with no version")
    {
      SUBCASE("a,b,c")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models", "a,b,c"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      SUBCASE("a, b, c")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models", "a, b, c"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      SUBCASE("a,b, c")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models", "a,b, c"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      SUBCASE("a, b,c")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models", "a, b,c"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      SUBCASE("a, b,  c")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models", "a, b,  c"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }

      CHECK(!parser.UsageCalled());
      REQUIRE(act->bls_composing_models.size() == 3);
      CHECK_STRING(act->bls_composing_models[0].first, "a");
      CHECK_STRING(act->bls_composing_models[1].first, "b");
      CHECK_STRING(act->bls_composing_models[2].first, "c");
      CHECK_STRING(act->bls_composing_models[0].second, "");
      CHECK_STRING(act->bls_composing_models[1].second, "");
      CHECK_STRING(act->bls_composing_models[2].second, "");
    }
    SUBCASE("list with version")
    {
      SUBCASE("a:1,b:2,c:1")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models",
            "a:1,b:2,c:1"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      SUBCASE("a:1, b:2, c:1")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models",
            "a:1, b:2, c:1"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      SUBCASE("a:1,  b:2, c:1")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models",
            "a:1,  b:2, c:1"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      SUBCASE("a:1 ,  b:2, c:1")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models",
            "a:1 ,  b:2, c:1"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      CHECK(!parser.UsageCalled());
      REQUIRE(act->bls_composing_models.size() == 3);
      CHECK_STRING(act->bls_composing_models[0].first, "a");
      CHECK_STRING(act->bls_composing_models[1].first, "b");
      CHECK_STRING(act->bls_composing_models[2].first, "c");
      CHECK_STRING(act->bls_composing_models[0].second, "1");
      CHECK_STRING(act->bls_composing_models[1].second, "2");
      CHECK_STRING(act->bls_composing_models[2].second, "1");
    }
    SUBCASE("list with some versions")
    {
      SUBCASE("a,b:3,c")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models", "a,b:3,c"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      CHECK(!parser.UsageCalled());
      REQUIRE(act->bls_composing_models.size() == 3);
      CHECK_STRING(act->bls_composing_models[0].first, "a");
      CHECK_STRING(act->bls_composing_models[1].first, "b");
      CHECK_STRING(act->bls_composing_models[2].first, "c");
      CHECK_STRING(act->bls_composing_models[0].second, "");
      CHECK_STRING(act->bls_composing_models[1].second, "3");
      CHECK_STRING(act->bls_composing_models[2].second, "");
    }
    SUBCASE("multiple versions of the same model")
    {
      SUBCASE("a:1,b:2,a:2")
      {
        char* argv[argc] = {
            app_name, "-m", model_name, "--bls-composing-models", "a:1,b,a:2"};
        REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      }
      CHECK(!parser.UsageCalled());
      REQUIRE(act->bls_composing_models.size() == 3);
      CHECK_STRING(act->bls_composing_models[0].first, "a");
      CHECK_STRING(act->bls_composing_models[1].first, "b");
      CHECK_STRING(act->bls_composing_models[2].first, "a");
      CHECK_STRING(act->bls_composing_models[0].second, "1");
      CHECK_STRING(act->bls_composing_models[1].second, "");
      CHECK_STRING(act->bls_composing_models[2].second, "2");
    }
  }

  if (check_params) {
    if (act == nullptr) {
      std::cerr
          << "Error: Attempting to access `act` but was not initialized. Check "
             "if the test cases are missing `check_params = false` statement."
          << std::endl;
      exit(1);
    }
    CHECK_PARAMS(act, exp);
  }
  optind = 1;  // Reset GotOpt index, needed to parse the next command line
}
}}  // namespace triton::perfanalyzer
