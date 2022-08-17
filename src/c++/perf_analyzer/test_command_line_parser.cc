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
//
#include <getopt.h>
#include <array>
#include "command_line_parser.h"
#include "doctest.h"

namespace triton { namespace perfanalyzer {

inline void
CHECK_STRING(const char* name, std::string str, const char* val)
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
  CHECK(params->user_data.size() == 0);
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
  CHECK_STRING("triton_server_path", params->triton_server_path, "");
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
    usage_called_ = true;
    usage_message_ = msg;
  }
};

TEST_CASE("Testing Command Line Parser")
{
  char* model_name = "my_model";
  char* app_name = "test_perf_analyzer";

  opterr = 0;  // Disable error output for GetOpt library

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

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    REQUIRE(parser.UsageCalled());
    CHECK_STRING(
        "Usage Message", parser.GetUsageMessage(), "-m flag must be specified");

    exp->model_name = "";
    CHECK_PARAMS(act, exp);
  }

  SUBCASE("with min parameters")
  {
    int argc = 3;
    char* argv[argc] = {app_name, "-m", model_name};

    PAParamsPtr act;
    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    REQUIRE(!parser.UsageCalled());

    CHECK_PARAMS(act, exp);
  }

  SUBCASE("Option : --streaming")
  {
    SUBCASE("streaming option - without model")
    {
      int argc = 2;
      char* argv[argc] = {app_name, "--streaming"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      REQUIRE(parser.UsageCalled());
      CHECK_STRING(
          "Usage Message", parser.GetUsageMessage(),
          "streaming is only allowed with gRPC protocol");

      exp->model_name = "";
      exp->streaming = true;
      // exp->max_threads = 16;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("with model")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "-m", model_name, "--streaming"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      REQUIRE(parser.UsageCalled());

      // NOTE: This is not an informative error message, how do I specify a gRPC
      // protocol? Error ouput should list missing params.
      //
      CHECK_STRING(
          "Usage Message", parser.GetUsageMessage(),
          "streaming is only allowed with gRPC protocol");

      exp->streaming = true;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("with model last")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "--streaming", "-m", model_name};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));

      REQUIRE(parser.UsageCalled());
      CHECK_STRING(
          "Usage Message", parser.GetUsageMessage(),
          "streaming is only allowed with gRPC protocol");

      exp->streaming = true;
      CHECK_PARAMS(act, exp);
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
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("set to max")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--max-threads", "65535"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      REQUIRE(!parser.UsageCalled());

      exp->max_threads = 65535;
      exp->max_threads_specified = true;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("missing value")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "-m", model_name, "--max-threads"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      REQUIRE(parser.UsageCalled());

      // NOTE: Empty message is not helpful
      //
      CHECK_STRING("Usage Message", parser.GetUsageMessage(), "");
      // BUG: Dumping string "option '--max-threads' requires an argument"
      // directly to std::out, instead of through usage()
      //

      CHECK_PARAMS(act, exp);
    }

    SUBCASE("bad value")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "-m", model_name, "--max-threads", "bad"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      REQUIRE(parser.UsageCalled());

      // NOTE: Empty message is not helpful
      //
      CHECK_STRING("Usage Message", parser.GetUsageMessage(), "");
      // BUG: Dumping string "option '--max-threads' requires an argument"
      // directly to std::out, instead of through usage()
      //

      CHECK_PARAMS(act, exp);
    }
  }

  SUBCASE("Option : --sequence-length")
  {
    SUBCASE("set to 2000")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--sequence-length",
                          "2000"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->sequence_length = 2000;
      CHECK_PARAMS(act, exp);
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
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("set to 225 - overflow check")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--percentile", "225"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(parser.UsageCalled());
      CHECK_STRING(
          "Usage Message", parser.GetUsageMessage(),
          "percentile must be -1 for not reporting or in range (0, 100)");

      exp->percentile = 225;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("set to -1 - use average latency")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--percentile", "-1"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->percentile = -1;
      CHECK_PARAMS(act, exp);
    }
  }

  SUBCASE("Option : --data-directory")
  {
    SUBCASE("set to `/usr/data`")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--data-directory",
                          "/usr/data"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->user_data.push_back("/usr/data");
      CHECK_PARAMS(act, exp);
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
      CHECK_PARAMS(act, exp);
    }
  }

  SUBCASE("Option : --shape")
  {
    SUBCASE("expected input, single shape")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--shape",
                          "input_name:1,2,3"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->input_shapes.emplace(
          std::string("input_name"), std::vector<int64_t>{1, 2, 3});
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("expected input, multiple shapes")
    {
      int argc = 9;
      char* argv[argc] = {app_name,
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
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("using negative dims")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--shape",
                          "input_name:-1,2,3"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(parser.UsageCalled());
      CHECK_STRING(
          "Usage Message", parser.GetUsageMessage(), "input shape must be > 0");

      exp->input_shapes.emplace(
          std::string("input_name"), std::vector<int64_t>{-1, 2, 3});
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("equals sign, not colon")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--shape",
                          "input_name=-1,2,3"};

      // BUG this should call usages with the message
      // "failed to parse input shape. There must be a colon after input name
      //
      CHECK_THROWS_WITH(
          act = parser.Parse(argc, argv),
          "basic_string::substr: __pos (which is 18) > this->size() (which is "
          "17)");
    }

    SUBCASE("missing shape")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--shape", "input_name"};

      // BUG this should call usages with the message
      // "failed to parse input shape. There must be a colon after input name
      //
      CHECK_THROWS_WITH(
          act = parser.Parse(argc, argv),
          "basic_string::substr: __pos (which is 11) > this->size() (which is "
          "10)");
    }

    SUBCASE("missing colon")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--shape",
                          "input_name1,2,3"};

      // BUG this should call usages with the message
      // "failed to parse input shape. There must be a colon after input name
      //
      CHECK_THROWS_WITH(
          act = parser.Parse(argc, argv),
          "basic_string::substr: __pos (which is 16) > this->size() (which is "
          "15)");
    }

    SUBCASE("bad shapes - a,b,c")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--shape",
                          "input_name:a,b,c"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(parser.UsageCalled());
      CHECK_STRING(
          "Usage Message", parser.GetUsageMessage(),
          "failed to parse input shape: input_name:a,b,c");

      exp->input_shapes.emplace(
          std::string("input_name"), std::vector<int64_t>{});
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("bad shapes - [1,2,3]")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--shape",
                          "input_name:[1,2,3]"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(parser.UsageCalled());
      CHECK_STRING(
          "Usage Message", parser.GetUsageMessage(),
          "failed to parse input shape: input_name:[1,2,3]");

      exp->input_shapes.emplace(
          std::string("input_name"), std::vector<int64_t>{});
      CHECK_PARAMS(act, exp);
    }
  }

  SUBCASE("Option : --measurement-interval")
  {
    SUBCASE("set to 500")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "", "500"};

      SUBCASE("Long form") { argv[3] = "--measurement-interval"; }

      SUBCASE("Short form") { argv[3] = "-p"; }

      CAPTURE(argv[3]);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->measurement_window_ms = 500;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("set to -200")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "", "-200"};

      SUBCASE("Long form") { argv[3] = "--measurement-interval"; }

      SUBCASE("Short form") { argv[3] = "-p"; }

      CAPTURE(argv[3]);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      // BUG: may want to actually error out here, and not just use the unsigned
      // conversion. This will result in unexpected behavior. The actual value
      // becomes 18446744073709551416ULL, which is not what you would want.
      //
      exp->measurement_window_ms = -200;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("set to non-numeric value")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "", "foobar"};

      SUBCASE("Long form") { argv[3] = "--measurement-interval"; }

      SUBCASE("Short form") { argv[3] = "-p"; }

      CAPTURE(argv[3]);

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(parser.UsageCalled());
      CHECK_STRING(
          "Usage Message", parser.GetUsageMessage(),
          "measurement window must be > 0 in msec");

      exp->measurement_window_ms = 0;
      CHECK_PARAMS(act, exp);
    }
  }

  SUBCASE("Option : --concurrency-range")
  {
    SUBCASE("expected use")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                          "100:400:10"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->using_concurrency_range = true;
      exp->concurrency_range.start = 100;
      exp->concurrency_range.end = 400;
      exp->concurrency_range.step = 10;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("only two options")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                          "100:400"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->using_concurrency_range = true;
      exp->concurrency_range.start = 100;
      exp->concurrency_range.end = 400;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("only one options")
    {
      int argc = 5;
      char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                          "100"};

      // QUESTION: What does this mean? Why pass only one?
      //
      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(!parser.UsageCalled());

      exp->using_concurrency_range = true;
      exp->concurrency_range.start = 100;
      CHECK_PARAMS(act, exp);
    }

    SUBCASE("no options")
    {
      int argc = 4;
      char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range"};

      REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
      CHECK(parser.UsageCalled());

      // BUG: Usage message does not contain error. Error statement
      // "option '--concurrency-range' requires an argument" written directly
      // to std::out
      //
      CHECK_STRING("Usage Message", parser.GetUsageMessage(), "");

      CHECK_PARAMS(act, exp);
    }
  }

  SUBCASE("too many options")
  {
    int argc = 5;
    char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                        "200:100:25:10"};

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(parser.UsageCalled());
    CHECK_STRING(
        "Usage Message", parser.GetUsageMessage(),
        "option concurrency-range can have maximum of three elements");

    exp->using_concurrency_range = true;
    exp->concurrency_range.start = 200;
    exp->concurrency_range.end = 100;
    exp->concurrency_range.step = 25;
    CHECK_PARAMS(act, exp);
  }

  SUBCASE("way too many options")
  {
    int argc = 5;
    char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                        "200:100:25:10:20:30"};

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(parser.UsageCalled());
    CHECK_STRING(
        "Usage Message", parser.GetUsageMessage(),
        "option concurrency-range can have maximum of three elements");

    exp->using_concurrency_range = true;
    exp->concurrency_range.start = 200;
    exp->concurrency_range.end = 100;
    exp->concurrency_range.step = 25;
    CHECK_PARAMS(act, exp);
  }

  SUBCASE("wrong separator")
  {
    int argc = 5;
    char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                        "100,400,10"};

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(!parser.UsageCalled());

    // BUG: Should detect this and through an error. User will enter this and
    // have no clue why the end and step sizes are not used correctly.
    //

    exp->using_concurrency_range = true;
    exp->concurrency_range.start = 100;
    CHECK_PARAMS(act, exp);
  }

  SUBCASE("bad start value")
  {
    int argc = 5;
    char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                        "bad:400:10"};

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(parser.UsageCalled());
    CHECK_STRING(
        "Usage Message", parser.GetUsageMessage(),
        "failed to parse concurrency range: bad:400:10");

    exp->using_concurrency_range = true;
    CHECK_PARAMS(act, exp);
  }

  SUBCASE("bad end value")
  {
    int argc = 5;
    char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                        "100:bad:10"};

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(parser.UsageCalled());
    CHECK_STRING(
        "Usage Message", parser.GetUsageMessage(),
        "failed to parse concurrency range: 100:bad:10");

    exp->using_concurrency_range = true;
    exp->concurrency_range.start = 100;
    CHECK_PARAMS(act, exp);
  }

  SUBCASE("bad step value")
  {
    int argc = 5;
    char* argv[argc] = {app_name, "-m", model_name, "--concurrency-range",
                        "100:400:bad"};

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(parser.UsageCalled());
    CHECK_STRING(
        "Usage Message", parser.GetUsageMessage(),
        "failed to parse concurrency range: 100:400:bad");

    exp->using_concurrency_range = true;
    exp->concurrency_range.start = 100;
    exp->concurrency_range.end = 400;
    CHECK_PARAMS(act, exp);
  }

  SUBCASE("invalid condition - end and latency threshold are 0")
  {
    int argc = 7;
    char* argv[argc] = {app_name,   "-m",
                        model_name, "--concurrency-range",
                        "100:0:25", "--latency-threshold",
                        "0"};

    REQUIRE_NOTHROW(act = parser.Parse(argc, argv));
    CHECK(parser.UsageCalled());
    CHECK_STRING(
        "Usage Message", parser.GetUsageMessage(),
        "The end of the search range and the latency limit can not be both 0 "
        "(or 0.0) simultaneously");

    exp->using_concurrency_range = true;
    exp->concurrency_range.start = 100;
    exp->concurrency_range.end = 0;
    exp->concurrency_range.step = 25;
    exp->latency_threshold_ms = 0;
    CHECK_PARAMS(act, exp);
  }

  optind = 1;  // Reset GotOpt index, needed to parse the next command line
}
}}  // namespace triton::perfanalyzer
