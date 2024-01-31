// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "constants.h"
#include "mpi_utils.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

enum SEARCH_RANGE { kSTART = 0, kEND = 1, kSTEP = 2 };

// Perf Analyzer command line parameters.
// PAParams are used to initialize PerfAnalyzer and track configuration
//
struct PerfAnalyzerParameters {
  bool verbose = false;
  bool extra_verbose = false;
  bool streaming = false;
  size_t max_threads = 4;
  bool max_threads_specified = false;
  size_t sequence_length = 20;  // average length of a sentence
  bool sequence_length_specified = false;
  double sequence_length_variation = 20.0;
  int32_t percentile = -1;
  std::vector<std::string> user_data;
  std::unordered_map<std::string, std::vector<int64_t>> input_shapes;
  std::vector<cb::ModelIdentifier> bls_composing_models;
  uint64_t measurement_window_ms = 5000;
  bool using_concurrency_range = false;
  Range<uint64_t> concurrency_range{1, 1, 1};
  std::unordered_map<std::string, cb::RequestParameter> request_parameters;
  uint64_t latency_threshold_ms = NO_LIMIT;
  double stability_threshold = 0.1;
  size_t max_trials = 10;
  bool zero_input = false;
  size_t string_length = 128;
  std::string string_data;
  bool async = false;
  bool forced_sync = false;
  bool using_request_rate_range = false;
  double request_rate_range[3] = {1.0, 1.0, 1.0};
  uint32_t num_of_sequences = 4;
  bool serial_sequences = false;
  SearchMode search_mode = SearchMode::LINEAR;
  Distribution request_distribution = Distribution::CONSTANT;
  bool using_custom_intervals = false;
  std::string request_intervals_file{""};
  SharedMemoryType shared_memory_type = NO_SHARED_MEMORY;
  size_t output_shm_size = 100 * 1024;
  clientbackend::BackendKind kind = clientbackend::BackendKind::TRITON;
  std::string model_signature_name{"serving_default"};
  bool using_grpc_compression = false;
  clientbackend::GrpcCompressionAlgorithm compression_algorithm =
      clientbackend::GrpcCompressionAlgorithm::COMPRESS_NONE;
  MeasurementMode measurement_mode = MeasurementMode::TIME_WINDOWS;
  uint64_t measurement_request_count = 50;
  std::string triton_server_path = "/opt/tritonserver";
  std::string model_repository_path;
  uint64_t start_sequence_id = 1;
  uint64_t sequence_id_range = UINT32_MAX;
  clientbackend::SslOptionsBase ssl_options;  // gRPC and HTTP SSL options

  // Verbose csv option for including additional information
  bool verbose_csv = false;

  // Enable MPI option for using MPI functionality with multi-model mode.
  bool enable_mpi = false;
  std::map<std::string, std::vector<std::string>> trace_options;
  bool using_old_options = false;
  bool dynamic_concurrency_mode = false;
  bool url_specified = false;
  std::string url{"localhost:8000"};
  std::string model_name;
  std::string model_version;
  uint64_t batch_size = 1;
  bool using_batch_size = false;
  int32_t concurrent_request_count = 1;
  clientbackend::ProtocolType protocol = clientbackend::ProtocolType::HTTP;
  std::shared_ptr<clientbackend::Headers> http_headers{
      new clientbackend::Headers()};
  size_t max_concurrency = 0;
  std::string filename{""};
  std::shared_ptr<MPIDriver> mpi_driver;
  std::string memory_type{"system"};  // currently not used, to be removed

  // Enable collection of server-side metrics from inference server.
  bool should_collect_metrics{false};

  // The URL to query for server-side inference server metrics.
  std::string metrics_url{"localhost:8002/metrics"};
  bool metrics_url_specified{false};

  // How often, within each measurement window, to query for server-side
  // inference server metrics.
  uint64_t metrics_interval_ms{1000};
  bool metrics_interval_ms_specified{false};

  // Model is LLM. Will determine if PA outputs LLM related metrics
  bool is_llm{false};

  // Return true if targeting concurrency
  //
  bool targeting_concurrency() const
  {
    return (
        using_concurrency_range || using_old_options ||
        !(using_request_rate_range || using_custom_intervals ||
          is_using_periodic_concurrency_mode));
  }

  // Sets the threshold for PA client overhead.
  // Overhead is defined as the percentage of time when PA is doing work and
  // requests are not outstanding to the triton server. If the overhead
  // percentage exceeds the threshold, a warning is displayed.
  //
  double overhead_pct_threshold{50.0};

  // Triton inference request input tensor format.
  cb::TensorFormat input_tensor_format{cb::TensorFormat::BINARY};

  // Triton inference response output tensor format.
  cb::TensorFormat output_tensor_format{cb::TensorFormat::BINARY};

  // The profile export file path.
  std::string profile_export_file{""};

  bool is_using_periodic_concurrency_mode{false};
  Range<uint64_t> periodic_concurrency_range{1, 1, 1};
  uint64_t request_period{10};
};

using PAParamsPtr = std::shared_ptr<PerfAnalyzerParameters>;

class CLParser {
 public:
  CLParser() : params_(new PerfAnalyzerParameters{}) {}

  // Parse command line arguments into a parameters struct
  //
  PAParamsPtr Parse(int argc, char** argv);

 private:
  char** argv_;
  int argc_;
  PAParamsPtr params_;

  std::string FormatMessage(std::string str, int offset) const;
  virtual void Usage(const std::string& msg = std::string());
  void PrintVersion();
  void ParseCommandLine(int argc, char** argv);
  void VerifyOptions();
};
}}  // namespace triton::perfanalyzer
