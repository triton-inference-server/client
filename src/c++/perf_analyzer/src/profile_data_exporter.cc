// Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "profile_data_exporter.h"

#include <rapidjson/filewritestream.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include "client_backend/client_backend.h"

namespace triton { namespace perfanalyzer {

cb::Error
ProfileDataExporter::Create(std::shared_ptr<ProfileDataExporter>* exporter)
{
  std::shared_ptr<ProfileDataExporter> local_exporter{
      new ProfileDataExporter()};
  *exporter = std::move(local_exporter);
  return cb::Error::Success;
}

void
ProfileDataExporter::Export(
    const std::vector<Experiment>& raw_experiments, std::string& raw_version,
    std::string& file_path, cb::BackendKind& service_kind,
    std::string& endpoint)
{
  ConvertToJson(raw_experiments, raw_version, service_kind, endpoint);
  OutputToFile(file_path);
}

void
ProfileDataExporter::ConvertToJson(
    const std::vector<Experiment>& raw_experiments, std::string& raw_version,
    cb::BackendKind& service_kind, std::string& endpoint)
{
  ClearDocument();
  rapidjson::Value experiments(rapidjson::kArrayType);

  for (const auto& raw_experiment : raw_experiments) {
    rapidjson::Value entry(rapidjson::kObjectType);
    rapidjson::Value experiment(rapidjson::kObjectType);
    rapidjson::Value requests(rapidjson::kArrayType);
    rapidjson::Value window_boundaries(rapidjson::kArrayType);

    AddExperiment(entry, experiment, raw_experiment);
    AddRequests(entry, requests, raw_experiment);
    AddWindowBoundaries(entry, window_boundaries, raw_experiment);

    experiments.PushBack(entry, document_.GetAllocator());
  }

  document_.AddMember("experiments", experiments, document_.GetAllocator());
  AddVersion(raw_version);
  AddServiceKind(service_kind);
  AddEndpoint(endpoint);
}

void
ProfileDataExporter::ClearDocument()
{
  rapidjson::Document d{};
  document_.Swap(d);
  document_.SetObject();
}

void
ProfileDataExporter::AddExperiment(
    rapidjson::Value& entry, rapidjson::Value& experiment,
    const Experiment& raw_experiment)
{
  rapidjson::Value mode;
  rapidjson::Value value;
  if (raw_experiment.mode.concurrency != 0) {
    mode = rapidjson::StringRef("concurrency");
    value.SetUint64(raw_experiment.mode.concurrency);
  } else {
    mode = rapidjson::StringRef("request_rate");
    value.SetDouble(raw_experiment.mode.request_rate);
  }
  experiment.AddMember("mode", mode, document_.GetAllocator());
  experiment.AddMember("value", value, document_.GetAllocator());
  entry.AddMember("experiment", experiment, document_.GetAllocator());
}

void
ProfileDataExporter::AddRequests(
    rapidjson::Value& entry, rapidjson::Value& requests,
    const Experiment& raw_experiment)
{
  for (auto& raw_request : raw_experiment.requests) {
    rapidjson::Value request(rapidjson::kObjectType);
    rapidjson::Value timestamp;

    timestamp.SetUint64(raw_request.start_time_.time_since_epoch().count());
    request.AddMember("timestamp", timestamp, document_.GetAllocator());

    if (raw_request.sequence_id_ != 0) {
      rapidjson::Value sequence_id;
      sequence_id.SetUint64(raw_request.sequence_id_);
      request.AddMember("sequence_id", sequence_id, document_.GetAllocator());
    }

    rapidjson::Value request_inputs(rapidjson::kObjectType);
    AddRequestInputs(request_inputs, raw_request.request_inputs_);
    request.AddMember(
        "request_inputs", request_inputs, document_.GetAllocator());

    rapidjson::Value response_timestamps(rapidjson::kArrayType);
    AddResponseTimestamps(
        response_timestamps, raw_request.response_timestamps_);
    request.AddMember(
        "response_timestamps", response_timestamps, document_.GetAllocator());

    rapidjson::Value response_outputs(rapidjson::kArrayType);
    AddResponseOutputs(response_outputs, raw_request.response_outputs_);
    request.AddMember(
        "response_outputs", response_outputs, document_.GetAllocator());

    requests.PushBack(request, document_.GetAllocator());
  }
  entry.AddMember("requests", requests, document_.GetAllocator());
}

void
ProfileDataExporter::AddResponseTimestamps(
    rapidjson::Value& timestamps_json,
    const std::vector<std::chrono::time_point<std::chrono::system_clock>>&
        timestamps)
{
  for (auto& timestamp : timestamps) {
    rapidjson::Value timestamp_json;
    timestamp_json.SetUint64(timestamp.time_since_epoch().count());
    timestamps_json.PushBack(timestamp_json, document_.GetAllocator());
  }
}

void
ProfileDataExporter::AddRequestInputs(
    rapidjson::Value& request_inputs_json,
    const std::vector<RequestRecord::RequestInput>& request_inputs)
{
  for (const auto& request_input : request_inputs) {
    for (const auto& input : request_input) {
      const auto& name{input.first};
      const auto& buf{input.second.data_.get()};
      const auto& byte_size{input.second.size_};
      const auto& data_type{input.second.data_type_};
      rapidjson::Value name_json(name.c_str(), document_.GetAllocator());
      rapidjson::Value input_json{};
      // TMA-1777: support other data types
      if (buf != nullptr) {
        if (data_type == "BYTES" || data_type == "JSON") {
          input_json.SetString(
              reinterpret_cast<const char*>(buf), byte_size,
              document_.GetAllocator());
        } else if (data_type == "INT32") {
          auto* val = reinterpret_cast<int32_t*>(buf);
          input_json.SetInt(*val);
        } else if (data_type == "BOOL") {
          bool is_true = (*buf > 0);
          input_json.SetBool(is_true);
        } else {
          std::cerr << "WARNING: data type '" + data_type +
                           "' is not supported with JSON."
                    << std::endl;
        }
      } else {
        input_json.SetString("", 0, document_.GetAllocator());
      }
      request_inputs_json.AddMember(
          name_json, input_json, document_.GetAllocator());
    }
  }
}

void
ProfileDataExporter::AddResponseOutputs(
    rapidjson::Value& outputs_json,
    const std::vector<RequestRecord::ResponseOutput>& response_outputs)
{
  for (const auto& response_output : response_outputs) {
    rapidjson::Value response_output_json(rapidjson::kObjectType);
    for (const auto& output : response_output) {
      const auto& name{output.first};
      const auto& buf{output.second.data_.get()};
      const auto& byte_size{output.second.size_};
      rapidjson::Value name_json(name.c_str(), document_.GetAllocator());
      rapidjson::Value output_json{};
      // TMA-1777: support other data types
      if (buf != nullptr) {
        output_json.SetString(
            reinterpret_cast<const char*>(buf), byte_size,
            document_.GetAllocator());
      } else {
        output_json.SetString("", 0, document_.GetAllocator());
      }
      response_output_json.AddMember(
          name_json, output_json, document_.GetAllocator());
    }
    outputs_json.PushBack(response_output_json, document_.GetAllocator());
  }
}

void
ProfileDataExporter::AddWindowBoundaries(
    rapidjson::Value& entry, rapidjson::Value& window_boundaries,
    const Experiment& raw_experiment)
{
  for (auto& window : raw_experiment.window_boundaries) {
    rapidjson::Value w;
    w.SetUint64(window);
    window_boundaries.PushBack(w, document_.GetAllocator());
  }
  entry.AddMember(
      "window_boundaries", window_boundaries, document_.GetAllocator());
}

void
ProfileDataExporter::AddVersion(std::string& raw_version)
{
  rapidjson::Value version;
  version = rapidjson::StringRef(raw_version.c_str());
  document_.AddMember("version", version, document_.GetAllocator());
}

void
ProfileDataExporter::AddServiceKind(cb::BackendKind& kind)
{
  std::string raw_service_kind{""};
  if (kind == cb::BackendKind::TRITON) {
    raw_service_kind = "triton";
  } else if (kind == cb::BackendKind::TENSORFLOW_SERVING) {
    raw_service_kind = "tfserving";
  } else if (kind == cb::BackendKind::TORCHSERVE) {
    raw_service_kind = "torchserve";
  } else if (kind == cb::BackendKind::TRITON_C_API) {
    raw_service_kind = "triton_c_api";
  } else if (kind == cb::BackendKind::OPENAI) {
    raw_service_kind = "openai";
  } else {
    std::cerr << "Unknown service kind detected. The 'service_kind' will not "
                 "be specified."
              << std::endl;
  }

  rapidjson::Value service_kind;
  service_kind.SetString(raw_service_kind.c_str(), document_.GetAllocator());
  document_.AddMember("service_kind", service_kind, document_.GetAllocator());
}

void
ProfileDataExporter::AddEndpoint(std::string& raw_endpoint)
{
  rapidjson::Value endpoint;
  endpoint = rapidjson::StringRef(raw_endpoint.c_str());
  document_.AddMember("endpoint", endpoint, document_.GetAllocator());
}

void
ProfileDataExporter::OutputToFile(std::string& file_path)
{
  FILE* fp = fopen(file_path.c_str(), "w");
  if (fp == nullptr) {
    throw PerfAnalyzerException(
        "failed to open file for outputting raw profile data", GENERIC_ERROR);
  }
  char writeBuffer[65536];
  rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));

  rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
  document_.Accept(writer);

  fclose(fp);
}

}}  // namespace triton::perfanalyzer
