// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    std::string& file_path)
{
  ConvertToJson(raw_experiments, raw_version);
  OutputToFile(file_path);
}

void
ProfileDataExporter::ConvertToJson(
    const std::vector<Experiment>& raw_experiments, std::string& raw_version)
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
    value.SetUint64(raw_experiment.mode.request_rate);
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

    rapidjson::Value responses(rapidjson::kArrayType);
    AddResponses(responses, raw_request.response_times_);
    request.AddMember(
        "response_timestamps", responses, document_.GetAllocator());
    requests.PushBack(request, document_.GetAllocator());
  }
  entry.AddMember("requests", requests, document_.GetAllocator());
}

void
ProfileDataExporter::AddResponses(
    rapidjson::Value& responses,
    const std::vector<std::chrono::time_point<std::chrono::system_clock>>&
        response_times)
{
  for (auto& response : response_times) {
    rapidjson::Value time;
    time.SetUint64(response.time_since_epoch().count());
    responses.PushBack(time, document_.GetAllocator());
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
