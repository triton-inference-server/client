// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#define TRITON_INFERENCE_SERVER_CLIENT_CLASS \
  triton::perfanalyzer::clientbackend::tritoncapi::TritonLoader

#include "triton_loader.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <sys/stat.h>
#include <future>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include "c_api_infer_results.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {
namespace {
bool enforce_memory_type = false;
TRITONSERVER_MemoryType requested_memory_type;
bool helper_verbose = false;
/// Helper function for allocating memory
TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // Initially attempt to make the actual memory type and id that we
  // allocate be the same as preferred memory type
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    if (helper_verbose) {
      std::cout << "allocated " << byte_size << " bytes for result tensor "
                << tensor_name << std::endl;
    }
  } else {
    void* allocated_ptr = nullptr;
    if (enforce_memory_type) {
      *actual_memory_type = requested_memory_type;
    }

    switch (*actual_memory_type) {
      // Use CPU memory if the requested memory type is unknown
      // (default case).
      case TRITONSERVER_MEMORY_CPU:
      default: {
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        allocated_ptr = malloc(byte_size);
        break;
      }
    }

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      if (helper_verbose) {
        std::cout << "allocated " << byte_size << " bytes in "
                  << size_t(*actual_memory_type) << " for result tensor "
                  << tensor_name << std::endl;
      }
    }
  }

  return nullptr;  // Success
}

/// Helper function for releasing memory
TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }
  if (helper_verbose) {
    std::cout << "Releasing buffer " << buffer << " of size " << byte_size
              << " in " << size_t(memory_type) << " for result '" << *name
              << "'" << std::endl;
  }
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                << std::endl;
      break;
  }

  delete name;

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  // request is deleted at the end of the Infer call so don't need to do
  // anything here
}


void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

Error
GetModelVersionFromString(const std::string& version_string, int64_t* version)
{
  if (version_string.empty()) {
    *version = 1;
    return Error::Success;
  }

  try {
    *version = std::stol(version_string);
  }
  catch (std::exception& e) {
    return Error(
        std::string(
            "failed to get model version from specified version string '" +
            version_string + "' (details: " + e.what() +
            "), version should be an integral value > 0")
            .c_str());
  }

  if (*version < 0) {
    return Error(std::string(
                     "invalid model version specified '" + version_string +
                     "' , version should be an integral value > 0")
                     .c_str());
  }

  return Error::Success;
}

Error
FolderExists(const std::string& path)
{
  struct stat buffer;
  if (!stat(path.c_str(), &buffer)) {
    return Error::Success;
  } else {
    return Error("Unable to find filepath: " + path);
  }
}
}  // namespace
Error
TritonLoader::Create(
    const std::string& triton_server_path,
    const std::string& model_repository_path, const std::string& memory_type,
    bool verbose)
{
  if (!GetSingleton()->ServerIsReady()) {
    if (triton_server_path.empty() || model_repository_path.empty()) {
      return Error("cannot load server, paths are empty");
    }
    GetSingleton()->ClearHandles();
    FAIL_IF_ERR(
        GetSingleton()->PopulateInternals(
            triton_server_path, model_repository_path, memory_type, verbose),
        "Populating internal variables");
    FAIL_IF_ERR(
        GetSingleton()->LoadServerLibrary(), "Loading Triton Server library");
    FAIL_IF_ERR(
        GetSingleton()->StartTriton(memory_type), "Starting Triton Server");
  }

  return Error::Success;
}

Error
TritonLoader::Delete()
{
  if (GetSingleton()->server_ != nullptr) {
    GetSingleton()->server_is_ready_ = false;
    GetSingleton()->model_is_loaded_ = false;
    (GetSingleton()->server_).reset();
  }
  return Error::Success;
}

Error
TritonLoader::PopulateInternals(
    const std::string& triton_server_path,
    const std::string& model_repository_path, const std::string& memory_type,
    bool verbose)
{
  RETURN_IF_ERROR(FolderExists(triton_server_path));
  RETURN_IF_ERROR(FolderExists(model_repository_path));
  GetSingleton()->triton_server_path_ = triton_server_path;
  GetSingleton()->model_repository_path_ = model_repository_path;
  GetSingleton()->verbose_ = verbose;
  GetSingleton()->verbose_level_ = GetSingleton()->verbose_ ? 1 : 0;
  return Error::Success;
}

Error
TritonLoader::StartTriton(const std::string& memory_type)
{
  // Check API version.
  uint32_t api_version_major, api_version_minor;
  REPORT_TRITONSERVER_ERROR(
      GetSingleton()->api_version_fn_(&api_version_major, &api_version_minor));
  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    std::stringstream sstream;
    sstream << "triton server API version mismatch. \n"
            << "Expected version major:" << TRITONSERVER_API_VERSION_MAJOR
            << ", minor:" << TRITONSERVER_API_VERSION_MINOR << "\n"
            << "  Actual version major:" << api_version_major
            << ", minor:" << api_version_minor;
    return Error(sstream.str());
  }
  // Create the server...
  TRITONSERVER_ServerOptions* server_options = nullptr;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->options_new_fn_(&server_options),
      "creating server options");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->options_set_model_repo_path_fn_(
          server_options, GetSingleton()->model_repository_path_.c_str()),
      "setting model repository path");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->set_log_verbose_fn_(
          server_options, GetSingleton()->verbose_level_),
      "setting verbose logging level");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->set_log_info_fn_(
          server_options, GetSingleton()->verbose_),
      "setting if log verbose level is true");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->set_backend_directory_fn_(
          server_options,
          (GetSingleton()->triton_server_path_ + "/backends").c_str()),
      "setting backend directory");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->set_repo_agent_directory_fn_(
          server_options,
          (GetSingleton()->triton_server_path_ + "/repoagents").c_str()),
      "setting repository agent directory");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->set_strict_model_config_fn_(server_options, true),
      "setting strict model configuration");
  double min_compute_capability = 0;
  // FIXME: Do not have GPU support right now
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->set_min_supported_compute_capability_fn_(
          server_options, min_compute_capability),
      "setting minimum supported CUDA compute capability");
  TRITONSERVER_Server* server_ptr = nullptr;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->server_new_fn_(&server_ptr, server_options),
      "creating server");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->server_options_delete_fn_(server_options),
      "deleting server options");
  std::shared_ptr<TRITONSERVER_Server> shared_server(
      server_ptr, GetSingleton()->server_delete_fn_);
  GetSingleton()->server_ = shared_server;

  // Wait until the server is both live and ready.
  size_t health_iters = 0;
  while (true) {
    bool live, ready;
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->server_is_live_fn_(
            (GetSingleton()->server_).get(), &live),
        "unable to get server liveness");
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->server_is_ready_fn_(
            (GetSingleton()->server_).get(), &ready),
        "unable to get server readiness");
    if (live && ready) {
      std::cout << "server is alive!" << std::endl;
      GetSingleton()->server_is_ready_ = true;
      break;
    }

    if (++health_iters >= 10) {
      return Error("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  // Print status of the server.
  if (GetSingleton()->verbose_) {
    TRITONSERVER_Message* server_metadata_message;
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->server_metadata_fn_(
            (GetSingleton()->server_).get(), &server_metadata_message),
        "unable to get server metadata message");
    const char* buffer;
    size_t byte_size;
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->message_serialize_to_json_fn_(
            server_metadata_message, &buffer, &byte_size),
        "unable to serialize server metadata message");

    std::cout << "Server Status:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;

    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->message_delete_fn_(server_metadata_message),
        "deleting status metadata");
  }

  return Error::Success;
}

Error
TritonLoader::ServerMetaData(rapidjson::Document* server_metadata)
{
  if (!ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  TRITONSERVER_Message* server_metadata_message;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->server_metadata_fn_(
          (GetSingleton()->server_).get(), &server_metadata_message),
      "unable to get server metadata message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->message_serialize_to_json_fn_(
          server_metadata_message, &buffer, &byte_size),
      "unable to serialize server metadata message");
  server_metadata->Parse(buffer, byte_size);
  if (server_metadata->HasParseError()) {
    return Error(
        "error: failed to parse server metadata from JSON: " +
        std::string(GetParseError_En(server_metadata->GetParseError())) +
        " at " + std::to_string(server_metadata->GetErrorOffset()));
  }
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->message_delete_fn_(server_metadata_message),
      "deleting status metadata");
  return Error::Success;
}

Error
TritonLoader::LoadModel(
    const std::string& model_name, const std::string& model_version)
{
  if (!ServerIsReady()) {
    return Error("server is not ready, abort!");
  }
  GetSingleton()->model_name_ = model_name;

  RETURN_IF_ERROR(GetModelVersionFromString(
      model_version, &(GetSingleton()->model_version_)));
  // Wait for the model to become available.
  bool is_ready = false;
  size_t health_iters = 0;

  // some error handling
  if (GetSingleton()->model_repository_path_.empty()) {
    return Error("Need to specify model repository");
  }
  while (!is_ready) {
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->model_is_ready_fn_(
            GetSingleton()->server_.get(), GetSingleton()->model_name_.c_str(),
            GetSingleton()->model_version_, &is_ready),
        "unable to get model readiness");
    if (!is_ready) {
      if (++health_iters >= 10) {
        return Error("model failed to be ready in 10 iterations");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }
  }
  GetSingleton()->model_is_loaded_ =
      true;  // flag to confirm model is correct and loaded
  return Error::Success;
}

Error
TritonLoader::ModelMetadata(rapidjson::Document* model_metadata)
{
  if (!ModelIsLoaded() || !ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  TRITONSERVER_Message* model_metadata_message;

  // get model metadata
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->model_metadata_fn_(
          (GetSingleton()->server_).get(), GetSingleton()->model_name_.c_str(),
          GetSingleton()->model_version_, &model_metadata_message),
      "unable to get model metadata message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->message_serialize_to_json_fn_(
          model_metadata_message, &buffer, &byte_size),
      "unable to serialize model status protobuf");

  model_metadata->Parse(buffer, byte_size);
  if (model_metadata->HasParseError()) {
    return Error(
        "error: failed to parse model metadata from JSON: " +
        std::string(GetParseError_En(model_metadata->GetParseError())) +
        " at " + std::to_string(model_metadata->GetErrorOffset()));
  }

  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->message_delete_fn_(model_metadata_message),
      "deleting status protobuf");

  if (strcmp(
          (*model_metadata)["name"].GetString(),
          GetSingleton()->model_name_.c_str())) {
    return Error("unable to find metadata for model");
  }

  bool found_version = false;
  if (model_metadata->HasMember("versions")) {
    for (const auto& version : (*model_metadata)["versions"].GetArray()) {
      if (strcmp(
              version.GetString(),
              std::to_string(GetSingleton()->model_version_).c_str()) == 0) {
        found_version = true;
        break;
      }
    }
  }
  if (!found_version) {
    std::string msg = "unable to find version " +
                      std::to_string(GetSingleton()->model_version_) +
                      " status for model";
    return Error(msg);
  }
  return Error::Success;
}

Error
TritonLoader::ModelConfig(rapidjson::Document* model_config)
{
  if (!ModelIsLoaded() || !ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  TRITONSERVER_Message* model_config_message;
  uint32_t config_version = 1;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->model_config_fn_(
          (GetSingleton()->server_).get(), GetSingleton()->model_name_.c_str(),
          GetSingleton()->model_version_, config_version,
          &model_config_message),
      "unable to get model config message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->message_serialize_to_json_fn_(
          model_config_message, &buffer, &byte_size),
      "unable to serialize model config status protobuf");

  model_config->Parse(buffer, byte_size);
  if (model_config->HasParseError()) {
    return Error(
        "error: failed to parse model config from JSON: " +
        std::string(GetParseError_En(model_config->GetParseError())) + " at " +
        std::to_string(model_config->GetErrorOffset()));
  }

  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->message_delete_fn_(model_config_message),
      "deleting server config status protobuf");

  return Error::Success;
}

Error
TritonLoader::LoadServerLibrary()
{
  std::string full_path =
      GetSingleton()->triton_server_path_ + SERVER_LIBRARY_PATH;
  RETURN_IF_ERROR(FolderExists(full_path));
  FAIL_IF_ERR(
      OpenLibraryHandle(full_path, &dlhandle_),
      "shared library loading library:" + full_path);

  TritonServerApiVersionFn_t apifn;
  TritonServerOptionsNewFn_t onfn;
  TritonServerOptionSetModelRepoPathFn_t rpfn;
  TritonServerSetLogVerboseFn_t slvfn;

  TritonServerSetBackendDirFn_t sbdfn;
  TritonServerSetRepoAgentDirFn_t srdfn;
  TritonServerSetStrictModelConfigFn_t ssmcfn;
  TritonServerSetMinSupportedComputeCapabilityFn_t smsccfn;

  TritonServerNewFn_t snfn;
  TritonServerOptionsDeleteFn_t odfn;
  TritonServerDeleteFn_t sdfn;
  TritonServerIsLiveFn_t ilfn;

  TritonServerIsReadyFn_t irfn;
  TritonServerMetadataFn_t smfn;
  TritonServerMessageSerializeToJsonFn_t stjfn;
  TritonServerMessageDeleteFn_t mdfn;

  TritonServerModelIsReadyFn_t mirfn;
  TritonServerModelMetadataFn_t mmfn;
  TritonServerResponseAllocatorNewFn_t ranfn;
  TritonServerInferenceRequestNewFn_t irnfn;

  TritonServerInferenceRequestSetIdFn_t irsifn;
  TritonServerInferenceRequestSetReleaseCallbackFn_t irsrcfn;
  TritonServerInferenceRequestAddInputFn_t iraifn;
  TritonServerInferenceRequestAddRequestedOutputFn_t irarofn;

  TritonServerInferenceRequestAppendInputDataFn_t iraidfn;
  TritonServerInferenceRequestSetResponseCallbackFn_t irsrescfn;
  TritonServerInferAsyncFn_t iafn;
  TritonServerInferenceResponseErrorFn_t irefn;

  TritonServerInferenceResponseDeleteFn_t irdfn;
  TritonServerResponseAllocatorDeleteFn_t radfn;
  TritonServerErrorNewFn_t enfn;

  TritonServerMemoryTypeStringFn_t mtsfn;
  TritonServerInferenceResponseOutputCountFn_t irocfn;
  TritonServerDataTypeStringFn_t dtsfn;

  TritonServerErrorDeleteFn_t edfn;
  TritonServerErrorCodeToStringFn_t ectsfn;
  TritonServerModelConfigFn_t mcfn;
  TritonServerInferenceRequestSetCorrelationIdFn_t scidfn;
  TritonServerInferenceRequestSetStringCorrelationIdFn_t sscidfn;

  TritonServerInferenceRequestSetFlagsFn_t sffn;
  TritonServerInferenceRequestSetPriorityFn_t spfn;
  TritonServerInferenceRequestSetTimeoutMicrosecondsFn_t stmsfn;
  TritonServerStringToDatatypeFn_t stdtfn;

  TritonServerInferenceResponseOutputFn_t irofn;
  TritonServerRequestIdFn_t ridfn;
  TritonServerRequestDeleteFn_t rdfn;
  TritonServerModelStatisticsFn_t msfn;

  TritonSeverUnloadModelFn_t umfn;
  TritonSeverSetLogInfoFn_t slifn;

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ApiVersion", false /* optional */,
      reinterpret_cast<void**>(&apifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsNew", false /* optional */,
      reinterpret_cast<void**>(&onfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetModelRepositoryPath",
      false /* optional */, reinterpret_cast<void**>(&rpfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetLogVerbose",
      false /* optional */, reinterpret_cast<void**>(&slvfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetBackendDirectory",
      false /* optional */, reinterpret_cast<void**>(&sbdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetRepoAgentDirectory",
      false /* optional */, reinterpret_cast<void**>(&srdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetStrictModelConfig",
      false /* optional */, reinterpret_cast<void**>(&ssmcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability",
      false /* optional */, reinterpret_cast<void**>(&smsccfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerNew", false /* optional */,
      reinterpret_cast<void**>(&snfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsDelete", false /* optional */,
      reinterpret_cast<void**>(&odfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerDelete", false /* optional */,
      reinterpret_cast<void**>(&sdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerIsLive", false /* optional */,
      reinterpret_cast<void**>(&ilfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerIsReady", false /* optional */,
      reinterpret_cast<void**>(&irfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerMetadata", false /* optional */,
      reinterpret_cast<void**>(&smfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MessageSerializeToJson", false /* optional */,
      reinterpret_cast<void**>(&stjfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MessageDelete", false /* optional */,
      reinterpret_cast<void**>(&mdfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelIsReady", false /* optional */,
      reinterpret_cast<void**>(&mirfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelMetadata", false /* optional */,
      reinterpret_cast<void**>(&mmfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ResponseAllocatorNew", false /* optional */,
      reinterpret_cast<void**>(&ranfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestNew", false /* optional */,
      reinterpret_cast<void**>(&irnfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetId", false /* optional */,
      reinterpret_cast<void**>(&irsifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetReleaseCallback",
      false /* optional */, reinterpret_cast<void**>(&irsrcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAddInput", false /* optional */,
      reinterpret_cast<void**>(&iraifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAddRequestedOutput",
      false /* optional */, reinterpret_cast<void**>(&irarofn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAppendInputData",
      false /* optional */, reinterpret_cast<void**>(&iraidfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetResponseCallback",
      false /* optional */, reinterpret_cast<void**>(&irsrescfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerInferAsync", false /* optional */,
      reinterpret_cast<void**>(&iafn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseError", false /* optional */,
      reinterpret_cast<void**>(&irefn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseDelete", false /* optional */,
      reinterpret_cast<void**>(&irdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ResponseAllocatorDelete", false /* optional */,
      reinterpret_cast<void**>(&radfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorNew", false /* optional */,
      reinterpret_cast<void**>(&enfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MemoryTypeString", false /* optional */,
      reinterpret_cast<void**>(&mtsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseOutputCount",
      false /* optional */, reinterpret_cast<void**>(&irocfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_DataTypeString", false /* optional */,
      reinterpret_cast<void**>(&dtsfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorDelete", false /* optional */,
      reinterpret_cast<void**>(&edfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorCodeString", false /* optional */,
      reinterpret_cast<void**>(&ectsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelConfig", false /* optional */,
      reinterpret_cast<void**>(&mcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetCorrelationId",
      false /* optional */, reinterpret_cast<void**>(&scidfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetCorrelationIdString",
      false /* optional */, reinterpret_cast<void**>(&sscidfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetFlags", false /* optional */,
      reinterpret_cast<void**>(&sffn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetPriority",
      false /* optional */, reinterpret_cast<void**>(&spfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetTimeoutMicroseconds",
      false /* optional */, reinterpret_cast<void**>(&stmsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_StringToDataType", false /* optional */,
      reinterpret_cast<void**>(&stdtfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseOutput", false /* optional */,
      reinterpret_cast<void**>(&irofn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestId", false /* optional */,
      reinterpret_cast<void**>(&ridfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestDelete", false /* optional */,
      reinterpret_cast<void**>(&rdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelStatistics", false /* optional */,
      reinterpret_cast<void**>(&msfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerUnloadModel", false /* optional */,
      reinterpret_cast<void**>(&umfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetLogInfo", false /* optional */,
      reinterpret_cast<void**>(&slifn)));


  api_version_fn_ = apifn;
  options_new_fn_ = onfn;
  options_set_model_repo_path_fn_ = rpfn;
  set_log_verbose_fn_ = slvfn;

  set_backend_directory_fn_ = sbdfn;
  set_repo_agent_directory_fn_ = srdfn;
  set_strict_model_config_fn_ = ssmcfn;
  set_min_supported_compute_capability_fn_ = smsccfn;

  server_new_fn_ = snfn;
  server_options_delete_fn_ = odfn;
  server_delete_fn_ = sdfn;
  server_is_live_fn_ = ilfn;

  server_is_ready_fn_ = irfn;
  server_metadata_fn_ = smfn;
  message_serialize_to_json_fn_ = stjfn;
  message_delete_fn_ = mdfn;

  model_is_ready_fn_ = mirfn;
  model_metadata_fn_ = mmfn;
  response_allocator_new_fn_ = ranfn;
  inference_request_new_fn_ = irnfn;

  inference_request_set_id_fn_ = irsifn;
  inference_request_set_release_callback_fn_ = irsrcfn;
  inference_request_add_input_fn_ = iraifn;
  inference_request_add_requested_output_fn_ = irarofn;

  inference_request_append_input_data_fn_ = iraidfn;
  inference_request_set_response_callback_fn_ = irsrescfn;
  infer_async_fn_ = iafn;
  inference_response_error_fn_ = irefn;

  inference_response_delete_fn_ = irdfn;
  response_allocator_delete_fn_ = radfn;
  error_new_fn_ = enfn;

  memory_type_string_fn_ = mtsfn;
  inference_response_output_count_fn_ = irocfn;
  data_type_string_fn_ = dtsfn;

  error_delete_fn_ = edfn;
  error_code_to_string_fn_ = ectsfn;
  model_config_fn_ = mcfn;
  set_correlation_id_fn_ = scidfn;
  set_string_correlation_id_fn_ = sscidfn;

  set_flags_fn_ = sffn;
  set_priority_fn_ = spfn;
  set_timeout_ms_fn_ = stmsfn;
  string_to_datatype_fn_ = stdtfn;

  inference_response_output_fn_ = irofn;
  request_id_fn_ = ridfn;
  request_delete_fn_ = rdfn;
  model_statistics_fn_ = msfn;

  unload_model_fn_ = umfn;
  set_log_info_fn_ = slifn;

  return Error::Success;
}

void
TritonLoader::ClearHandles()
{
  dlhandle_ = nullptr;

  api_version_fn_ = nullptr;
  options_new_fn_ = nullptr;
  options_set_model_repo_path_fn_ = nullptr;
  set_log_verbose_fn_ = nullptr;

  set_backend_directory_fn_ = nullptr;
  set_repo_agent_directory_fn_ = nullptr;
  set_strict_model_config_fn_ = nullptr;
  set_min_supported_compute_capability_fn_ = nullptr;

  server_new_fn_ = nullptr;
  server_options_delete_fn_ = nullptr;
  server_delete_fn_ = nullptr;
  server_is_live_fn_ = nullptr;

  server_is_ready_fn_ = nullptr;
  server_metadata_fn_ = nullptr;
  message_serialize_to_json_fn_ = nullptr;
  message_delete_fn_ = nullptr;

  model_is_ready_fn_ = nullptr;
  model_metadata_fn_ = nullptr;
  response_allocator_new_fn_ = nullptr;
  inference_request_new_fn_ = nullptr;

  inference_request_set_id_fn_ = nullptr;
  inference_request_set_release_callback_fn_ = nullptr;
  inference_request_add_input_fn_ = nullptr;
  inference_request_add_requested_output_fn_ = nullptr;

  inference_request_append_input_data_fn_ = nullptr;
  inference_request_set_response_callback_fn_ = nullptr;
  infer_async_fn_ = nullptr;
  inference_response_error_fn_ = nullptr;

  inference_response_delete_fn_ = nullptr;
  response_allocator_delete_fn_ = nullptr;
  error_new_fn_ = nullptr;

  memory_type_string_fn_ = nullptr;
  inference_response_output_count_fn_ = nullptr;
  data_type_string_fn_ = nullptr;
  error_message_fn_ = nullptr;

  error_delete_fn_ = nullptr;
  error_code_to_string_fn_ = nullptr;
  model_config_fn_ = nullptr;
  set_correlation_id_fn_ = nullptr;
  set_string_correlation_id_fn_ = nullptr;

  set_flags_fn_ = nullptr;
  set_priority_fn_ = nullptr;
  set_timeout_ms_fn_ = nullptr;
  string_to_datatype_fn_ = nullptr;

  inference_response_output_fn_ = nullptr;
  request_id_fn_ = nullptr;
  request_delete_fn_ = nullptr;
  model_statistics_fn_ = nullptr;
  unload_model_fn_ = nullptr;
  set_log_info_fn_ = nullptr;
}

Error
TritonLoader::FileExists(std::string& filepath)
{
  std::ifstream ifile;
  ifile.open(filepath);
  if (!ifile) {
    return Error("unable to find local Triton library: " + filepath);
  } else {
    return Error::Success;
  }
}

Error
TritonLoader::Infer(
    const tc::InferOptions& options, const std::vector<tc::InferInput*>& inputs,
    const std::vector<const tc::InferRequestedOutput*>& outputs,
    InferResult** result)
{
  if (!ServerIsReady() || !ModelIsLoaded()) {
    return Error("Server is not ready and/or requested model is not loaded");
  }
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  tc::RequestTimers timer;
  timer.Reset();
  timer.CaptureTimestamp(tc::RequestTimers::Kind::REQUEST_START);
  GetSingleton()->InitializeRequest(options, outputs, &allocator, &irequest);
  GetSingleton()->AddInputs(inputs, irequest);
  GetSingleton()->AddOutputs(outputs, irequest);
  timer.CaptureTimestamp(tc::RequestTimers::Kind::SEND_START);
  // Perform inference...
  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->inference_request_set_response_callback_fn_(
          irequest, allocator, nullptr /* response_allocator_userp */,
          InferResponseComplete, reinterpret_cast<void*>(p)),
      "setting response callback");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->infer_async_fn_(
          (GetSingleton()->server_).get(), irequest, nullptr /* trace */),
      "running inference");
  timer.CaptureTimestamp(tc::RequestTimers::Kind::SEND_END);
  // Wait for the inference to complete.
  TRITONSERVER_InferenceResponse* completed_response = completed.get();
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->inference_response_error_fn_(completed_response),
      "response status");
  timer.CaptureTimestamp(tc::RequestTimers::Kind::RECV_START);
  timer.CaptureTimestamp(tc::RequestTimers::Kind::RECV_END);
  timer.CaptureTimestamp(tc::RequestTimers::Kind::REQUEST_END);

  tc::Error err = GetSingleton()->UpdateInferStat(timer);
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }
  const char* cid;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->request_id_fn_(irequest, &cid),
      "Failed to get request id");
  std::string id(cid);
  InferResult::Create(result, err, id);
  // clean up
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->inference_response_delete_fn_(completed_response),
      "deleting inference response");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->request_delete_fn_(irequest),
      "deleting inference request");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->response_allocator_delete_fn_(allocator),
      "deleting response allocator");
  return Error::Success;
}

Error
TritonLoader::InitializeRequest(
    const tc::InferOptions& options,
    const std::vector<const tc::InferRequestedOutput*>& outputs,
    TRITONSERVER_ResponseAllocator** allocator,
    TRITONSERVER_InferenceRequest** irequest)
{
  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()
          ->response_allocator_new_fn_(
              allocator,
              reinterpret_cast<
                  TRITONSERVER_Error* (*)(TRITONSERVER_ResponseAllocator * allocator, const char* tensor_name, size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id, void* userp, void** buffer, void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)>(
                  ResponseAlloc),
              reinterpret_cast<
                  TRITONSERVER_Error* (*)(TRITONSERVER_ResponseAllocator * allocator, void* buffer, void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)>(
                  ResponseRelease),
              nullptr /* start_fn */),
      "creating response allocator");

  // set up inference request
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->inference_request_new_fn_(
          irequest, (GetSingleton()->server_).get(),
          GetSingleton()->model_name_.c_str(), GetSingleton()->model_version_),
      "creating inference request");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->inference_request_set_id_fn_(
          *irequest, options.request_id_.c_str()),
      "setting ID for the request");
  if ((options.sequence_id_ != 0) || (options.sequence_id_str_ != "") ||
      (options.priority_ != 0) || (options.server_timeout_ != 0) ||
      outputs.empty()) {
    if (options.sequence_id_ != 0) {
      RETURN_IF_TRITONSERVER_ERROR(
          GetSingleton()->set_correlation_id_fn_(
              *irequest, options.sequence_id_),
          "setting sequence ID for the request");
    } else if (options.sequence_id_str_ != "") {
      RETURN_IF_TRITONSERVER_ERROR(
          GetSingleton()->set_string_correlation_id_fn_(
              *irequest, options.sequence_id_str_.c_str()),
          "setting sequence ID for the request");
    }
    uint32_t flags = 0;
    if (options.sequence_start_) {
      flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
    }
    if (options.sequence_start_) {
      flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
    }
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->set_flags_fn_(*irequest, flags),
        "setting inference flags for the request");
  }
  if (options.priority_ != 0) {
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->set_priority_fn_(*irequest, options.priority_),
        "setting priority for the request");
  }
  if (options.server_timeout_ != 0) {
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->set_timeout_ms_fn_(*irequest, options.server_timeout_),
        "setting timeout for the request");
  }
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->inference_request_set_release_callback_fn_(
          *irequest, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");
  return Error::Success;
}

Error
TritonLoader::AddInputs(
    const std::vector<tc::InferInput*>& inputs,
    TRITONSERVER_InferenceRequest* irequest)
{
  for (auto io : inputs) {
    const char* input_name = io->Name().c_str();
    const char* datatype = io->Datatype().c_str();
    const TRITONSERVER_DataType dtype =
        GetSingleton()->string_to_datatype_fn_(datatype);
    std::vector<int64_t> shape_vec;
    for (const int64_t dim : io->Shape()) {  // this is a vector, just use it
      shape_vec.push_back(dim);
    }
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->inference_request_add_input_fn_(
            irequest, input_name, dtype, &shape_vec[0], shape_vec.size()),
        "setting input for the request");
    if (io->IsSharedMemory()) {
      return Error("shared library not supported for C API");
    }
    size_t byte_size;
    tc::Error err = io->ByteSize(&byte_size);
    if (!err.IsOk()) {
      return Error(err.Message());
    }
    if (byte_size == 0) {
      RETURN_IF_TRITONSERVER_ERROR(
          GetSingleton()->inference_request_append_input_data_fn_(
              irequest, input_name, nullptr, 0 /* byte_size */,
              TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */),
          "appending input data with byte size zero");
    } else {
      io->PrepareForRequest();
      bool end_of_input = false;
      while (!end_of_input) {
        const uint8_t* buf;
        size_t buf_size;
        io->GetNext(&buf, &buf_size, &end_of_input);
        if (buf != nullptr) {
          RETURN_IF_TRITONSERVER_ERROR(
              GetSingleton()->inference_request_append_input_data_fn_(
                  irequest, input_name, const_cast<uint8_t*>(buf), buf_size,
                  GetSingleton()->requested_memory_type_,
                  0 /* memory_type_id */),
              "appending data to tritonserver");
        }
      }
    }
  }


  return Error::Success;
}

Error
TritonLoader::AddOutputs(
    const std::vector<const tc::InferRequestedOutput*>& outputs,
    TRITONSERVER_InferenceRequest* irequest)
{
  for (auto io : outputs) {
    const char* output_name = io->Name().c_str();
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->inference_request_add_requested_output_fn_(
            irequest, output_name),
        "setting output for the request");
  }
  return Error::Success;
}


Error
TritonLoader::ModelInferenceStatistics(
    const std::string& model_name, const std::string& model_version,
    rapidjson::Document* infer_stat)
{
  if (ServerIsReady() && ModelIsLoaded()) {
    TRITONSERVER_Message* model_stats_message = nullptr;
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(model_version, &requested_model_version);
    if (err.IsOk()) {
      RETURN_IF_TRITONSERVER_ERROR(
          GetSingleton()->model_statistics_fn_(
              (GetSingleton()->server_).get(), model_name.c_str(),
              requested_model_version, &model_stats_message),
          "getting model statistics from server");

      const char* buffer;
      size_t byte_size;
      RETURN_IF_TRITONSERVER_ERROR(
          GetSingleton()->message_serialize_to_json_fn_(
              model_stats_message, &buffer, &byte_size),
          "serializing message to json");

      infer_stat->Parse(buffer, byte_size);
      if (infer_stat->HasParseError()) {
        return Error(
            "error: failed to parse server metadata from JSON: " +
            std::string(GetParseError_En(infer_stat->GetParseError())) +
            " at " + std::to_string(infer_stat->GetErrorOffset()));
      }
      RETURN_IF_TRITONSERVER_ERROR(
          GetSingleton()->message_delete_fn_(model_stats_message),
          "deleting inference statistics message");
    }
    return err;
  } else {
    return Error(
        "Trying to get model statistics while server is not started or model "
        "is not ready");
  }
}

TritonLoader*
TritonLoader::GetSingleton()
{
  static TritonLoader loader;
  return &loader;
}

TritonLoader::~TritonLoader()
{
  FAIL_IF_ERR(Delete(), "dereferencing server instance...");
  FAIL_IF_ERR(CloseLibraryHandle(dlhandle_), "error on closing triton loader");
  GetSingleton()->ClearHandles();
}

}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
