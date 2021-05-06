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
  perfanalyzer::clientbackend::TritonLoader

#include "triton_loader.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <unordered_map>
#include "c_api_infer_results.h"

namespace nvidia { namespace inferenceserver { namespace client {
class InferResultCApi;
}}}  // namespace nvidia::inferenceserver::client
namespace perfanalyzer { namespace clientbackend {
namespace {
bool enforce_memory_type = false;
TRITONSERVER_MemoryType requested_memory_type;
bool verbose = false;
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
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
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
      if (verbose) {
        std::cout << "allocated " << byte_size << " bytes in "
                  << size_t(*actual_memory_type) << " for result tensor "
                  << tensor_name << std::endl;
      }
    }
  }

  return nullptr;  // Success
}

/// Helper function for allocating memory
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
  if (verbose) {
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
  // We reuse the request so we don't delete it here.
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
}  // namespace
Error
TritonLoader::Create(
    const std::string& library_directory, const std::string& model_repository,
    const std::string& memory_type, bool verbose)
{
  if (!GetSingleton()->ServerIsReady()) {
    if (library_directory.empty() || model_repository.empty()) {
      return Error("cannot load server, paths are empty");
    }

    Error status = GetSingleton()->PopulateInternals(
        library_directory, model_repository, memory_type, verbose);
    assert(status.IsOk());
    status = GetSingleton()->LoadServerLibrary();
    assert(status.IsOk());
    status = GetSingleton()->StartTriton(memory_type, false);
    assert(status.IsOk());
  }
  return Error::Success;
}

Error
TritonLoader::PopulateInternals(
    const std::string& library_directory, const std::string& model_repository,
    const std::string& memory_type, bool verbose)
{
  GetSingleton()->library_directory_ = library_directory;
  GetSingleton()->model_repository_path_ = model_repository;
  return Error::Success;
}

Error
TritonLoader::StartTriton(const std::string& memory_type, bool isVerbose)
{
  if (isVerbose) {
    GetSingleton()->verbose_level_ = 1;
  }

  // Check API version.
  uint32_t api_version_major, api_version_minor;
  REPORT_TRITONSERVER_ERROR(
      GetSingleton()->api_version_fn_(&api_version_major, &api_version_minor));
  std::cout << "api version major: " << api_version_major
            << ", minor: " << api_version_minor << std::endl;
  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    return Error("triton server API version mismatch");
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
      GetSingleton()->set_backend_directory_fn_(
          server_options,
          (GetSingleton()->library_directory_ + "/backends").c_str()),
      "setting backend directory");
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->set_repo_agent_directory_fn_(
          server_options,
          (GetSingleton()->library_directory_ + "/repoagents").c_str()),
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
    std::cout << "Server Health: live " << live << ", ready " << ready
              << std::endl;
    if (live && ready) {
      std::cout << "server is alive!" << std::endl;
      break;
    }

    if (++health_iters >= 10) {
      return Error("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  // Print status of the server.
  {
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
  GetSingleton()->server_is_ready_ = true;

  return Error::Success;
}

Error
TritonLoader::ServerMetaData(rapidjson::Document* server_metadata)
{
  if (!ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  std::cout << "ServerMetaData..." << std::endl;

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
  std::cout << "loading model..." << std::endl;
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

  std::cout << "loaded model " << GetSingleton()->model_name_ << std::endl;
  GetSingleton()->model_is_loaded_ =
      true;  // flag to confirm model is correct and loaded
  return Error::Success;
}

Error
TritonLoader::UnloadModel()
{
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->unload_model_fn_(
          (GetSingleton()->server_).get(),
          (GetSingleton()->model_name_).c_str()),
      "unloading the model");
  GetSingleton()->model_is_loaded_ = false;
  return Error::Success;
}

Error
TritonLoader::ModelMetadata(rapidjson::Document* model_metadata)
{
  if (!ModelIsLoaded() || !ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  std::cout << "ModelMetadata..." << std::endl;
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
  std::cout << "ModelConfig..." << std::endl;

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
      GetSingleton()->library_directory_ + SERVER_LIBRARY_PATH;
  RETURN_IF_ERROR(FileExists(full_path));
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
  TritonServerInferenceRequestRemoveAllInputDataFn_t irraidfn;
  TritonServerResponseAllocatorDeleteFn_t radfn;
  TritonServerErrorNewFn_t enfn;

  TritonServerMemoryTypeStringFn_t mtsfn;
  TritonServerInferenceResponseOutputCountFn_t irocfn;
  TritonServerDataTypeStringFn_t dtsfn;
  TritonServerErrorMessageFn_t emfn;

  TritonServerErrorDeleteFn_t edfn;
  TritonServerErrorCodeToStringFn_t ectsfn;
  TritonServerModelConfigFn_t mcfn;
  TritonServerInferenceRequestSetCorrelationIdFn_t scidfn;

  TritonServerInferenceRequestSetFlagsFn_t sffn;
  TritonServerInferenceRequestSetPriorityFn_t spfn;
  TritonServerInferenceRequestSetTimeoutMicrosecondsFn_t stmsfn;
  TritonServerStringToDatatypeFn_t stdtfn;
  TritonServerInferenceResponseOutputFn_t irofn;
  TritonServerRequestIdFn_t ridfn;
  TritonServerRequestDeleteFn_t rdfn;
  TritonServerModelStatisticsFn_t msfn;
  TritonSeverUnloadModelFn_t umfn;

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ApiVersion", true /* optional */,
      reinterpret_cast<void**>(&apifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsNew", true /* optional */,
      reinterpret_cast<void**>(&onfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetModelRepositoryPath",
      true /* optional */, reinterpret_cast<void**>(&rpfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetLogVerbose", true /* optional */,
      reinterpret_cast<void**>(&slvfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetBackendDirectory",
      true /* optional */, reinterpret_cast<void**>(&sbdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetRepoAgentDirectory",
      true /* optional */, reinterpret_cast<void**>(&srdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetStrictModelConfig",
      true /* optional */, reinterpret_cast<void**>(&ssmcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability",
      true /* optional */, reinterpret_cast<void**>(&smsccfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerNew", true /* optional */,
      reinterpret_cast<void**>(&snfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsDelete", true /* optional */,
      reinterpret_cast<void**>(&odfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerDelete", true /* optional */,
      reinterpret_cast<void**>(&sdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerIsLive", true /* optional */,
      reinterpret_cast<void**>(&ilfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerIsReady", true /* optional */,
      reinterpret_cast<void**>(&irfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerMetadata", true /* optional */,
      reinterpret_cast<void**>(&smfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MessageSerializeToJson", true /* optional */,
      reinterpret_cast<void**>(&stjfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MessageDelete", true /* optional */,
      reinterpret_cast<void**>(&mdfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelIsReady", true /* optional */,
      reinterpret_cast<void**>(&mirfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelMetadata", true /* optional */,
      reinterpret_cast<void**>(&mmfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ResponseAllocatorNew", true /* optional */,
      reinterpret_cast<void**>(&ranfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestNew", true /* optional */,
      reinterpret_cast<void**>(&irnfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetId", true /* optional */,
      reinterpret_cast<void**>(&irsifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetReleaseCallback",
      true /* optional */, reinterpret_cast<void**>(&irsrcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAddInput", true /* optional */,
      reinterpret_cast<void**>(&iraifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAddRequestedOutput",
      true /* optional */, reinterpret_cast<void**>(&irarofn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAppendInputData",
      true /* optional */, reinterpret_cast<void**>(&iraidfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetResponseCallback",
      true /* optional */, reinterpret_cast<void**>(&irsrescfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerInferAsync", true /* optional */,
      reinterpret_cast<void**>(&iafn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseError", true /* optional */,
      reinterpret_cast<void**>(&irefn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseDelete", true /* optional */,
      reinterpret_cast<void**>(&irdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestRemoveAllInputData",
      true /* optional */, reinterpret_cast<void**>(&irraidfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ResponseAllocatorDelete", true /* optional */,
      reinterpret_cast<void**>(&radfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorNew", true /* optional */,
      reinterpret_cast<void**>(&enfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MemoryTypeString", true /* optional */,
      reinterpret_cast<void**>(&mtsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseOutputCount",
      true /* optional */, reinterpret_cast<void**>(&irocfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_DataTypeString", true /* optional */,
      reinterpret_cast<void**>(&dtsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorMessage", true /* optional */,
      reinterpret_cast<void**>(&emfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorDelete", true /* optional */,
      reinterpret_cast<void**>(&edfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorCodeString", true /* optional */,
      reinterpret_cast<void**>(&ectsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelConfig", true /* optional */,
      reinterpret_cast<void**>(&mcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetCorrelationId",
      true /* optional */, reinterpret_cast<void**>(&scidfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetFlags", true /* optional */,
      reinterpret_cast<void**>(&sffn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetPriority",
      true /* optional */, reinterpret_cast<void**>(&spfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetTimeoutMicroseconds",
      true /* optional */, reinterpret_cast<void**>(&stmsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_StringToDataType", true /* optional */,
      reinterpret_cast<void**>(&stdtfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseOutput", true /* optional */,
      reinterpret_cast<void**>(&irofn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestId", true /* optional */,
      reinterpret_cast<void**>(&ridfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestDelete", true /* optional */,
      reinterpret_cast<void**>(&rdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelStatistics", true /* optional */,
      reinterpret_cast<void**>(&msfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerUnloadModel", true /* optional */,
      reinterpret_cast<void**>(&umfn)));


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
  inference_request_remove_all_input_data_fn_ = irraidfn;
  response_allocator_delete_fn_ = radfn;
  error_new_fn_ = enfn;

  memory_type_string_fn_ = mtsfn;
  inference_response_output_count_fn_ = irocfn;
  data_type_string_fn_ = dtsfn;
  error_message_fn_ = emfn;

  error_delete_fn_ = edfn;
  error_code_to_string_fn_ = ectsfn;
  model_config_fn_ = mcfn;
  set_correlation_id_fn_ = scidfn;

  set_flags_fn_ = sffn;
  set_priority_fn_ = spfn;
  set_timeout_ms_fn_ = stmsfn;
  string_to_datatype_fn_ = stdtfn;

  inference_response_output_fn_ = irofn;
  request_id_fn_ = ridfn;
  request_delete_fn_ = rdfn;
  model_statistics_fn_ = msfn;
  unload_model_fn_ = umfn;

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
  inference_request_remove_all_input_data_fn_ = nullptr;
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

  set_flags_fn_ = nullptr;
  set_priority_fn_ = nullptr;
  set_timeout_ms_fn_ = nullptr;
  string_to_datatype_fn_ = nullptr;

  inference_response_output_fn_ = nullptr;
  request_id_fn_ = nullptr;
  request_delete_fn_ = nullptr;
  model_statistics_fn_ = nullptr;
  unload_model_fn_ = nullptr;
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
    const nic::InferOptions& options,
    const std::vector<nic::InferInput*>& inputs,
    const std::vector<const nic::InferRequestedOutput*>& outputs,
    nic::InferResult** result)
{
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  Timer().Reset();
  Timer().CaptureTimestamp(nic::RequestTimers::Kind::REQUEST_START);
  GetSingleton()->InitializeRequest(options, outputs, &allocator, &irequest);
  GetSingleton()->AddInputs(options, inputs, irequest);
  GetSingleton()->AddOutputs(options, outputs, irequest);
  Timer().CaptureTimestamp(nic::RequestTimers::Kind::SEND_START);
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
  // Wait for the inference to complete.
  TRITONSERVER_InferenceResponse* completed_response = completed.get();
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->inference_response_error_fn_(completed_response),
      "response status");
  // get data out
  std::unordered_map<std::string, std::vector<char>> output_data;
  uint32_t output_count;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->inference_response_output_count_fn_(
          completed_response, &output_count),
      "getting number of response outputs");
  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->inference_response_output_fn_(
            completed_response, idx, &cname, &datatype, &shape, &dim_count,
            &base, &byte_size, &memory_type, &memory_type_id, &userp),
        "getting output info");
    if (cname == nullptr) {
      return Error("unable to get output name");
    }
    std::string name(cname);
    std::vector<char>& odata = output_data[name];
    const char* cbase = reinterpret_cast<const char*>(base);
    odata.assign(cbase, cbase + byte_size);
  }
  Timer().CaptureTimestamp(nic::RequestTimers::Kind::SEND_END);
  Timer().CaptureTimestamp(nic::RequestTimers::Kind::REQUEST_END);
  Timer().CaptureTimestamp(nic::RequestTimers::Kind::RECV_START);
  Timer().CaptureTimestamp(nic::RequestTimers::Kind::RECV_END);

  nic::Error err = GetSingleton()->UpdateInferStat(Timer());
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }
  const char* cid;
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->request_id_fn_(irequest, &cid),
      "Failed to get request id");
  std::string id(cid);
  nic::InferResultCApi::Create(result, err, id);
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
    const nic::InferOptions& options,
    const std::vector<const nic::InferRequestedOutput*>& outputs,
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
  if ((options.sequence_id_ != 0) || (options.priority_ != 0) ||
      (options.server_timeout_ != 0) || outputs.empty()) {
    RETURN_IF_TRITONSERVER_ERROR(
        GetSingleton()->set_correlation_id_fn_(*irequest, options.sequence_id_),
        "setting sequence ID for the request");

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
    const nic::InferOptions& options,
    const std::vector<nic::InferInput*>& inputs,
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
    nic::Error err = io->ByteSize(&byte_size);
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
    const nic::InferOptions& options,
    const std::vector<const nic::InferRequestedOutput*>& outputs,
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
    std::cout << "got inference statistics" << std::endl;
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

}}  // namespace perfanalyzer::clientbackend