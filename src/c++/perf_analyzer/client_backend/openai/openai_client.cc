// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Include this first to make sure we are a friend of common classes.
#define TRITON_INFERENCE_SERVER_CLIENT_CLASS InferenceServerHttpClient
#include "openai_client.h"

#include <curl/curl.h>

#include <atomic>
#include <climits>
#include <cstdint>
#include <deque>
#include <iostream>
#include <string>
#include <utility>

#include "common.h"

#ifdef TRITON_ENABLE_ZLIB
#include <zlib.h>
#endif

extern "C" {
#include "cencode.h"
}

#define TRITONJSON_STATUSTYPE triton::client::Error
#define TRITONJSON_STATUSRETURN(M) return triton::client::Error(M)
#define TRITONJSON_STATUSSUCCESS triton::client::Error::Success
#include "triton/common/triton_json.h"

#ifdef _WIN32
#define strncasecmp(x, y, z) _strnicmp(x, y, z)
#undef min  // NOMINMAX did not resolve std::min compile error
#endif      //_WIN32

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {
namespace {

constexpr char kContentLengthHTTPHeader[] = "Content-Length";

HttpClient::HttpClient(
    const std::string& server_url, bool verbose,
    const HttpSslOptions& ssl_options)
    : url_(server_url), verbose_(verbose), ssl_options_(ssl_options)
{
  auto* ver = curl_version_info(CURLVERSION_NOW);
  if (ver->features & CURL_VERSION_THREADSAFE == 0) {
    throw std::exception(
        "HTTP client has dependency on CURL library to have thread-safe "
        "support (CURL_VERSION_THREADSAFE set)");
  }
  if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
    throw std::exception("CURL global initialization failed");
  }
}

HttpClient::~HttpClient()
{
  curl_global_cleanup();
}

const std::string&
HttpClient::ParseSslCertType(HttpSslOptions::CERTTYPE cert_type)
{
  static std::string pem_str{"PEM"};
  static std::string der_str{"DER"};
  switch (cert_type) {
    case HttpSslOptions::CERTTYPE::CERT_PEM:
      return pem_str;
    case HttpSslOptions::CERTTYPE::CERT_DER:
      return der_str;
  }
  throw std::exception(
      "Unexpected SSL certificate type encountered. Only PEM and DER are "
      "supported.");
}

const std::string&
HttpClient::ParseSslKeyType(HttpSslOptions::KEYTYPE key_type)
{
  static std::string pem_str{"PEM"};
  static std::string der_str{"DER"};
  switch (key_type) {
    case HttpSslOptions::KEYTYPE::KEY_PEM:
      return pem_str;
    case HttpSslOptions::KEYTYPE::KEY_DER:
      return der_str;
  }
  throw std::exception(
      "unsupported SSL key type encountered. Only PEM and DER are "
      "supported.");
}

void
HttpClient::SetSSLCurlOptions(CURL** curl_handle)
{
  curl_easy_setopt(*curl, CURLOPT_SSL_VERIFYPEER, ssl_options_.verify_peer);
  curl_easy_setopt(*curl, CURLOPT_SSL_VERIFYHOST, ssl_options_.verify_host);
  if (!ssl_options_.ca_info.empty()) {
    curl_easy_setopt(*curl, CURLOPT_CAINFO, ssl_options_.ca_info.c_str());
  }
  const auto& curl_cert_type = ParseSslCertType(ssl_options_.cert_type);
  curl_easy_setopt(*curl, CURLOPT_SSLCERTTYPE, curl_cert_type.c_str());
  if (!ssl_options_.cert.empty()) {
    curl_easy_setopt(*curl, CURLOPT_SSLCERT, ssl_options_.cert.c_str());
  }
  const auto& curl_key_type = ParseSslKeyType(ssl_options_.key_type);
  curl_easy_setopt(*curl, CURLOPT_SSLKEYTYPE, curl_key_type.c_str());
  if (!ssl_options_.key.empty()) {
    curl_easy_setopt(*curl, CURLOPT_SSLKEY, ssl_options_.key.c_str());
  }
}

}  // namespace

//==============================================================================


HttpRequest::HttpRequest(
    std::function<void(HttpRequest::HttpRequest*)>&& response_callback,
    const bool verbose)
    : response_callback_(std::move(response_callback)), verbose_(verbose)
{
}

HttpRequest::~HttpRequest()
{
  if (header_list_ != nullptr) {
    curl_slist_free_all(header_list_);
    header_list_ = nullptr;
  }
}

void
HttpRequest::AddInput(uint8_t* buf, size_t byte_size)
{
  data_buffers_.push_back(std::pair<uint8_t*, size_t>(buf, byte_size));
  total_input_byte_size_ += byte_size;
}

void
HttpRequest::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  while (!data_buffers_.empty() && size > 0) {
    const size_t csz = std::min(data_buffers_.front().second, size);
    if (csz > 0) {
      const uint8_t* input_ptr = data_buffers_.front().first;
      std::copy(input_ptr, input_ptr + csz, buf);
      size -= csz;
      buf += csz;
      *input_bytes += csz;


      data_buffers_.front().first += csz;
      data_buffers_.front().second -= csz;
    }
    if (data_buffers_.front().second == 0) {
      data_buffers_.pop_front();
    }
  }
}

//==============================================================================

class InferResultHttp : public InferResult {
 public:
  static void Create(
      InferResult** infer_result,
      std::shared_ptr<HttpInferRequest> infer_request);

  static Error Create(InferResult** infer_result, const Error err);

  Error RequestStatus() const override;
  Error ModelName(std::string* name) const override;
  Error ModelVersion(std::string* version) const override;
  Error Id(std::string* id) const override;
  Error Shape(const std::string& output_name, std::vector<int64_t>* shape)
      const override;
  Error Datatype(
      const std::string& output_name, std::string* datatype) const override;
  Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override;
  Error IsFinalResponse(bool* is_final_response) const override;
  Error IsNullResponse(bool* is_null_response) const override;
  Error StringData(
      const std::string& output_name,
      std::vector<std::string>* string_result) const override;
  std::string DebugString() const override;

 private:
  InferResultHttp(std::shared_ptr<HttpInferRequest> infer_request);
  InferResultHttp(const Error err) : status_(err) {}

 protected:
  InferResultHttp() {}
  ~InferResultHttp();

 private:
  std::map<std::string, triton::common::TritonJson::Value>
      output_name_to_result_map_;
  std::map<std::string, std::pair<const uint8_t*, const size_t>>
      output_name_to_buffer_map_;

  Error status_;
  triton::common::TritonJson::Value response_json_;
  std::shared_ptr<HttpInferRequest> infer_request_;

  bool binary_data_{true};
  bool is_final_response_{true};
  bool is_null_response_{false};
};

void
InferResultHttp::Create(
    InferResult** infer_result, std::shared_ptr<HttpInferRequest> infer_request)
{
  *infer_result =
      reinterpret_cast<InferResult*>(new InferResultHttp(infer_request));
}

Error
InferResultHttp::Create(InferResult** infer_result, const Error err)
{
  if (err.IsOk()) {
    return Error(
        "Error is not provided for error reporting override of "
        "InferResultHttp::Create()");
  }
  *infer_result = reinterpret_cast<InferResult*>(new InferResultHttp(err));
  return Error::Success;
}

Error
InferResultHttp::ModelName(std::string* name) const
{
  if (!status_.IsOk()) {
    return status_;
  }

  const char* name_str;
  size_t name_strlen;
  Error err =
      response_json_.MemberAsString("model_name", &name_str, &name_strlen);
  if (!err.IsOk()) {
    return Error("model name was not returned in the response");
  }

  name->assign(name_str, name_strlen);
  return Error::Success;
}

Error
InferResultHttp::ModelVersion(std::string* version) const
{
  if (!status_.IsOk()) {
    return status_;
  }

  const char* version_str;
  size_t version_strlen;
  Error err = response_json_.MemberAsString(
      "model_version", &version_str, &version_strlen);
  if (!err.IsOk()) {
    return Error("model version was not returned in the response");
  }

  version->assign(version_str, version_strlen);
  return Error::Success;
}

Error
InferResultHttp::Id(std::string* id) const
{
  if (!status_.IsOk()) {
    return status_;
  }

  const char* id_str;
  size_t id_strlen;
  Error err = response_json_.MemberAsString("id", &id_str, &id_strlen);
  if (!err.IsOk()) {
    return Error("model id was not returned in the response");
  }

  id->assign(id_str, id_strlen);
  return Error::Success;
}

namespace {

Error
ShapeHelper(
    const std::string& result_name,
    const triton::common::TritonJson::Value& result_json,
    std::vector<int64_t>* shape)
{
  triton::common::TritonJson::Value shape_json;
  if (!const_cast<triton::common::TritonJson::Value&>(result_json)
           .Find("shape", &shape_json)) {
    return Error(
        "The response does not contain shape for output name " + result_name);
  }

  for (size_t i = 0; i < shape_json.ArraySize(); i++) {
    int64_t dim;
    Error err = shape_json.IndexAsInt(i, &dim);
    if (!err.IsOk()) {
      return err;
    }

    shape->push_back(dim);
  }

  return Error::Success;
}

}  // namespace

Error
InferResultHttp::Shape(
    const std::string& output_name, std::vector<int64_t>* shape) const
{
  if (!status_.IsOk()) {
    return status_;
  }

  shape->clear();
  auto itr = output_name_to_result_map_.find(output_name);
  if (itr == output_name_to_result_map_.end()) {
    return Error(
        "The response does not contain results for output name " + output_name);
  }

  return ShapeHelper(output_name, itr->second, shape);
}

Error
InferResultHttp::Datatype(
    const std::string& output_name, std::string* datatype) const
{
  if (!status_.IsOk()) {
    return status_;
  }
  auto itr = output_name_to_result_map_.find(output_name);
  if (itr == output_name_to_result_map_.end()) {
    return Error(
        "The response does not contain results for output name " + output_name);
  }

  const char* dtype_str;
  size_t dtype_strlen;
  Error err = itr->second.MemberAsString("datatype", &dtype_str, &dtype_strlen);
  if (!err.IsOk()) {
    return Error(
        "The response does not contain datatype for output name " +
        output_name);
  }

  datatype->assign(dtype_str, dtype_strlen);
  return Error::Success;
}

Error
InferResultHttp::RawData(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
  if (!status_.IsOk()) {
    return status_;
  }
  auto itr = output_name_to_buffer_map_.find(output_name);
  if (itr != output_name_to_buffer_map_.end()) {
    *buf = itr->second.first;
    *byte_size = itr->second.second;
  } else {
    return Error(
        "The response does not contain results for output name " + output_name);
  }

  return Error::Success;
}

Error
InferResultHttp::IsFinalResponse(bool* is_final_response) const
{
  if (is_final_response == nullptr) {
    return Error("is_final_response cannot be nullptr");
  }
  *is_final_response = is_final_response_;
  return Error::Success;
}

Error
InferResultHttp::IsNullResponse(bool* is_null_response) const
{
  if (is_null_response == nullptr) {
    return Error("is_null_response cannot be nullptr");
  }
  *is_null_response = is_null_response_;
  return Error::Success;
}

Error
InferResultHttp::StringData(
    const std::string& output_name,
    std::vector<std::string>* string_result) const
{
  if (!status_.IsOk()) {
    return status_;
  }
  std::string datatype;
  Error err = Datatype(output_name, &datatype);
  if (!err.IsOk()) {
    return err;
  }
  if (datatype.compare("BYTES") != 0) {
    return Error(
        "This function supports tensors with datatype 'BYTES', requested "
        "output tensor '" +
        output_name + "' with datatype '" + datatype + "'");
  }

  const uint8_t* buf;
  size_t byte_size;
  err = RawData(output_name, &buf, &byte_size);
  string_result->clear();
  size_t buf_offset = 0;
  while (byte_size > buf_offset) {
    const uint32_t element_size =
        *(reinterpret_cast<const uint32_t*>(buf + buf_offset));
    string_result->emplace_back(
        reinterpret_cast<const char*>(buf + buf_offset + sizeof(element_size)),
        element_size);
    buf_offset += (sizeof(element_size) + element_size);
  }

  return Error::Success;
}

std::string
InferResultHttp::DebugString() const
{
  if (!status_.IsOk()) {
    return status_.Message();
  }

  triton::common::TritonJson::WriteBuffer buffer;
  Error err = response_json_.Write(&buffer);
  if (!err.IsOk()) {
    return "<failed>";
  }

  return buffer.Contents();
}

Error
InferResultHttp::RequestStatus() const
{
  return status_;
}

InferResultHttp::InferResultHttp(
    std::shared_ptr<HttpInferRequest> infer_request)
    : infer_request_(infer_request)
{
  size_t offset = infer_request->response_json_size_;
  if (infer_request->http_code_ == 499) {
    status_ = Error("Deadline Exceeded");
  } else {
    if (offset != 0) {
      if (infer_request->verbose_) {
        std::cout << "inference response: "
                  << infer_request->infer_response_buffer_->substr(0, offset)
                  << std::endl;
      }
      status_ = response_json_.Parse(
          (char*)infer_request->infer_response_buffer_.get()->c_str(), offset);
    } else {
      if (infer_request->verbose_) {
        std::cout << "inference response: "
                  << *infer_request->infer_response_buffer_ << std::endl;
      }
      status_ = response_json_.Parse(
          (char*)infer_request->infer_response_buffer_.get()->c_str());
    }
  }

  // There should be a valid JSON response in all cases. Either the
  // successful infer response or an error response.
  if (status_.IsOk()) {
    if (infer_request->http_code_ != 200) {
      const char* err_str;
      size_t err_strlen;
      if (!response_json_.MemberAsString("error", &err_str, &err_strlen)
               .IsOk()) {
        status_ = Error("inference failed with unknown error");
      } else {
        status_ = Error(std::string(err_str, err_strlen));
      }
    } else {
      triton::common::TritonJson::Value outputs_json;
      if (response_json_.Find("outputs", &outputs_json)) {
        for (size_t i = 0; i < outputs_json.ArraySize(); i++) {
          triton::common::TritonJson::Value output_json;
          status_ = outputs_json.IndexAsObject(i, &output_json);
          if (!status_.IsOk()) {
            break;
          }

          const char* name_str;
          size_t name_strlen;
          status_ = output_json.MemberAsString("name", &name_str, &name_strlen);
          if (!status_.IsOk()) {
            break;
          }

          std::string output_name(name_str, name_strlen);

          triton::common::TritonJson::Value param_json, data_json;
          if (output_json.Find("parameters", &param_json)) {
            uint64_t data_size = 0;
            status_ = param_json.MemberAsUInt("binary_data_size", &data_size);
            if (!status_.IsOk()) {
              break;
            }

            output_name_to_buffer_map_.emplace(
                output_name,
                std::pair<const uint8_t*, const size_t>(
                    (uint8_t*)(infer_request->infer_response_buffer_.get()
                                   ->c_str()) +
                        offset,
                    data_size));
            offset += data_size;
          } else if (output_json.Find("data", &data_json)) {
            binary_data_ = false;
            std::string datatype;
            status_ = output_json.MemberAsString("datatype", &datatype);
            if (!status_.IsOk()) {
              break;
            }

            const uint8_t* buf{nullptr};
            size_t buf_size{0};
            status_ =
                ConvertJSONOutputToBinary(data_json, datatype, &buf, &buf_size);
            if (!status_.IsOk()) {
              break;
            }

            output_name_to_buffer_map_.emplace(
                output_name,
                std::pair<const uint8_t*, const size_t>(buf, buf_size));
          }

          output_name_to_result_map_[output_name] = std::move(output_json);
        }
      }
    }
  }
}

InferResultHttp::~InferResultHttp()
{
  if (binary_data_) {
    return;
  }

  for (auto& buf_pair : output_name_to_buffer_map_) {
    const uint8_t* buf{buf_pair.second.first};
    delete buf;
  }
}

//==============================================================================

Error
InferenceServerHttpClient::Create(
    std::unique_ptr<InferenceServerHttpClient>* client,
    const std::string& server_url, bool verbose,
    const HttpSslOptions& ssl_options)
{
  client->reset(
      new InferenceServerHttpClient(server_url, verbose, ssl_options));
  return Error::Success;
}

InferenceServerHttpClient::InferenceServerHttpClient(
    const std::string& url, bool verbose, const HttpSslOptions& ssl_options)
    : InferenceServerClient(verbose), url_(url), ssl_options_(ssl_options),
      multi_handle_(curl_multi_init())
{
}

InferenceServerHttpClient::~InferenceServerHttpClient()
{
  exiting_ = true;

  // thread not joinable if AsyncInfer() is not called
  // (it is default constructed thread before the first AsyncInfer() call)
  if (worker_.joinable()) {
    cv_.notify_all();
    worker_.join();
  }

  if (multi_handle_ != nullptr) {
    for (auto& request : ongoing_async_requests_) {
      CURL* easy_handle = reinterpret_cast<CURL*>(request.first);
      curl_multi_remove_handle(multi_handle_, easy_handle);
      curl_easy_cleanup(easy_handle);
    }
    curl_multi_cleanup(multi_handle_);
  }
}

Error
InferenceServerHttpClient::AsyncInfer(
    std::function<void(std::unique_ptr<Result>&&)> callback,
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers)
{
  if (callback == nullptr) {
    return Error(
        "Callback function must be provided along with AsyncInfer() call.");
  }

  std::shared_ptr<HttpInferRequest> async_request;
  if (!multi_handle_) {
    return Error("failed to start HTTP asynchronous client");
  } else if (!worker_.joinable()) {
    worker_ = std::thread(&InferenceServerHttpClient::AsyncTransfer, this);
  }

  std::string request_uri(url_ + "/v2/models/" + options.model_name_);
  if (!options.model_version_.empty()) {
    request_uri = request_uri + "/versions/" + options.model_version_;
  }
  request_uri = request_uri + "/infer";

  HttpInferRequest* raw_async_request =
      new HttpInferRequest(std::move(callback), verbose_);
  async_request.reset(raw_async_request);

  async_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_START);

  CURL* multi_easy_handle = curl_easy_init();
  Error err = PreRunProcessing(
      reinterpret_cast<void*>(multi_easy_handle), request_uri, options, inputs,
      outputs, headers, query_params, request_compression_algorithm,
      response_compression_algorithm, async_request);
  if (!err.IsOk()) {
    curl_easy_cleanup(multi_easy_handle);
    return err;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto insert_result = ongoing_async_requests_.emplace(std::make_pair(
        reinterpret_cast<uintptr_t>(multi_easy_handle), async_request));
    if (!insert_result.second) {
      curl_easy_cleanup(multi_easy_handle);
      return Error("Failed to insert new asynchronous request context.");
    }

    async_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_START);
    if (async_request->total_input_byte_size_ == 0) {
      // Set SEND_END here because CURLOPT_READFUNCTION will not be called if
      // content length is 0. In that case, we can't measure SEND_END properly
      // (send ends after sending request header).
      async_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_END);
    }

    curl_multi_add_handle(multi_handle_, multi_easy_handle);
  }

  cv_.notify_all();
  return Error::Success;
}

size_t
InferenceServerHttpClient::InferRequestProvider(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  size_t input_bytes = 0;
  Error err = request->GetNextInput(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &input_bytes);
  if (!err.IsOk()) {
    std::cerr << "RequestProvider: " << err << std::endl;
    return CURL_READFUNC_ABORT;
  }

  return input_bytes;
}

size_t
InferenceServerHttpClient::InferResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kInferHeaderContentLengthHTTPHeader);
  size_t length_idx = strlen(kContentLengthHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kInferHeaderContentLengthHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      request->response_json_size_ = std::stoi(hdr);
    }
  } else if (
      (length_idx < byte_size) &&
      !strncasecmp(buf, kContentLengthHTTPHeader, length_idx)) {
    while ((length_idx < byte_size) && (buf[length_idx] != ':')) {
      ++length_idx;
    }

    if (length_idx < byte_size) {
      std::string hdr(buf + length_idx + 1, byte_size - length_idx - 1);
      request->infer_response_buffer_->reserve(std::stoi(hdr));
    }
  }

  return byte_size;
}

size_t
InferenceServerHttpClient::InferResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  if (request->Timer().Timestamp(RequestTimers::Kind::RECV_START) == 0) {
    request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_START);
  }

  char* buf = reinterpret_cast<char*>(contents);
  size_t result_bytes = size * nmemb;
  request->infer_response_buffer_->append(buf, result_bytes);

  // InferResponseHandler may be called multiple times so we overwrite
  // RECV_END so that we always have the time of the last.
  request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_END);

  return result_bytes;
}

Error
InferenceServerHttpClient::PreRunProcessing(
    void* vcurl, std::string& request_uri, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, std::shared_ptr<HttpInferRequest>& http_request)
{
  CURL* curl = reinterpret_cast<CURL*>(vcurl);

  // Prepare the request object to provide the data for inference.
  Error err = http_request->InitializeRequest(options, inputs, outputs);
  if (!err.IsOk()) {
    return err;
  }

  // Add the buffers holding input tensor data
  bool all_inputs_are_json{true};
  for (const auto this_input : inputs) {
    if (this_input->BinaryData()) {
      all_inputs_are_json = false;
    }

    if (!this_input->IsSharedMemory() && this_input->BinaryData()) {
      this_input->PrepareForRequest();
      bool end_of_input = false;
      while (!end_of_input) {
        const uint8_t* buf;
        size_t buf_size;
        this_input->GetNext(&buf, &buf_size, &end_of_input);
        if (buf != nullptr) {
          http_request->AddInput(const_cast<uint8_t*>(buf), buf_size);
        }
      }
    }
  }

  // Prepare curl

  curl_easy_setopt(curl, CURLOPT_URL, request_uri.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);

  if (options.client_timeout_ != 0) {
    uint64_t timeout_ms = (options.client_timeout_ / 1000);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
  }

  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  const long buffer_byte_size = 16 * 1024 * 1024;
  curl_easy_setopt(curl, CURLOPT_UPLOAD_BUFFERSIZE, buffer_byte_size);
  curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, buffer_byte_size);

  // request data provided by InferRequestProvider()
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, InferRequestProvider);
  curl_easy_setopt(curl, CURLOPT_READDATA, http_request.get());

  // response headers handled by InferResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, InferResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, http_request.get());

  // response data handled by InferResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, InferResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, http_request.get());

  const curl_off_t post_byte_size = http_request->total_input_byte_size_;
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, post_byte_size);

  SetSSLCurlOptions(&curl);

  struct curl_slist* list = nullptr;

  std::string infer_hdr{
      std::string(kInferHeaderContentLengthHTTPHeader) + ": " +
      std::to_string(http_request->request_json_.Size())};
  list = curl_slist_append(list, infer_hdr.c_str());
  list = curl_slist_append(list, "Expect:");
  if (all_inputs_are_json) {
    list = curl_slist_append(list, "Content-Type: application/json");
  } else {
    list = curl_slist_append(list, "Content-Type: application/octet-stream");
  }

  for (const auto& pr : headers) {
    std::string hdr = pr.first + ": " + pr.second;
    list = curl_slist_append(list, hdr.c_str());
  }

  // Compress data if requested
  switch (request_compression_algorithm) {
    case CompressionType::NONE:
      break;
    case CompressionType::DEFLATE:
      list = curl_slist_append(list, "Content-Encoding: deflate");
      break;
    case CompressionType::GZIP:
      list = curl_slist_append(list, "Content-Encoding: gzip");
      break;
  }
  switch (response_compression_algorithm) {
    case CompressionType::NONE:
      break;
    case CompressionType::DEFLATE:
      curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "deflate");
      break;
    case CompressionType::GZIP:
      curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "gzip");
      break;
  }
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

  // The list will be freed when the request is destructed
  http_request->header_list_ = list;

  if (verbose_) {
    std::cout << "inference request: " << http_request->request_json_.Contents()
              << std::endl;
  }

  return Error::Success;
}

void
InferenceServerHttpClient::AsyncTransfer()
{
  int place_holder = 0;
  CURLMsg* msg = nullptr;
  do {
    std::vector<std::shared_ptr<HttpInferRequest>> request_list;

    // sleep if no work is available
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      if (this->exiting_) {
        return true;
      }
      // wake up if an async request has been generated
      return !this->ongoing_async_requests_.empty();
    });

    CURLMcode mc = curl_multi_perform(multi_handle_, &place_holder);
    int numfds;
    if (mc == CURLM_OK) {
      // Wait for activity. If there are no descriptors in the multi_handle_
      // then curl_multi_wait will return immediately
      mc = curl_multi_wait(multi_handle_, NULL, 0, INT_MAX, &numfds);
      if (mc == CURLM_OK) {
        while ((msg = curl_multi_info_read(multi_handle_, &place_holder))) {
          uintptr_t identifier = reinterpret_cast<uintptr_t>(msg->easy_handle);
          auto itr = ongoing_async_requests_.find(identifier);
          // This shouldn't happen
          if (itr == ongoing_async_requests_.end()) {
            std::cerr
                << "Unexpected error: received completed request that is not "
                   "in the list of asynchronous requests"
                << std::endl;
            curl_multi_remove_handle(multi_handle_, msg->easy_handle);
            curl_easy_cleanup(msg->easy_handle);
            continue;
          }

          long http_code = 400;
          if (msg->data.result == CURLE_OK) {
            curl_easy_getinfo(
                msg->easy_handle, CURLINFO_RESPONSE_CODE, &http_code);
          } else if (msg->data.result == CURLE_OPERATION_TIMEDOUT) {
            http_code = 499;
          }

          request_list.emplace_back(itr->second);
          ongoing_async_requests_.erase(itr);
          curl_multi_remove_handle(multi_handle_, msg->easy_handle);
          curl_easy_cleanup(msg->easy_handle);

          std::shared_ptr<HttpInferRequest> async_request = request_list.back();
          async_request->http_code_ = http_code;

          if (msg->msg != CURLMSG_DONE) {
            // Something wrong happened.
            std::cerr << "Unexpected error: received CURLMsg=" << msg->msg
                      << std::endl;
          } else {
            async_request->Timer().CaptureTimestamp(
                RequestTimers::Kind::REQUEST_END);
            Error err = UpdateInferStat(async_request->Timer());
            if (!err.IsOk()) {
              std::cerr << "Failed to update context stat: " << err
                        << std::endl;
            }
          }
        }
      } else {
        std::cerr << "Unexpected error: curl_multi failed. Code:" << mc
                  << std::endl;
      }
    } else {
      std::cerr << "Unexpected error: curl_multi failed. Code:" << mc
                << std::endl;
    }
    lock.unlock();

    for (auto& this_request : request_list) {
      InferResult* result;
      InferResultHttp::Create(&result, this_request);
      this_request->callback_(result);
    }
  } while (!exiting_);
}

//==============================================================================

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
