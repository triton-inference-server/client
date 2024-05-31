#include "http_client.h"

#include <boost/thread.hpp>
#include <boost/unordered/concurrent_flat_map.hpp>
#include <functional>
#include <iostream>

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

HttpRequest::HttpRequest(
    std::function<void(HttpRequest*)>&& completion_callback, const bool verbose)
    : completion_callback_(std::move(completion_callback)), verbose_(verbose)
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
  data_buffers_.emplace_back(buf, byte_size);
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

boost::mutex HttpClient::curl_init_mtx_{};

HttpClient::HttpClient(
    const std::string& server_url, bool verbose,
    const HttpSslOptions& ssl_options)
    : url_(server_url), verbose_(verbose), ssl_options_(ssl_options)
{
  {
    boost::lock_guard<boost::mutex> lk(curl_init_mtx_);
    if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
      throw std::runtime_error("CURL global initialization failed");
    }
  }

  multi_handle_ = curl_multi_init();

  worker_ = boost::thread(&HttpClient::AsyncTransfer, this);
}

HttpClient::~HttpClient()
{
  exiting_ = true;

  if (worker_.joinable()) {
    cv_.notify_all();
    worker_.join();
  }

  ongoing_async_requests_.visit_all([this](auto& request) {
    CURL* easy_handle = reinterpret_cast<CURL*>(request.second.get());
    curl_multi_remove_handle(multi_handle_, easy_handle);
    curl_easy_cleanup(easy_handle);
  });
  curl_multi_cleanup(multi_handle_);

  {
    boost::lock_guard<boost::mutex> lk(curl_init_mtx_);
    curl_global_cleanup();
  }
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
  throw std::runtime_error(
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
  throw std::runtime_error(
      "unsupported SSL key type encountered. Only PEM and DER are "
      "supported.");
}

void
HttpClient::SetSSLCurlOptions(CURL* curl_handle)
{
  curl_easy_setopt(
      curl_handle, CURLOPT_SSL_VERIFYPEER, ssl_options_.verify_peer);
  curl_easy_setopt(
      curl_handle, CURLOPT_SSL_VERIFYHOST, ssl_options_.verify_host);
  if (!ssl_options_.ca_info.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_CAINFO, ssl_options_.ca_info.c_str());
  }
  const auto& curl_cert_type = ParseSslCertType(ssl_options_.cert_type);
  curl_easy_setopt(curl_handle, CURLOPT_SSLCERTTYPE, curl_cert_type.c_str());
  if (!ssl_options_.cert.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_SSLCERT, ssl_options_.cert.c_str());
  }
  const auto& curl_key_type = ParseSslKeyType(ssl_options_.key_type);
  curl_easy_setopt(curl_handle, CURLOPT_SSLKEYTYPE, curl_key_type.c_str());
  if (!ssl_options_.key.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_SSLKEY, ssl_options_.key.c_str());
  }
}

void
HttpClient::Send(CURL* handle, std::unique_ptr<HttpRequest>&& request)
{
  ongoing_async_requests_.emplace(
      reinterpret_cast<uintptr_t>(handle), std::move(request));

  curl_multi_add_handle(multi_handle_, handle);

  cv_.notify_all();

  curl_multi_wakeup(multi_handle_);
}

void
HttpClient::AsyncTransfer()
{
  int place_holder = 0;
  CURLMsg* msg = nullptr;

  while (!exiting_) {
    std::vector<std::unique_ptr<HttpRequest>> request_list;
    std::vector<uintptr_t> identifiers_to_erase;

    {
      boost::unique_lock<boost::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
        return this->exiting_ || !this->ongoing_async_requests_.empty();
      });

      if (exiting_) {
        break;
      }
    }

    CURLMcode mc;
    mc = curl_multi_perform(multi_handle_, &place_holder);

    if (mc != CURLM_OK) {
      std::cerr << "Unexpected error: curl_multi_perform failed. Code:" << mc
                << std::endl;
      continue;
    }

    int numfds;
    mc = curl_multi_poll(multi_handle_, NULL, 0, 1000, &numfds);

    if (mc != CURLM_OK) {
      std::cerr << "Unexpected error: curl_multi_poll failed. Code:" << mc
                << std::endl;
      continue;
    }

    while (true) {
      {
        boost::lock_guard<boost::mutex> lock(mutex_);
        msg = curl_multi_info_read(multi_handle_, &place_holder);
      }

      if (msg == nullptr) {
        break;
      }

      uintptr_t identifier = reinterpret_cast<uintptr_t>(msg->easy_handle);
      ongoing_async_requests_.visit(identifier, [&](auto& request) {
        uint32_t http_code = 400;
        if (msg->data.result == CURLE_OK) {
          curl_easy_getinfo(
              msg->easy_handle, CURLINFO_RESPONSE_CODE, &http_code);
        } else if (msg->data.result == CURLE_OPERATION_TIMEDOUT) {
          http_code = 499;
        }

        request_list.emplace_back(std::move(request.second));
        identifiers_to_erase.push_back(identifier);
        curl_multi_remove_handle(multi_handle_, msg->easy_handle);
        curl_easy_cleanup(msg->easy_handle);

        request_list.back()->http_code_ = http_code;

        if (msg->msg != CURLMSG_DONE) {
          std::cerr << "Unexpected error: received CURLMsg=" << msg->msg
                    << std::endl;
        }
      });
    }

    // Perform bulk erase outside the visit loop to minimize lock contention
    if (!identifiers_to_erase.empty()) {
      boost::lock_guard<boost::mutex> lock(mutex_);
      for (auto identifier : identifiers_to_erase) {
        ongoing_async_requests_.erase(identifier);
      }
    }

    for (auto& this_request : request_list) {
      this_request->completion_callback_(this_request.get());
    }
  }
}

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
