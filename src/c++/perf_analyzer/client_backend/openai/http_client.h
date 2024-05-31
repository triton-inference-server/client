#pragma once

#include <curl/curl.h>

#include <boost/thread.hpp>
#include <boost/unordered/concurrent_flat_map.hpp>
#include <deque>

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

struct HttpSslOptions {
  enum CERTTYPE { CERT_PEM = 0, CERT_DER = 1 };
  enum KEYTYPE { KEY_PEM = 0, KEY_DER = 1 };
  explicit HttpSslOptions()
      : verify_peer(1), verify_host(2), cert_type(CERT_PEM), key_type(KEY_PEM)
  {
  }

  long verify_peer;
  long verify_host;
  std::string ca_info;
  CERTTYPE cert_type;
  std::string cert;
  KEYTYPE key_type;
  std::string key;
};

class HttpRequest {
 public:
  HttpRequest(
      std::function<void(HttpRequest*)>&& completion_callback,
      const bool verbose = false);
  virtual ~HttpRequest();

  void AddInput(uint8_t* buf, size_t byte_size);
  void GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

  std::string response_buffer_;
  size_t total_input_byte_size_{0};
  uint32_t http_code_{200};
  std::function<void(HttpRequest*)> completion_callback_{nullptr};
  struct curl_slist* header_list_{nullptr};

 protected:
  const bool verbose_{false};
  std::deque<std::pair<uint8_t*, size_t>> data_buffers_;
};

class HttpClient {
 public:
  enum class CompressionType { NONE, DEFLATE, GZIP };

  virtual ~HttpClient();

 protected:
  void SetSSLCurlOptions(CURL* curl_handle);

  HttpClient(
      const std::string& server_url, bool verbose = false,
      const HttpSslOptions& ssl_options = HttpSslOptions());

  void Send(CURL* handle, std::unique_ptr<HttpRequest>&& request);

 protected:
  void AsyncTransfer();

  bool exiting_{false};

  boost::thread worker_;
  boost::mutex mutex_;
  boost::condition_variable cv_;

  const std::string url_;
  HttpSslOptions ssl_options_;

  using AsyncReqMap = boost::unordered::concurrent_flat_map<
      uintptr_t, std::unique_ptr<HttpRequest>>;
  void* multi_handle_;
  AsyncReqMap ongoing_async_requests_;

  bool verbose_;

 private:
  const std::string& ParseSslKeyType(HttpSslOptions::KEYTYPE key_type);
  const std::string& ParseSslCertType(HttpSslOptions::CERTTYPE cert_type);
  static boost::mutex curl_init_mtx_;
};
}}}}  // namespace triton::perfanalyzer::clientbackend::openai
