// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "openai_http_client.h"

#include <rapidjson/rapidjson.h>


namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {


Error
HttpClient::Create(
    std::unique_ptr<HttpClient>* client, const std::string& server_url,
    bool verbose)
{
  client->reset(new HttpClient(server_url, verbose));
  return Error::Success;
}

Error
HttpClient::AsyncInfer(
    OpenAiOnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers)
{
  // TODO FIXME implement

  // TODO FIXME cleanup or remove this. It just proves the json data arrives
  rapidjson::Document d{};

  if (inputs.size() != 1) {
    return Error("Only expecting one input");
  }

  auto raw_input = dynamic_cast<OpenAiInferInput*>(inputs[0]);

  raw_input->PrepareForRequest();
  bool end_of_input = false;
  const uint8_t* buf;
  size_t buf_size;
  raw_input->GetNext(&buf, &buf_size, &end_of_input);
  if (!end_of_input) {
    return Error("Unexpected multiple json data inputs");
  }
  if (buf == nullptr) {
    return Error("Unexpected null json data");
  }

  std::string json_str(reinterpret_cast<const char*>(buf), buf_size);
  std::cout << "FIXME TODO: JSON data string is " << json_str << std::endl;


  if (d.Parse(json_str.c_str()).HasParseError()) {
    return Error("Unable to parse json string: " + json_str);
  }

  // FIXME TKG -- where/how would the 'streaming' option get plugged in?

  // FIXME TKG -- GOOD GOD! Is it this hard to add a single value into a json
  // object??
  // FIXME TKG -- what if the user supplied this in the input json file?
  d.AddMember(
      "model",
      rapidjson::Value().SetString(
          options.model_name_.c_str(),
          static_cast<rapidjson::SizeType>(options.model_name_.length()),
          d.GetAllocator()),
      d.GetAllocator());

  for (auto itr = d.MemberBegin(); itr != d.MemberEnd(); ++itr) {
    std::cout << "FIXME TODO: valid JSON object has key "
              << itr->name.GetString() << std::endl;
  }

  return Error::Success;
}

HttpClient::HttpClient(const std::string& url, bool verbose)
    : InferenceServerClient(verbose), url_(url)
// ,easy_handle_(reinterpret_cast<void*>(curl_easy_init()) // TODO FIXME TKG
{
}

HttpClient::~HttpClient()
{
  exiting_ = true;

  // FIXME TODO TKG
  // if (easy_handle_ != nullptr) {
  //  curl_easy_cleanup(reinterpret_cast<CURL*>(easy_handle_));
  //}
}

}}}}  // namespace triton::perfanalyzer::clientbackend::openai