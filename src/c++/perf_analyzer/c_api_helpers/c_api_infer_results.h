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
#pragma once
#include "../../library/common.h"
namespace nvidia { namespace inferenceserver { namespace client {
class InferResultCApi : public InferResult {
 public:
  static void Create(
      InferResult** infer_result, const Error& err, const std::string& id);
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
  Error StringData(
      const std::string& output_name,
      std::vector<std::string>* string_result) const override;
  std::string DebugString() const override;
  Error RequestStatus() const override;

 private:
  InferResultCApi(const Error& err, const std::string& id);

  std::string request_id_;
  Error status_;
};

void
InferResultCApi::Create(
    InferResult** infer_result, const Error& err, const std::string& id)
{
  *infer_result = reinterpret_cast<InferResult*>(new InferResultCApi(err, id));
}

Error
InferResultCApi::ModelName(std::string* name) const
{
  return Error("Do not know model name");
}

Error
InferResultCApi::ModelVersion(std::string* version) const
{
  return Error("Do not know model version");
}

Error
InferResultCApi::Id(std::string* id) const
{
  *id = request_id_;
  return Error::Success;
}

Error
InferResultCApi::Shape(
    const std::string& output_name, std::vector<int64_t>* shape) const
{
  return Error("Do not know shape");
}

Error
InferResultCApi::Datatype(
    const std::string& output_name, std::string* datatype) const
{
  return Error("Do not know datatype");
}

Error
InferResultCApi::RawData(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
  return Error("Do not have raw data");
}

Error
InferResultCApi::StringData(
    const std::string& output_name,
    std::vector<std::string>* string_result) const
{
  return Error("Do not have string data");
}
std::string
InferResultCApi::DebugString() const
{
  std::string err = "Does not have debug info";
  return err;
}
Error
InferResultCApi::RequestStatus() const
{
  return status_;
}

InferResultCApi::InferResultCApi(const Error& err, const std::string& id)
{
  status_ = err;
  request_id_ = id;
}
}}}  // namespace nvidia::inferenceserver::client