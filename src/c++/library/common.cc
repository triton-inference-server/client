// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common.h"

#include <numeric>

#include "triton/common/model_config.h"

namespace triton { namespace client {

//==============================================================================

const Error Error::Success("");

Error::Error(const std::string& msg) : msg_(msg) {}

std::ostream&
operator<<(std::ostream& out, const Error& err)
{
  if (!err.msg_.empty()) {
    out << err.msg_;
  }
  return out;
}

//==============================================================================

Error
InferenceServerClient::ClientInferStat(InferStat* infer_stat) const
{
  *infer_stat = infer_stat_;
  return Error::Success;
}

Error
InferenceServerClient::UpdateInferStat(const RequestTimers& timer)
{
  const uint64_t request_time_ns = timer.Duration(
      RequestTimers::Kind::REQUEST_START, RequestTimers::Kind::REQUEST_END);
  const uint64_t send_time_ns = timer.Duration(
      RequestTimers::Kind::SEND_START, RequestTimers::Kind::SEND_END);
  const uint64_t recv_time_ns = timer.Duration(
      RequestTimers::Kind::RECV_START, RequestTimers::Kind::RECV_END);

  if ((request_time_ns == std::numeric_limits<uint64_t>::max()) ||
      (send_time_ns == std::numeric_limits<uint64_t>::max()) ||
      (recv_time_ns == std::numeric_limits<uint64_t>::max())) {
    return Error(
        "Timer not set correctly." +
        ((timer.Timestamp(RequestTimers::Kind::REQUEST_START) >
          timer.Timestamp(RequestTimers::Kind::REQUEST_END))
             ? (" Request time from " +
                std::to_string(
                    timer.Timestamp(RequestTimers::Kind::REQUEST_START)) +
                " to " +
                std::to_string(
                    timer.Timestamp(RequestTimers::Kind::REQUEST_END)) +
                ".")
             : "") +
        ((timer.Timestamp(RequestTimers::Kind::SEND_START) >
          timer.Timestamp(RequestTimers::Kind::SEND_END))
             ? (" Send time from " +
                std::to_string(
                    timer.Timestamp(RequestTimers::Kind::SEND_START)) +
                " to " +
                std::to_string(timer.Timestamp(RequestTimers::Kind::SEND_END)) +
                ".")
             : "") +
        ((timer.Timestamp(RequestTimers::Kind::RECV_START) >
          timer.Timestamp(RequestTimers::Kind::RECV_END))
             ? (" Receive time from " +
                std::to_string(
                    timer.Timestamp(RequestTimers::Kind::RECV_START)) +
                " to " +
                std::to_string(timer.Timestamp(RequestTimers::Kind::RECV_END)) +
                ".")
             : ""));
  }

  infer_stat_.completed_request_count++;
  infer_stat_.cumulative_total_request_time_ns += request_time_ns;
  infer_stat_.cumulative_send_time_ns += send_time_ns;
  infer_stat_.cumulative_receive_time_ns += recv_time_ns;

  return Error::Success;
}

//==============================================================================

Error
InferInput::Create(
    InferInput** infer_input, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  *infer_input = new InferInput(name, dims, datatype);
  return Error::Success;
}

Error
InferInput::SetShape(const std::vector<int64_t>& shape)
{
  shape_ = shape;
  return Error::Success;
}

Error
InferInput::Reset()
{
  bufs_.clear();
  buf_byte_sizes_.clear();
  str_bufs_.clear();
  bufs_idx_ = 0;
  byte_size_ = 0;
  io_type_ = NONE;
  return Error::Success;
}

Error
InferInput::AppendRaw(const std::vector<uint8_t>& input)
{
  return AppendRaw(&input[0], input.size());
}

Error
InferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  byte_size_ += input_byte_size;

  bufs_.push_back(input);
  buf_byte_sizes_.push_back(input_byte_size);
  io_type_ = RAW;

  return Error::Success;
}

Error
InferInput::SetSharedMemory(
    const std::string& name, size_t byte_size, size_t offset)
{
  shm_name_ = name;
  shm_offset_ = offset;
  byte_size_ = byte_size;
  io_type_ = SHARED_MEMORY;

  return Error::Success;
}

Error
InferInput::AppendFromString(const std::vector<std::string>& input)
{
  // Serialize the strings into a "raw" buffer. The first 4-bytes are
  // the length of the string length. Next are the actual string
  // characters. There is *not* a null-terminator on the string.
  str_bufs_.emplace_back();
  std::string& sbuf = str_bufs_.back();
  for (const auto& str : input) {
    uint32_t len = str.size();
    sbuf.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    sbuf.append(str);
  }

  return AppendRaw(reinterpret_cast<const uint8_t*>(&sbuf[0]), sbuf.size());
}

Error
InferInput::RawData(const uint8_t** buf, size_t* byte_size)
{
  if (bufs_.size()) {
    // TMA-1775 - handle multi-batch case
    *buf = bufs_[0];
    *byte_size = buf_byte_sizes_[0];
  } else {
    *buf = nullptr;
    *byte_size = 0;
  }
  return Error::Success;
}

Error
InferInput::ByteSize(size_t* byte_size) const
{
  *byte_size = byte_size_;
  return Error::Success;
}

InferInput::InferInput(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::string& datatype)
    : name_(name), shape_(shape), datatype_(datatype), byte_size_(0),
      bufs_idx_(0), buf_pos_(0), io_type_(NONE), shm_name_(""), shm_offset_(0)
{
}

Error
InferInput::SharedMemoryInfo(
    std::string* name, size_t* byte_size, size_t* offset) const
{
  if (io_type_ != SHARED_MEMORY) {
    return Error("The input has not been set with the shared memory.");
  }
  *name = shm_name_;
  *offset = shm_offset_;
  *byte_size = byte_size_;

  return Error::Success;
}

Error
InferInput::SetBinaryData(const bool binary_data)
{
  binary_data_ = binary_data;
  return Error::Success;
}

Error
InferInput::GetStringCount(size_t* str_cnt) const
{
  int64_t str_checked = 0;
  size_t remaining_str_size = 0;

  size_t next_buf_idx = 0;
  const size_t buf_cnt = bufs_.size();

  const uint8_t* buf = nullptr;
  size_t remaining_buf_size = 0;

  // Validate elements until all buffers have been fully processed.
  while (remaining_buf_size || next_buf_idx < buf_cnt) {
    // Get the next buf if not currently processing one.
    if (!remaining_buf_size) {
      // Reset remaining buf size and pointers for next buf.
      buf = bufs_[next_buf_idx];
      remaining_buf_size = buf_byte_sizes_[next_buf_idx];
      next_buf_idx++;
    }

    constexpr size_t kStringSizeIndicator = sizeof(uint32_t);
    // Get the next element if not currently processing one.
    if (!remaining_str_size) {
      // FIXME: Assume the string element's byte size indicator is not spread
      // across buf boundaries for simplicity. Also needs better log msg.
      if (remaining_buf_size < kStringSizeIndicator) {
        return Error("element byte size indicator exceeds the end of the buf.");
      }

      // Start the next element and reset the remaining element size.
      remaining_str_size = *(reinterpret_cast<const uint32_t*>(buf));
      str_checked++;

      // Advance pointer and remainder by the indicator size.
      buf += kStringSizeIndicator;
      remaining_buf_size -= kStringSizeIndicator;
    }

    // If the remaining buf fits it: consume the rest of the element, proceed
    // to the next element.
    if (remaining_buf_size >= remaining_str_size) {
      buf += remaining_str_size;
      remaining_buf_size -= remaining_str_size;
      remaining_str_size = 0;
    }
    // Otherwise the remaining element is larger: consume the rest of the
    // buf, proceed to the next buf.
    else {
      remaining_str_size -= remaining_buf_size;
      remaining_buf_size = 0;
    }
  }

  // FIXME: If more than expected, should stop earlier
  // Validate the number of processed elements exactly match expectations.
  *str_cnt = str_checked;
  return Error::Success;
}

Error
InferInput::ValidateData() const
{
  inference::DataType datatype =
      triton::common::ProtocolStringToDataType(datatype_);
  if (io_type_ == SHARED_MEMORY) {
    if (datatype == inference::DataType::TYPE_STRING) {
      // TODO Didn't find any shm and BYTES inputs inference example
    } else {
      int64_t expected_byte_size =
          triton::common::GetByteSize(datatype, shape_);
      if ((int64_t)byte_size_ != expected_byte_size) {
        return Error(
            "'" + name_ + "' got unexpected byte size " +
            std::to_string(byte_size_) + ", expected " +
            std::to_string(expected_byte_size));
      }
    }
  } else {
    if (datatype == inference::DataType::TYPE_STRING) {
      int64_t expected_str_cnt = triton::common::GetElementCount(shape_);
      size_t str_cnt;
      Error err = GetStringCount(&str_cnt);
      if (!err.IsOk()) {
        return err;
      }
      if ((int64_t)str_cnt != expected_str_cnt) {
        return Error(
            "'" + name_ + "' got unexpected string count " +
            std::to_string(str_cnt) + ", expected " +
            std::to_string(expected_str_cnt));
      }
    } else {
      int64_t expected_byte_size =
          triton::common::GetByteSize(datatype, shape_);
      if ((int64_t)byte_size_ != expected_byte_size) {
        return Error(
            "'" + name_ + "' got unexpected byte size " +
            std::to_string(byte_size_) + ", expected " +
            std::to_string(expected_byte_size));
      }
    }
  }
  return Error::Success;
}

Error
InferInput::PrepareForRequest()
{
  // Reset position so request sends entire input.
  bufs_idx_ = 0;
  buf_pos_ = 0;
  return Error::Success;
}

Error
InferInput::GetNext(
    uint8_t* buf, size_t size, size_t* input_bytes, bool* end_of_input)
{
  size_t total_size = 0;

  while ((bufs_idx_ < bufs_.size()) && (size > 0)) {
    const size_t buf_byte_size = buf_byte_sizes_[bufs_idx_];
    const size_t csz = (std::min)(buf_byte_size - buf_pos_, size);
    if (csz > 0) {
      const uint8_t* input_ptr = bufs_[bufs_idx_] + buf_pos_;
      std::copy(input_ptr, input_ptr + csz, buf);
      buf_pos_ += csz;
      buf += csz;
      size -= csz;
      total_size += csz;
    }

    if (buf_pos_ == buf_byte_size) {
      bufs_idx_++;
      buf_pos_ = 0;
    }
  }

  *input_bytes = total_size;
  *end_of_input = (bufs_idx_ >= bufs_.size());

  return Error::Success;
}

Error
InferInput::GetNext(
    const uint8_t** buf, size_t* input_bytes, bool* end_of_input)
{
  if (bufs_idx_ < bufs_.size()) {
    *buf = bufs_[bufs_idx_];
    *input_bytes = buf_byte_sizes_[bufs_idx_];
    bufs_idx_++;
  } else {
    *buf = nullptr;
    *input_bytes = 0;
  }
  *end_of_input = (bufs_idx_ >= bufs_.size());

  return Error::Success;
}

//==============================================================================

Error
InferRequestedOutput::Create(
    InferRequestedOutput** infer_output, const std::string& name,
    const size_t class_count, const std::string& datatype)
{
  *infer_output = new InferRequestedOutput(name, datatype, class_count);
  return Error::Success;
}

Error
InferRequestedOutput::SetSharedMemory(
    const std::string& region_name, const size_t byte_size, const size_t offset)
{
  shm_name_ = region_name;
  shm_byte_size_ = byte_size;
  shm_offset_ = offset;
  io_type_ = SHARED_MEMORY;

  return Error::Success;
}

Error
InferRequestedOutput::UnsetSharedMemory()
{
  shm_name_ = "";
  shm_byte_size_ = 0;
  shm_offset_ = 0;
  io_type_ = NONE;

  return Error::Success;
}

InferRequestedOutput::InferRequestedOutput(
    const std::string& name, const std::string& datatype,
    const size_t class_count)
    : name_(name), datatype_(datatype), class_count_(class_count),
      io_type_(NONE)
{
}

Error
InferRequestedOutput::SharedMemoryInfo(
    std::string* name, size_t* byte_size, size_t* offset) const
{
  if (io_type_ != SHARED_MEMORY) {
    return Error("The input has not been set with the shared memory.");
  }

  *name = shm_name_;
  *byte_size = shm_byte_size_;
  *offset = shm_offset_;

  return Error::Success;
}

Error
InferRequestedOutput::SetBinaryData(const bool binary_data)
{
  binary_data_ = binary_data;
  return Error::Success;
}

//==============================================================================

}}  // namespace triton::client
