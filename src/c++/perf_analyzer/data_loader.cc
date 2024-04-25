// Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "data_loader.h"

#include <b64/decode.h>
#include <rapidjson/filereadstream.h>

#include <fstream>

namespace triton { namespace perfanalyzer {

DataLoader::DataLoader(const size_t batch_size)
    : batch_size_(batch_size), data_stream_cnt_(0)
{
}

cb::Error
DataLoader::ReadDataFromDir(
    const std::shared_ptr<ModelTensorMap>& inputs,
    const std::shared_ptr<ModelTensorMap>& outputs,
    const std::string& data_directory)
{
  // Directory structure supports only a single data stream and step
  data_stream_cnt_ = 1;
  step_num_.push_back(1);

  for (const auto& file : std::filesystem::directory_iterator(data_directory)) {
    std::string input_name = file.path().filename().string();
    if (inputs->find(input_name) == inputs->end()) {
      return cb::Error(
          "Provided data file '" + input_name +
              "' does not correspond to a valid model input.",
          pa::GENERIC_ERROR);
    }
  }

  for (const auto& input : *inputs) {
    if (input.second.datatype_.compare("BYTES") != 0) {
      const auto file_path = data_directory + "/" + input.second.name_;
      std::string key_name(
          input.second.name_ + "_" + std::to_string(0) + "_" +
          std::to_string(0));
      auto it = input_data_.emplace(key_name, std::vector<char>()).first;
      RETURN_IF_ERROR(ReadFile(file_path, &it->second));
      int64_t byte_size = ByteSize(input.second.shape_, input.second.datatype_);
      if (byte_size < 0) {
        return cb::Error(
            "input " + input.second.name_ +
                " contains dynamic shape, provide shapes to send along with "
                "the request",
            pa::GENERIC_ERROR);
      }
      if (it->second.size() != byte_size) {
        return cb::Error(
            "provided data for input " + input.second.name_ +
                " has byte size " + std::to_string(it->second.size()) +
                ", expect " + std::to_string(byte_size),
            pa::GENERIC_ERROR);
      }
    } else {
      const auto file_path = data_directory + "/" + input.second.name_;
      std::vector<std::string> input_string_data;
      RETURN_IF_ERROR(ReadTextFile(file_path, &input_string_data));
      std::string key_name(
          input.second.name_ + "_" + std::to_string(0) + "_" +
          std::to_string(0));
      auto it = input_data_.emplace(key_name, std::vector<char>()).first;
      SerializeStringTensor(input_string_data, &it->second);
      int64_t batch1_num_strings = ElementCount(input.second.shape_);
      if (batch1_num_strings == -1) {
        return cb::Error(
            "input " + input.second.name_ +
                " contains dynamic shape, provide shapes to send along with "
                "the request",
            pa::GENERIC_ERROR);
      }
      if (input_string_data.size() != batch1_num_strings) {
        return cb::Error(
            "provided data for input " + input.second.name_ + " has " +
                std::to_string(input_string_data.size()) +
                " elements, expect " + std::to_string(batch1_num_strings),
            pa::GENERIC_ERROR);
      }
    }
  }

  for (const auto& output : *outputs) {
    if (output.second.datatype_.compare("BYTES") != 0) {
      const auto file_path = data_directory + "/" + output.second.name_;
      std::string key_name(
          output.second.name_ + "_" + std::to_string(0) + "_" +
          std::to_string(0));
      auto it = output_data_.emplace(key_name, std::vector<char>()).first;
      if (!ReadFile(file_path, &it->second).IsOk()) {
        output_data_.erase(it);
      }
    } else {
      const auto file_path = data_directory + "/" + output.second.name_;
      std::vector<std::string> output_string_data;
      if (!ReadTextFile(file_path, &output_string_data).IsOk()) {
        continue;
      }
      std::string key_name(
          output.second.name_ + "_" + std::to_string(0) + "_" +
          std::to_string(0));
      auto it = output_data_.emplace(key_name, std::vector<char>()).first;
      SerializeStringTensor(output_string_data, &it->second);
    }
  }

  for (const auto& file : std::filesystem::directory_iterator(data_directory)) {
    std::string input_name = file.path().filename().string();
    if (inputs->find(input_name) == inputs->end()) {
      return cb::Error(
          "Provided data file '" + input_name +
              "' does not correspond to a valid model input.",
          pa::GENERIC_ERROR);
    }
  }

  return cb::Error::Success;
}

cb::Error
DataLoader::ReadDataFromJSON(
    const std::shared_ptr<ModelTensorMap>& inputs,
    const std::shared_ptr<ModelTensorMap>& outputs,
    const std::string& json_file)
{
  FILE* data_file = fopen(json_file.c_str(), "r");
  if (data_file == nullptr) {
    return cb::Error(
        "failed to open file for reading provided data", pa::GENERIC_ERROR);
  }

  char readBuffer[65536];
  rapidjson::FileReadStream fs(data_file, readBuffer, sizeof(readBuffer));

  rapidjson::Document d{};
  const unsigned int parseFlags = rapidjson::kParseNanAndInfFlag;
  d.ParseStream<parseFlags>(fs);

  fclose(data_file);

  return ParseData(d, inputs, outputs);
}

cb::Error
DataLoader::ParseData(
    const rapidjson::Document& json,
    const std::shared_ptr<ModelTensorMap>& inputs,
    const std::shared_ptr<ModelTensorMap>& outputs)
{
  if (json.HasParseError()) {
    std::cerr << "cb::Error  : " << json.GetParseError() << '\n'
              << "Offset : " << json.GetErrorOffset() << '\n';
    return cb::Error(
        "failed to parse the specified json file for reading provided data",
        pa::GENERIC_ERROR);
  }

  if (!json.HasMember("data")) {
    return cb::Error(
        "The json file doesn't contain data field", pa::GENERIC_ERROR);
  }

  const rapidjson::Value& streams = json["data"];

  // Validation data is optional, once provided, it must align with 'data'
  const rapidjson::Value* out_streams = nullptr;
  if (json.HasMember("validation_data")) {
    out_streams = &json["validation_data"];
    if (out_streams->Size() != streams.Size()) {
      return cb::Error(
          "The 'validation_data' field doesn't align with 'data' field in the "
          "json file",
          pa::GENERIC_ERROR);
    }
  }

  int count = streams.Size();

  data_stream_cnt_ += count;
  int offset = step_num_.size();
  for (size_t i = offset; i < data_stream_cnt_; i++) {
    const rapidjson::Value& steps = streams[i - offset];
    const rapidjson::Value* output_steps =
        (out_streams == nullptr) ? nullptr : &(*out_streams)[i - offset];

    RETURN_IF_ERROR(ValidateParsingMode(steps));

    if (steps.IsArray()) {
      step_num_.push_back(steps.Size());
      for (size_t k = 0; k < step_num_[i]; k++) {
        RETURN_IF_ERROR(ReadTensorData(steps[k], inputs, i, k, true));
      }

      if (output_steps != nullptr) {
        if (!output_steps->IsArray() ||
            (output_steps->Size() != steps.Size())) {
          return cb::Error(
              "The 'validation_data' field doesn't align with 'data' field in "
              "the json file",
              pa::GENERIC_ERROR);
        }
        for (size_t k = 0; k < step_num_[i]; k++) {
          RETURN_IF_ERROR(
              ReadTensorData((*output_steps)[k], outputs, i, k, false));
        }
      }
    } else {
      // There is no nesting of tensors, hence, will interpret streams as steps
      // and add the tensors to a single stream '0'.
      int offset = 0;
      if (step_num_.empty()) {
        step_num_.push_back(count);
      } else {
        offset = step_num_[0];
        step_num_[0] += (count);
      }
      data_stream_cnt_ = 1;
      for (size_t k = offset; k < step_num_[0]; k++) {
        RETURN_IF_ERROR(
            ReadTensorData(streams[k - offset], inputs, 0, k, true));
      }

      if (out_streams != nullptr) {
        for (size_t k = offset; k < step_num_[0]; k++) {
          RETURN_IF_ERROR(
              ReadTensorData((*out_streams)[k - offset], outputs, 0, k, false));
        }
      }
      break;
    }
  }

  return cb::Error::Success;
}

cb::Error
DataLoader::GenerateData(
    std::shared_ptr<ModelTensorMap> inputs, const bool zero_input,
    const size_t string_length, const std::string& string_data)
{
  // Data generation supports only a single data stream and step
  // Not supported for inputs with dynamic shapes
  data_stream_cnt_ = 1;
  step_num_.push_back(1);

  // Validate the absence of shape tensors
  for (const auto& input : *inputs) {
    if (input.second.is_shape_tensor_) {
      return cb::Error(
          "can not generate data for shape tensor '" + input.second.name_ +
              "', user-provided data is needed.",
          pa::GENERIC_ERROR);
    }
  }

  uint64_t max_input_byte_size = 0;
  for (const auto& input : *inputs) {
    if (input.second.datatype_.compare("BYTES") != 0) {
      int64_t byte_size = ByteSize(input.second.shape_, input.second.datatype_);
      if (byte_size < 0) {
        return cb::Error(
            "input " + input.second.name_ +
                " contains dynamic shape, provide shapes to send along with "
                "the request",
            pa::GENERIC_ERROR);
      }
      max_input_byte_size = std::max(max_input_byte_size, (size_t)byte_size);
    } else {
      // Generate string input and store it into map
      std::vector<std::string> input_string_data;
      int64_t batch1_num_strings = ElementCount(input.second.shape_);
      if (batch1_num_strings == -1) {
        return cb::Error(
            "input " + input.second.name_ +
                " contains dynamic shape, provide shapes to send along with "
                "the request",
            pa::GENERIC_ERROR);
      }
      input_string_data.resize(batch1_num_strings);
      if (!string_data.empty()) {
        for (size_t i = 0; i < batch1_num_strings; i++) {
          input_string_data[i] = string_data;
        }
      } else {
        for (size_t i = 0; i < batch1_num_strings; i++) {
          input_string_data[i] = GetRandomString(string_length);
        }
      }

      std::string key_name(
          input.second.name_ + "_" + std::to_string(0) + "_" +
          std::to_string(0));
      auto it = input_data_.emplace(key_name, std::vector<char>()).first;
      SerializeStringTensor(input_string_data, &it->second);
    }
  }

  // Create a zero or randomly (as indicated by zero_input)
  // initialized buffer that is large enough to provide the largest
  // needed input. We (re)use this buffer for all non-string input values.
  if (max_input_byte_size > 0) {
    if (zero_input) {
      input_buf_.resize(max_input_byte_size, 0);
    } else {
      input_buf_.resize(max_input_byte_size);
      for (auto& byte : input_buf_) {
        byte = rand();
      }
    }
  }

  return cb::Error::Success;
}

cb::Error
DataLoader::GetInputData(
    const ModelTensor& input, const int stream_id, const int step_id,
    TensorData& data)
{
  data.data_ptr = nullptr;
  data.batch1_size = 0;
  data.is_valid = false;

  // If json data is available then try to retrieve the data from there
  if (!input_data_.empty()) {
    RETURN_IF_ERROR(ValidateIndexes(stream_id, step_id));

    std::string key_name(
        input.name_ + "_" + std::to_string(stream_id) + "_" +
        std::to_string(step_id));

    // Get the data and the corresponding byte-size
    auto it = input_data_.find(key_name);
    if (it != input_data_.end()) {
      std::vector<char>* data_vec = &it->second;
      data.is_valid = true;
      data.batch1_size = data_vec->size();
      data.data_ptr = (const uint8_t*)data_vec->data();
    }
  }

  if (!data.is_valid) {
    if ((input.datatype_.compare("BYTES") != 0) && (input_buf_.size() != 0)) {
      int64_t byte_size = ByteSize(input.shape_, input.datatype_);
      if (byte_size < 0) {
        return cb::Error(
            "failed to get correct byte size for '" + input.name_ + "'.",
            pa::GENERIC_ERROR);
      }
      data.batch1_size = (size_t)byte_size;
      data.data_ptr = &input_buf_[0];
      data.is_valid = true;
    }
  }

  if (input.is_optional_ == false && !data.is_valid) {
    return cb::Error(
        "unable to find data for input '" + input.name_ + "'.",
        pa::GENERIC_ERROR);
  }

  return cb::Error::Success;
}

cb::Error
DataLoader::GetOutputData(
    const std::string& output_name, const int stream_id, const int step_id,
    TensorData& data)
{
  data.data_ptr = nullptr;
  data.batch1_size = 0;
  data.is_valid = false;

  // If json data is available then try to retrieve the data from there
  if (!output_data_.empty()) {
    RETURN_IF_ERROR(ValidateIndexes(stream_id, step_id));

    std::string key_name(
        output_name + "_" + std::to_string(stream_id) + "_" +
        std::to_string(step_id));
    // Get the data and the corresponding byte-size
    auto it = output_data_.find(key_name);
    if (it != output_data_.end()) {
      std::vector<char>* data_vec = &it->second;
      data.is_valid = true;
      data.batch1_size = data_vec->size();
      data.data_ptr = (const uint8_t*)data_vec->data();
    }
  }
  return cb::Error::Success;
}

cb::Error
DataLoader::ValidateIndexes(int stream_id, int step_id)
{
  if (stream_id < 0 || stream_id >= (int)data_stream_cnt_) {
    return cb::Error(
        "stream_id for retrieving the data should be less than " +
            std::to_string(data_stream_cnt_) + ", got " +
            std::to_string(stream_id),
        pa::GENERIC_ERROR);
  }
  if (step_id < 0 || step_id >= (int)step_num_[stream_id]) {
    return cb::Error(
        "step_id for retrieving the data should be less than " +
            std::to_string(step_num_[stream_id]) + ", got " +
            std::to_string(step_id),
        pa::GENERIC_ERROR);
  }
  return cb::Error::Success;
}


cb::Error
DataLoader::GetInputShape(
    const ModelTensor& input, const int stream_id, const int step_id,
    std::vector<int64_t>* provided_shape)
{
  std::string key_name(
      input.name_ + "_" + std::to_string(stream_id) + "_" +
      std::to_string(step_id));

  provided_shape->clear();

  // Prefer the values read from file over the ones provided from
  // CLI
  auto it = input_shapes_.find(key_name);
  if (it != input_shapes_.end()) {
    *provided_shape = it->second;
  } else {
    *provided_shape = input.shape_;
  }
  return cb::Error::Success;
}

cb::Error
DataLoader::ReadTensorData(
    const rapidjson::Value& step,
    const std::shared_ptr<ModelTensorMap>& tensors, const int stream_index,
    const int step_index, const bool is_input)
{
  std::unordered_set<std::string> model_io_names;
  auto& tensor_data = is_input ? input_data_ : output_data_;
  auto& tensor_shape = is_input ? input_shapes_ : output_shapes_;
  for (const auto& io : *tensors) {
    std::cerr << "Checking tensor: " << io.first << std::endl;
    model_io_names.insert(io.first);
    if (step.HasMember(io.first.c_str())) {
      std::string key_name(
          io.first + "_" + std::to_string(stream_index) + "_" +
          std::to_string(step_index));

      auto it = tensor_data.emplace(key_name, std::vector<char>()).first;

      const rapidjson::Value& tensor = step[(io.first).c_str()];

      const rapidjson::Value* content;

      // Check if the input data file is malformed
      if (!(tensor.IsArray() || tensor.IsObject())) {
        return cb::Error("Input data file is malformed.", pa::GENERIC_ERROR);
      }

      if (tensor.IsArray()) {
        content = &tensor;
      } else {
        // Populate the shape values first if available
        if (tensor.HasMember("shape")) {
          auto shape_it =
              tensor_shape.emplace(key_name, std::vector<int64_t>()).first;
          for (const auto& value : tensor["shape"].GetArray()) {
            if (!value.IsInt()) {
              return cb::Error(
                  "shape values must be integers.", pa::GENERIC_ERROR);
            }
            shape_it->second.push_back(value.GetInt());
          }
        }

        if (tensor.HasMember("b64")) {
          content = &tensor;
        } else {
          if (!tensor.HasMember("content")) {
            return cb::Error(
                "missing content field. ( Location stream id: " +
                    std::to_string(stream_index) +
                    ", step id: " + std::to_string(step_index) + ")",
                pa::GENERIC_ERROR);
          }

          content = &tensor["content"];
        }
      }

      if (content->IsArray()) {
        RETURN_IF_ERROR(SerializeExplicitTensor(
            *content, io.second.datatype_, &it->second));
      } else {
        if (content->IsObject() && content->HasMember("b64")) {
          if ((*content)["b64"].IsString()) {
            const std::string& encoded = (*content)["b64"].GetString();
            it->second.resize(encoded.length());
            base64::decoder D;
            int size =
                D.decode(encoded.c_str(), encoded.length(), &it->second[0]);
            it->second.resize(size);
          } else {
            return cb::Error(
                "the value of b64 field should be of type string ( "
                "Location stream id: " +
                    std::to_string(stream_index) +
                    ", step id: " + std::to_string(step_index) + ")",
                pa::GENERIC_ERROR);
          }
        } else {
          return cb::Error(
              "The tensor values are not supported. Expected an array or "
              "b64 string ( Location stream id: " +
                  std::to_string(stream_index) +
                  ", step id: " + std::to_string(step_index) + ")",
              pa::GENERIC_ERROR);
        }
      }

      RETURN_IF_ERROR(ValidateTensor(io.second, stream_index, step_index));

    } else if (io.second.is_optional_ == false) {
      return cb::Error(
          "missing tensor " + io.first +
              " ( Location stream id: " + std::to_string(stream_index) +
              ", step id: " + std::to_string(step_index) + ")",
          pa::GENERIC_ERROR);
    }
  }

  // Add allowed non-model inputs/outputs to the model_io_names set
  model_io_names.insert("model");

  for (auto itr = step.MemberBegin(); itr != step.MemberEnd(); ++itr) {
    if (model_io_names.find(itr->name.GetString()) == model_io_names.end()) {
      return cb::Error(
          "The input or output '" + std::string(itr->name.GetString()) +
              "' is not found in the model configuration",
          pa::GENERIC_ERROR);
    }
  }

  return cb::Error::Success;
}


cb::Error
DataLoader::ReadFile(const std::string& path, std::vector<char>* contents)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return cb::Error("failed to open file '" + path + "'", pa::GENERIC_ERROR);
  }

  in.seekg(0, std::ios::end);

  int file_size = in.tellg();
  if (file_size > 0) {
    contents->resize(file_size);
    in.seekg(0, std::ios::beg);
    in.read(&(*contents)[0], contents->size());
  }

  in.close();

  // If size is invalid, report after ifstream is closed
  if (file_size < 0) {
    return cb::Error(
        "failed to get size for file '" + path + "'", pa::GENERIC_ERROR);
  } else if (file_size == 0) {
    return cb::Error("file '" + path + "' is empty", pa::GENERIC_ERROR);
  }

  return cb::Error::Success;
}

cb::Error
DataLoader::ReadTextFile(
    const std::string& path, std::vector<std::string>* contents)
{
  std::ifstream in(path);
  if (!in) {
    return cb::Error("failed to open file '" + path + "'", pa::GENERIC_ERROR);
  }

  std::string current_string;
  while (std::getline(in, current_string)) {
    contents->push_back(current_string);
  }
  in.close();

  if (contents->size() == 0) {
    return cb::Error("file '" + path + "' is empty", pa::GENERIC_ERROR);
  }
  return cb::Error::Success;
}

cb::Error
DataLoader::ValidateTensor(
    const ModelTensor& model_tensor, const int stream_index,
    const int step_index)
{
  std::string key_name(
      model_tensor.name_ + "_" + std::to_string(stream_index) + "_" +
      std::to_string(step_index));

  auto data_it = input_data_.find(key_name);
  if (data_it == input_data_.end()) {
    data_it = output_data_.find(key_name);
  }
  if (data_it == output_data_.end()) {
    return cb::Error("Can't validate a nonexistent tensor");
  }

  auto shape_it = input_shapes_.find(key_name);

  const std::vector<char>& data = data_it->second;
  const std::vector<int64_t>& shape = (shape_it == input_shapes_.end())
                                          ? model_tensor.shape_
                                          : shape_it->second;

  int64_t batch1_byte = ByteSize(shape, model_tensor.datatype_);

  RETURN_IF_ERROR(ValidateTensorShape(shape, model_tensor));
  RETURN_IF_ERROR(ValidateTensorDataSize(data, batch1_byte, model_tensor));

  return cb::Error::Success;
}

cb::Error
DataLoader::ValidateTensorShape(
    const std::vector<int64_t>& shape, const ModelTensor& model_tensor)
{
  int element_count = ElementCount(shape);
  if (element_count < 0) {
    return cb::Error(
        "The variable-sized tensor \"" + model_tensor.name_ +
            "\" with model shape " + ShapeVecToString(model_tensor.shape_) +
            " needs to have its shape fully defined. See the --shape option.",
        pa::GENERIC_ERROR);
  }

  bool is_error = false;

  if (shape.size() != model_tensor.shape_.size()) {
    is_error = true;
  }

  for (size_t i = 0; i < shape.size() && !is_error; i++) {
    if (shape[i] != model_tensor.shape_[i] && model_tensor.shape_[i] != -1) {
      is_error = true;
    }
  }

  if (is_error) {
    return cb::Error(
        "The supplied shape of " + ShapeVecToString(shape) + " for input \"" +
        model_tensor.name_ +
        "\" is incompatible with the model's input shape of " +
        ShapeVecToString(model_tensor.shape_));
  }

  return cb::Error::Success;
}

cb::Error
DataLoader::ValidateTensorDataSize(
    const std::vector<char>& data, int64_t batch1_byte,
    const ModelTensor& model_tensor)
{
  // Validate that the supplied data matches the amount of data expected based
  // on the shape
  if (batch1_byte > 0 && (size_t)batch1_byte != data.size()) {
    return cb::Error(
        "mismatch in the data provided for " + model_tensor.name_ +
            ". Expected: " + std::to_string(batch1_byte) +
            " bytes, Got: " + std::to_string(data.size()) + " bytes",
        pa::GENERIC_ERROR);
  }

  return cb::Error::Success;
}

cb::Error
DataLoader::ValidateParsingMode(const rapidjson::Value& steps)
{
  // If our first time parsing data, capture the mode
  if (step_num_.size() == 0) {
    multiple_stream_mode_ = steps.IsArray();
  } else {
    if (steps.IsArray() != multiple_stream_mode_) {
      return cb::Error(
          "Inconsistency in input-data provided. Can not have a combination of "
          "objects and arrays inside of the Data array",
          pa::GENERIC_ERROR);
    }
  }
  return cb::Error::Success;
}

}}  // namespace triton::perfanalyzer
