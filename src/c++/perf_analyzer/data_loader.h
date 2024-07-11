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
#pragma once

#include <fstream>

#include "model_parser.h"
#include "perf_utils.h"
#include "tensor_data.h"

namespace triton { namespace perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class NaggyMockDataLoader;
#endif


class DataLoader {
 public:
  DataLoader(size_t batch_size);

  /// Returns the total number of data streams available.
  size_t GetDataStreamsCount() { return data_stream_cnt_; }

  /// Returns the total data steps supported for a requested data stream
  /// id.
  /// \param stream_id The target stream id
  virtual size_t GetTotalSteps(size_t stream_id)
  {
    if (stream_id < data_stream_cnt_) {
      return step_num_[stream_id];
    }
    return 0;
  }

  /// Reads the input data from the specified data directory.
  /// \param inputs The pointer to the map holding the information about
  /// input tensors of a model
  /// \param data_directory The path to the directory containing the data
  cb::Error ReadDataFromDir(
      const std::shared_ptr<ModelTensorMap>& inputs,
      const std::shared_ptr<ModelTensorMap>& outputs,
      const std::string& data_directory);

  /// Reads the input data from the specified json file.
  /// \param inputs The pointer to the map holding the information about
  /// input tensors of a model
  /// \param json_file The json file containing the user-provided input
  /// data.
  /// Returns error object indicating status
  virtual cb::Error ReadDataFromJSON(
      const std::shared_ptr<ModelTensorMap>& inputs,
      const std::shared_ptr<ModelTensorMap>& outputs,
      const std::string& json_file);

  /// Generates the input data to use with the inference requests
  /// \param inputs The pointer to the map holding the information about
  /// input tensors of a model
  /// \param zero_input Whether or not to use zero value for buffer
  /// initialization.
  /// \param string_length The length of the string to generate for
  /// tensor inputs.
  /// \param string_data The user provided string to use to populate
  /// string tensors
  /// Returns error object indicating status
  cb::Error GenerateData(
      std::shared_ptr<ModelTensorMap> inputs, const bool zero_input,
      const size_t string_length, const std::string& string_data);

  /// Helper function to access data for the specified input
  /// \param input The target model input tensor
  /// \param stream_id The data stream_id to use for retrieving input data.
  /// \param step_id The data step_id to use for retrieving input data.
  /// \param data Returns the input TensorData
  /// Returns error object indicating status
  cb::Error GetInputData(
      const ModelTensor& input, const int stream_id, const int step_id,
      TensorData& data);

  /// Helper function to get the shape values to the input
  /// \param input The target model input tensor
  /// \param stream_id The data stream_id to use for retrieving input shape.
  /// \param step_id The data step_id to use for retrieving input shape.
  /// \param shape returns the pointer to the vector containing the shape
  /// values.
  /// Returns error object indicating status
  cb::Error GetInputShape(
      const ModelTensor& input, const int stream_id, const int step_id,
      std::vector<int64_t>* shape);

  /// Helper function to access data for the specified output. nullptr will be
  /// returned if there is no data specified.
  /// \param output_name The name of the output tensor
  /// \param stream_id The data stream_id to use for retrieving output data.
  /// \param step_id The data step_id to use for retrieving output data.
  /// \param data Returns the output TensorData
  /// Returns error object indicating status
  cb::Error GetOutputData(
      const std::string& output_name, const int stream_id, const int step_id,
      TensorData& data);

  /// Return an error if the stream index or step index are invalid
  cb::Error ValidateIndexes(int stream_index, int step_index);

 protected:
  /// Parses the input and output data from the json document
  /// \param inputs The input tensors of a model
  /// \param outputs The output tensors of a model
  /// \param json The json document containing the raw json inputs/outputs
  /// \return Returns error object indicating status
  cb::Error ParseData(
      const rapidjson::Document& json,
      const std::shared_ptr<ModelTensorMap>& inputs,
      const std::shared_ptr<ModelTensorMap>& outputs);

 private:
  /// Reads the data from file specified by path into vector of characters
  /// \param path The complete path to the file to be read
  /// \param contents The character vector that will contain the data read
  /// \return error status. Returns Non-Ok if an error is encountered during
  ///  read operation.
  virtual cb::Error ReadFile(
      const std::string& path, std::vector<char>* contents);

  /// Reads the string from file specified by path into vector of strings
  /// \param path The complete path to the file to be read
  /// \param contents The string vector that will contain the data read
  /// \return error status. Returns Non-Ok if an error is encountered during
  ///  read operation.
  virtual cb::Error ReadTextFile(
      const std::string& path, std::vector<std::string>* contents);

  /// Helper function to read data for the specified input from json
  /// \param step the DOM for current step
  /// \param inputs The pointer to the map holding the information about
  /// input tensors of a model
  /// \param stream_index the stream index the data should be exported to.
  /// \param step_index the step index the data should be exported to.
  /// Returns error object indicating status
  cb::Error ReadTensorData(
      const rapidjson::Value& step,
      const std::shared_ptr<ModelTensorMap>& tensors, const int stream_index,
      const int step_index, const bool is_input);

  /// Helper function to validate the provided data and shape for the tensor
  /// \param input The target model input or output tensor
  /// \param stream_index the stream index the data should be exported to.
  /// \param step_index the step index the data should be exported to.
  /// Returns error object indicating status
  cb::Error ValidateTensor(
      const ModelTensor& model_tensor, const int stream_index,
      const int step_index);

  /// Helper function to validate the provided shape for a tensor
  /// \param shape Shape for the tensor
  /// \param model_tensor The tensor to validate
  /// Returns error object indicating status
  cb::Error ValidateTensorShape(
      const std::vector<int64_t>& shape, const ModelTensor& model_tensor);

  /// Helper function to validate the provided data's size
  /// \param data The provided data for the tensor
  /// \param batch1_byte The expected number of bytes of data
  /// \param model_tensor The tensor to validate
  /// Returns error object indicating status
  cb::Error ValidateTensorDataSize(
      const std::vector<char>& data, int64_t batch1_byte,
      const ModelTensor& model_tensor);

  /// Helper function to validate consistency of parsing mode for provided input
  /// data.  The code explicitly does not support a mixture of objects (multiple
  /// entries of a single stream) and arrays (multiple streams)
  ///
  /// \param steps The json data provided for one or multiple streams
  cb::Error ValidateParsingMode(const rapidjson::Value& steps);

  // The batch_size_ for the data
  size_t batch_size_{1};
  // The total number of data streams available.
  size_t data_stream_cnt_{0};
  // A vector containing the supported step number for respective stream
  // ids.
  std::vector<size_t> step_num_;

  // User provided input data, it will be preferred over synthetic data
  std::unordered_map<std::string, std::vector<char>> input_data_;
  std::unordered_map<std::string, std::vector<int64_t>> input_shapes_;

  // User provided output data for validation
  std::unordered_map<std::string, std::vector<char>> output_data_;
  std::unordered_map<std::string, std::vector<int64_t>> output_shapes_;

  // Placeholder for generated input data, which will be used for all inputs
  // except string
  std::vector<uint8_t> input_buf_;

  // Tracks what type of input data has been provided
  bool multiple_stream_mode_ = false;

#ifndef DOCTEST_CONFIG_DISABLE
  friend NaggyMockDataLoader;

 public:
  DataLoader() = default;
#endif
};

}}  // namespace triton::perfanalyzer
