// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

namespace triton { namespace perfanalyzer {

class DataLoader {
 public:
  DataLoader(size_t batch_size);

  /// Returns the total number of data steps that can be supported by a
  /// non-sequence model.
  size_t GetTotalStepsNonSequence() { return max_non_sequence_step_id_; }

  /// Returns the total number of data streams available.
  size_t GetDataStreamsCount() { return data_stream_cnt_; }

  /// Returns the total data steps supported for a requested data stream
  /// id.
  /// \param stream_id The target stream id
  size_t GetTotalSteps(size_t stream_id)
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
  cb::Error ReadDataFromJSON(
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
  /// \param data Returns the pointer to the data for the requested input.
  /// \param batch1_size Returns the size of the input data in bytes.
  /// Returns error object indicating status
  cb::Error GetInputData(
      const ModelTensor& input, const int stream_id, const int step_id,
      const uint8_t** data_ptr, size_t* batch1_size);

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
  /// \param data Returns the pointer to the data for the requested output.
  /// \param batch1_size Returns the size of the output data in bytes.
  /// Returns error object indicating status
  cb::Error GetOutputData(
      const std::string& output_name, const int stream_id, const int step_id,
      const uint8_t** data_ptr, size_t* batch1_size);

 private:
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

  // The batch_size_ for the data
  size_t batch_size_;
  // The total number of data streams available.
  size_t data_stream_cnt_;
  // A vector containing the supported step number for respective stream
  // ids.
  std::vector<size_t> step_num_;
  // The maximum supported data step id for non-sequence model.
  size_t max_non_sequence_step_id_;

  // User provided input data, it will be preferred over synthetic data
  std::unordered_map<std::string, std::vector<char>> input_data_;
  std::unordered_map<std::string, std::vector<int64_t>> input_shapes_;

  // User provided output data for validation
  std::unordered_map<std::string, std::vector<char>> output_data_;
  std::unordered_map<std::string, std::vector<int64_t>> output_shapes_;

  // Placeholder for generated input data, which will be used for all inputs
  // except string
  std::vector<uint8_t> input_buf_;
};

}}  // namespace triton::perfanalyzer
