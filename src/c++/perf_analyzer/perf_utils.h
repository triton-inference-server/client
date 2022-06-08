// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <sys/stat.h>
#include <time.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include "client_backend/client_backend.h"

namespace pa = triton::perfanalyzer;
namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

constexpr uint64_t NANOS_PER_SECOND = 1000000000;
constexpr uint64_t NANOS_PER_MILLIS = 1000000;
#define CHRONO_TO_NANOS(TS)                                                    \
  (std::chrono::duration_cast<std::chrono::nanoseconds>(TS.time_since_epoch()) \
       .count())
#define CHRONO_TO_MILLIS(TS) (CHRONO_TO_NANOS(TS) / pa::NANOS_PER_MILLIS)

//==============================================================================
using TimestampVector = std::vector<std::tuple<
    std::chrono::time_point<std::chrono::system_clock>,
    std::chrono::time_point<std::chrono::system_clock>, uint32_t, bool>>;

// Will use the characters specified here to construct random strings
std::string const character_set =
    "abcdefghijklmnaoqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 .?!";

// A boolean flag to mark an interrupt and commencement of early exit
extern volatile bool early_exit;

enum Distribution { POISSON = 0, CONSTANT = 1, CUSTOM = 2 };
enum SearchMode { LINEAR = 0, BINARY = 1, NONE = 2 };
enum SharedMemoryType {
  SYSTEM_SHARED_MEMORY = 0,
  CUDA_SHARED_MEMORY = 1,
  NO_SHARED_MEMORY = 2
};

constexpr uint64_t NO_LIMIT = 0;

// Converts the datatype from tensorflow to perf analyzer space
// \param tf_dtype The data type string returned from the model metadata.
// \param datatype Returns the datatype in perf_analyzer space.
// \return error status. Returns Non-Ok if an error is encountered during
//  read operation.
cb::Error ConvertDTypeFromTFS(
    const std::string& tf_dtype, std::string* datatype);

// Parse the communication protocol type
cb::ProtocolType ParseProtocol(const std::string& str);

// Reads the data from file specified by path into vector of characters
// \param path The complete path to the file to be read
// \param contents The character vector that will contain the data read
// \return error status. Returns Non-Ok if an error is encountered during
//  read operation.
cb::Error ReadFile(const std::string& path, std::vector<char>* contents);

// Reads the string from file specified by path into vector of strings
// \param path The complete path to the file to be read
// \param contents The string vector that will contain the data read
// \return error status. Returns Non-Ok if an error is encountered during
//  read operation.
cb::Error ReadTextFile(
    const std::string& path, std::vector<std::string>* contents);

// Reads the time intervals in microseconds from file specified by path into
// vector of time intervals in nanoseconds.
// \param path The complete path to the file to be read
// \param contents The time interval vector that will contain the data read.
// \return error status. Returns Non-Ok if an error is encountered during
//  read operation.
cb::Error ReadTimeIntervalsFile(
    const std::string& path, std::vector<std::chrono::nanoseconds>* contents);

// To check whether the path points to a valid system directory
bool IsDirectory(const std::string& path);

// To check whether the path points to a valid system file
bool IsFile(const std::string& complete_path);

// Calculates the byte size tensor for given shape and datatype.
int64_t ByteSize(
    const std::vector<int64_t>& shape, const std::string& datatype);

// Get the number of elements in the tensor for given shape.
int64_t ElementCount(const std::vector<int64_t>& shape);

// Serializes the string tensor to length prepended bytes.
void SerializeStringTensor(
    std::vector<std::string> string_tensor, std::vector<char>* serialized_data);

// Serializes an explicit tensor read from the data file to the
// raw bytes.
cb::Error SerializeExplicitTensor(
    const rapidjson::Value& tensor, const std::string& dt,
    std::vector<char>* decoded_data);

// Generates a random string of specified length using characters specified in
// character_set.
std::string GetRandomString(const int string_length);

// Returns the shape string containing the values provided in the vector
std::string ShapeVecToString(
    const std::vector<int64_t> shape_vec, bool skip_first = false);

// Returns the string containing the shape tensor values
std::string ShapeTensorValuesToString(const int* data_ptr, const int count);

// Returns the request schedule distribution generator with the specified
// request rate.
template <Distribution distribution>
std::function<std::chrono::nanoseconds(std::mt19937&)> ScheduleDistribution(
    const double request_rate);

}}  // namespace triton::perfanalyzer
