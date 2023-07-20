// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <filesystem>
#include <fstream>

#include "doctest.h"
#include "perf_utils.h"
#include "test_utils.h"

namespace triton { namespace perfanalyzer {

/// Helper class to test perf_utils.cc
///
class TestPerfUtils {
 public:
  /// Given a distributionType and request rate, confirm that request pattern
  /// matches what is expected.
  ///
  static void TestDistribution(
      Distribution distribution_type, uint32_t request_rate)
  {
    std::mt19937 schedule_rng;
    std::vector<int64_t> delays;

    double avg, variance;
    double expected_avg, expected_variance;

    auto dist_func = GetDistributionFunction(distribution_type, request_rate);

    for (int i = 0; i < 100000; i++) {
      auto delay = dist_func(schedule_rng);
      delays.push_back(delay.count());
    }

    avg = CalculateAverage(delays);
    variance = CalculateVariance(delays, avg);

    std::chrono::nanoseconds ns_in_one_second =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::seconds(1));
    expected_avg = ns_in_one_second.count() / request_rate;

    if (distribution_type == CONSTANT) {
      expected_variance = 0;
    } else {
      // By definition, variance = mean for poisson
      expected_variance = expected_avg;
    }

    CHECK(avg == doctest::Approx(expected_avg).epsilon(0.005));
    CHECK(variance == doctest::Approx(expected_variance).epsilon(0.005));
  }


 private:
  static std::function<std::chrono::nanoseconds(std::mt19937&)>
  GetDistributionFunction(Distribution type, uint32_t request_rate)
  {
    std::function<std::chrono::nanoseconds(std::mt19937&)> distributionFunction;

    if (type == CONSTANT) {
      distributionFunction = ScheduleDistribution<CONSTANT>(request_rate);
    } else if (type == POISSON) {
      distributionFunction = ScheduleDistribution<POISSON>(request_rate);
    } else {
      throw std::invalid_argument("Unexpected distribution type");
    }
    return distributionFunction;
  }
};

/// Test all distributions across various request rates
///
TEST_CASE("perf_utils: TestDistribution")
{
  std::vector<Distribution> distTypes{CONSTANT, POISSON};
  std::vector<uint32_t> requestRates{10, 100, 1000, 10000};

  for (auto dist : distTypes) {
    for (auto rate : requestRates) {
      TestPerfUtils::TestDistribution(dist, rate);
    }
  }
}

TEST_CASE("perf_utils: ParseTensorFormat")
{
  CHECK(ParseTensorFormat("binary") == cb::TensorFormat::BINARY);
  CHECK(ParseTensorFormat("BINARY") == cb::TensorFormat::BINARY);
  CHECK(ParseTensorFormat("json") == cb::TensorFormat::JSON);
  CHECK(ParseTensorFormat("JSON") == cb::TensorFormat::JSON);
  CHECK(ParseTensorFormat("abc") == cb::TensorFormat::UNKNOWN);
  CHECK(ParseTensorFormat("") == cb::TensorFormat::UNKNOWN);
}

TEST_CASE("perf_utils: ParseProtocol")
{
  CHECK(ParseProtocol("HTTP") == cb::ProtocolType::HTTP);
  CHECK(ParseProtocol("http") == cb::ProtocolType::HTTP);
  CHECK(ParseProtocol("GRPC") == cb::ProtocolType::GRPC);
  CHECK(ParseProtocol("grpc") == cb::ProtocolType::GRPC);
  CHECK(ParseProtocol("hhttp") == cb::ProtocolType::UNKNOWN);
  CHECK(ParseProtocol("") == cb::ProtocolType::UNKNOWN);
  CHECK(ParseProtocol("http2") == cb::ProtocolType::UNKNOWN);
}

TEST_CASE("perf_utils: ConvertDTypeFromTFS")
{
  std::string datatype;
  cb::Error status;

  SUBCASE("Check for correct conversion")
  {
    status = ConvertDTypeFromTFS("DT_HALF", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "FP16");

    status = ConvertDTypeFromTFS("DT_BFLOAT16", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "BF16");

    status = ConvertDTypeFromTFS("DT_FLOAT", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "FP32");

    status = ConvertDTypeFromTFS("DT_DOUBLE", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "FP64");

    status = ConvertDTypeFromTFS("DT_INT32", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "INT32");

    status = ConvertDTypeFromTFS("DT_INT16", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "INT16");

    status = ConvertDTypeFromTFS("DT_UINT16", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "UINT16");

    status = ConvertDTypeFromTFS("DT_INT8", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "INT8");

    status = ConvertDTypeFromTFS("DT_UINT8", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "UINT8");

    status = ConvertDTypeFromTFS("DT_STRING", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "BYTES");

    status = ConvertDTypeFromTFS("DT_INT64", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "INT64");

    status = ConvertDTypeFromTFS("DT_BOOL", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "BOOL");

    status = ConvertDTypeFromTFS("DT_UINT32", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "UINT32");

    status = ConvertDTypeFromTFS("DT_UINT64", &datatype);
    CHECK(status.IsOk());
    CHECK(datatype == "UINT64");
  }

  SUBCASE("Invalid tensorflow datatype")
  {
    status = ConvertDTypeFromTFS("dt_bool", &datatype);
    CHECK(!status.IsOk());
    CHECK(datatype == "");

    status = ConvertDTypeFromTFS("dt_uint8", &datatype);
    CHECK(!status.IsOk());
    CHECK(datatype == "");

    status = ConvertDTypeFromTFS("abcdef", &datatype);
    CHECK(!status.IsOk());
    CHECK(datatype == "");

    status = ConvertDTypeFromTFS("", &datatype);
    CHECK(!status.IsOk());
    CHECK(datatype == "");
  }
}

TEST_CASE("perf_utils: IsDirectory")
{
  // Create a temporary directory /tmp/abcdef1234
  std::filesystem::path temp_path = std::filesystem::temp_directory_path();
  temp_path /= "abcdef1234";

  CHECK(!IsDirectory(temp_path));

  std::filesystem::create_directory(temp_path);
  CHECK(IsDirectory(temp_path));

  std::filesystem::remove_all(temp_path);
  CHECK(!IsDirectory(temp_path));
}

TEST_CASE("perf_utils: IsFile")
{
  // Create a temporary file /tmp/abc/test.txt
  std::filesystem::path temp_path = std::filesystem::temp_directory_path();
  temp_path /= "abc/test.txt";

  CHECK(!IsFile(temp_path));

  std::filesystem::create_directory(temp_path.parent_path());
  std::ofstream file(temp_path);
  CHECK(IsFile(temp_path));

  std::filesystem::remove_all(temp_path.parent_path());
  CHECK(!IsFile(temp_path));
}

TEST_CASE("perf_utils: ByteSize")
{
  std::vector<int64_t> shape{3, 4, 5};

  SUBCASE("Single byte elements")
  {
    CHECK(ByteSize(shape, "BOOL") == 1 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "INT8") == 1 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "UINT8") == 1 * 3 * 4 * 5);
  }

  SUBCASE("2 byte elements")
  {
    CHECK(ByteSize(shape, "INT16") == 2 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "UINT16") == 2 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "FP16") == 2 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "BF16") == 2 * 3 * 4 * 5);
  }

  SUBCASE("4 byte elements")
  {
    CHECK(ByteSize(shape, "INT32") == 4 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "UINT32") == 4 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "FP32") == 4 * 3 * 4 * 5);
  }

  SUBCASE("8 byte elements")
  {
    CHECK(ByteSize(shape, "INT64") == 8 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "UINT64") == 8 * 3 * 4 * 5);
    CHECK(ByteSize(shape, "FP64") == 8 * 3 * 4 * 5);
  }

  SUBCASE("Dynamic shape tensor")
  {
    shape.insert(shape.begin(), -1);

    CHECK(ByteSize(shape, "BOOL") == -1);
    CHECK(ByteSize(shape, "INT8") == -1);
    CHECK(ByteSize(shape, "UINT8") == -1);

    CHECK(ByteSize(shape, "INT16") == -1);
    CHECK(ByteSize(shape, "UINT16") == -1);
    CHECK(ByteSize(shape, "FP16") == -1);
    CHECK(ByteSize(shape, "BF16") == -1);

    CHECK(ByteSize(shape, "INT32") == -1);
    CHECK(ByteSize(shape, "UINT32") == -1);
    CHECK(ByteSize(shape, "FP32") == -1);

    CHECK(ByteSize(shape, "INT64") == -1);
    CHECK(ByteSize(shape, "UINT64") == -1);
    CHECK(ByteSize(shape, "FP64") == -1);
  }

  SUBCASE("Unknown data types")
  {
    CHECK(ByteSize(shape, "bool") == -1);
    CHECK(ByteSize(shape, "int8") == -1);
    CHECK(ByteSize(shape, "uint8") == -1);

    CHECK(ByteSize(shape, "int16") == -1);
    CHECK(ByteSize(shape, "uint16") == -1);
    CHECK(ByteSize(shape, "fp16") == -1);
    CHECK(ByteSize(shape, "bf16") == -1);

    CHECK(ByteSize(shape, "int32") == -1);
    CHECK(ByteSize(shape, "uint32") == -1);
    CHECK(ByteSize(shape, "fp32") == -1);

    CHECK(ByteSize(shape, "int64") == -1);
    CHECK(ByteSize(shape, "uint64") == -1);
    CHECK(ByteSize(shape, "fp64") == -1);

    CHECK(ByteSize(shape, "abc") == -1);
    CHECK(ByteSize(shape, "1234") == -1);
    CHECK(ByteSize(shape, "") == -1);
  }
}

TEST_CASE("perf_utils: ElementCount")
{
  std::vector<int64_t> shape{3, 4, 5};

  SUBCASE("Static tensor shape")
  {
    CHECK(ElementCount(shape) == 60);

    shape.push_back(1);
    CHECK(ElementCount(shape) == 60);

    shape.push_back(300);
    CHECK(ElementCount(shape) == 18000);
  }

  SUBCASE("Dynamic tensor shape")
  {
    CHECK(ElementCount(shape) == 60);

    shape.push_back(-1);
    CHECK(ElementCount(shape) == -1);

    shape.pop_back();
    shape.insert(shape.begin(), -1);
    CHECK(ElementCount(shape) == -1);
  }
}

TEST_CASE("perf_utils: ShapeVecToString")
{
  std::vector<int64_t> shape{3, 4, 5};

  SUBCASE("No skipping first dim")
  {
    CHECK(ShapeVecToString(shape, false) == "[3,4,5]");

    shape.push_back(10);
    CHECK(ShapeVecToString(shape, false) == "[3,4,5,10]");

    shape.push_back(-1);
    CHECK(ShapeVecToString(shape, false) == "[3,4,5,10,-1]");

    shape.pop_back();
    shape.insert(shape.begin(), -1);
    CHECK(ShapeVecToString(shape, false) == "[-1,3,4,5,10]");

    shape.clear();
    CHECK(ShapeVecToString(shape, false) == "[]");
  }

  SUBCASE("Skipping first dim")
  {
    CHECK(ShapeVecToString(shape, true) == "[4,5]");

    shape.push_back(-1);
    CHECK(ShapeVecToString(shape, true) == "[4,5,-1]");

    shape.pop_back();
    shape.insert(shape.begin(), -1);
    CHECK(ShapeVecToString(shape, true) == "[3,4,5]");

    shape.clear();
    CHECK(ShapeVecToString(shape, true) == "[]");
  }
}

TEST_CASE("perf_utils: TensorToRegionName")
{
  CHECK(TensorToRegionName("name/with/slash") == "namewithslash");
  CHECK(TensorToRegionName("name//with//slash") == "namewithslash");
  CHECK(TensorToRegionName("name\\with\\backslash") == "namewithbackslash");
  CHECK(TensorToRegionName("name\\\\with\\\\backslash") == "namewithbackslash");
  CHECK(TensorToRegionName("name_without_slash") == "name_without_slash");
  CHECK(TensorToRegionName("abc123!@#") == "abc123!@#");
  CHECK(TensorToRegionName("") == "");
}


}}  // namespace triton::perfanalyzer
