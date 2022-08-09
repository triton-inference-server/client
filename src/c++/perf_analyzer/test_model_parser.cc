// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <rapidjson/document.h>
#include <cstdint>
#include "client_backend/client_backend.h"
#include "constants.h"
#include "doctest.h"
#include "model_parser.h"

namespace triton { namespace perfanalyzer {

class TestModelParser {
 public:
  static cb::Error GetInt(const rapidjson::Value& value, int64_t* integer_value)
  {
    ModelParser mp{};
    return mp.GetInt(value, integer_value);
  }
};

TEST_CASE("testing the GetInt function")
{
  int64_t integer_value{0};

  SUBCASE("valid string")
  {
    rapidjson::Value value("100");
    cb::Error result{TestModelParser::GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 100);
  }

  SUBCASE("invalid string, alphabet")
  {
    rapidjson::Value value("abc");
    cb::Error result{TestModelParser::GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(result.Message() == "unable to convert 'abc' to integer");
    CHECK(integer_value == 0);
  }

  SUBCASE("invalid string, number out of range")
  {
    rapidjson::Value value("9223372036854775808");
    cb::Error result{TestModelParser::GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(
        result.Message() ==
        "unable to convert '9223372036854775808' to integer");
    CHECK(integer_value == 0);
  }

  SUBCASE("valid int, lowest Int64")
  {
    rapidjson::Value value(2147483648);
    cb::Error result{TestModelParser::GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 2147483648);
  }

  SUBCASE("valid int, highest Int32")
  {
    rapidjson::Value value(2147483647);
    cb::Error result{TestModelParser::GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 2147483647);
  }

  SUBCASE("invalid floating point")
  {
    rapidjson::Value value(100.1);
    cb::Error result{TestModelParser::GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(result.Message() == "failed to parse the integer value");
    CHECK(integer_value == 0);
  }
}

}}  // namespace triton::perfanalyzer
