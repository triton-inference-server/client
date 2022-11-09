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

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "client_backend/client_backend.h"
#include "constants.h"
#include "custom_load_manager.h"
#include "doctest.h"
#include "request_rate_manager.h"

namespace triton { namespace perfanalyzer {

class TestCustomLoadManager : public CustomLoadManager {
 public:
  std::unique_ptr<std::chrono::nanoseconds>& gen_duration_{
      RequestRateManager::gen_duration_};
  std::vector<std::chrono::nanoseconds>& schedule_{
      RequestRateManager::schedule_};
  std::string& request_intervals_file_{
      CustomLoadManager::request_intervals_file_};
  std::vector<std::chrono::nanoseconds>& custom_intervals_{
      CustomLoadManager::custom_intervals_};

  cb::Error ReadTimeIntervalsFile(
      const std::string& path,
      std::vector<std::chrono::nanoseconds>* contents) override
  {
    return cb::Error::Success;
  }
};

TEST_CASE("testing the InitCustomIntervals function")
{
  TestCustomLoadManager tclm{};

  SUBCASE("no file provided")
  {
    cb::Error result{tclm.InitCustomIntervals()};

    CHECK(result.Err() == SUCCESS);
    CHECK(tclm.schedule_.size() == 1);
    CHECK(tclm.schedule_[0] == std::chrono::nanoseconds(0));
  }

  SUBCASE("file provided")
  {
    tclm.request_intervals_file_ = "nonexistent_file.txt";
    tclm.gen_duration_ = std::make_unique<std::chrono::nanoseconds>(350000000);
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(100000000));
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(110000000));
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(130000000));

    cb::Error result{tclm.InitCustomIntervals()};

    CHECK(result.Err() == SUCCESS);
    CHECK(tclm.schedule_.size() == 5);
    CHECK(tclm.schedule_[0] == std::chrono::nanoseconds(0));
    CHECK(tclm.schedule_[1] == std::chrono::nanoseconds(100000000));
    CHECK(tclm.schedule_[2] == std::chrono::nanoseconds(210000000));
    CHECK(tclm.schedule_[3] == std::chrono::nanoseconds(340000000));
    CHECK(tclm.schedule_[4] == std::chrono::nanoseconds(440000000));
  }
}

TEST_CASE("testing the GetCustomRequestRate function")
{
  TestCustomLoadManager tclm{};
  double request_rate{0.0};

  SUBCASE("custom_intervals_ empty")
  {
    cb::Error result{tclm.GetCustomRequestRate(&request_rate)};

    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(result.Message() == "The custom intervals vector is empty");
  }

  SUBCASE("custom_intervals_ populated")
  {
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(100000000));
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(110000000));
    tclm.custom_intervals_.push_back(std::chrono::nanoseconds(130000000));

    cb::Error result{tclm.GetCustomRequestRate(&request_rate)};

    CHECK(result.Err() == SUCCESS);
    CHECK(request_rate == doctest::Approx(8.0));
  }
}

}}  // namespace triton::perfanalyzer
