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

#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

namespace triton { namespace perfanalyzer {

/// This class will create a thread that will raise an error after a fixed
/// amount of time, unless the stop function is called.
///
/// It can be used to detect livelock/deadlock cases in tests so that the test
/// will be guarenteed to finish instead of hang
///
class TestWatchDog {
 public:
  /// Create the watchdog
  ///
  /// @param max_time_ms How long (in milliseconds) until this watchdog will
  /// raise an error
  TestWatchDog(unsigned int max_time_ms) { start(max_time_ms); }

  /// Stop the watchdog so that it will not raise any errors
  ///
  void stop()
  {
    running_ = false;
    thread_.join();
  }

 private:
  uint sleep_interval_ms{40};
  uint max_time_ms_;
  std::atomic<unsigned int> timer_;
  std::atomic<bool> running_;
  std::thread thread_;

  void start(unsigned int max_time_ms)
  {
    max_time_ms_ = max_time_ms;
    timer_ = 0;
    running_ = true;
    thread_ = std::thread(&TestWatchDog::loop, this);
  }

  void loop()
  {
    while (running_) {
      if (timer_ >= max_time_ms_) {
        running_ = false;
        REQUIRE_MESSAGE(false, "WATCHDOG TIMEOUT!");
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval_ms));
      timer_ += sleep_interval_ms;
    }
  }
};

/// Calculate the average of a vector of integers
///
static double
CalculateAverage(const std::vector<int64_t>& values)
{
  double avg =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  return avg;
}

/// Calculate the variance of a vector of integers
///
static double
CalculateVariance(const std::vector<int64_t>& values, double average)
{
  double tmp = 0;
  for (auto value : values) {
    tmp += (value - average) * (value - average) / values.size();
  }
  double variance = std::sqrt(tmp);
  return variance;
}

}}  // namespace triton::perfanalyzer
