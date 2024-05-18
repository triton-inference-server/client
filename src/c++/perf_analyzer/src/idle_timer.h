// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <chrono>
#include <mutex>
#include <stdexcept>

namespace triton { namespace perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class TestLoadManager;
#endif


/// Class to track idle periods of time
///
class IdleTimer {
 public:
  void Start()
  {
    std::lock_guard<std::mutex> lk(mtx_);
    StartImpl();
  }

  void Stop()
  {
    std::lock_guard<std::mutex> lk(mtx_);
    StopImpl();
  }

  /// Reset the time counter, and restart the timer if it is active
  ///
  void Reset()
  {
    Restart();
    idle_ns_ = 0;
  }

  /// Returns the number of nanoseconds this timer has counted as being idle
  /// If the timer was already active, then it will first stop (and count the
  /// pending time), and then start back up
  ///
  uint64_t GetIdleTime()
  {
    Restart();
    return idle_ns_;
  }

 private:
  std::mutex mtx_;
  uint64_t idle_ns_{0};
  bool is_idle_{false};
  std::chrono::_V2::steady_clock::time_point start_time_;

  void Restart()
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (is_idle_) {
      StopImpl();
      StartImpl();
    }
  }

  void StartImpl()
  {
    if (is_idle_) {
      throw std::runtime_error("Can't start a timer that is already active\n");
    }

    is_idle_ = true;
    start_time_ = std::chrono::steady_clock::now();
  }

  void StopImpl()
  {
    if (!is_idle_) {
      throw std::runtime_error("Can't stop a timer that isn't active\n");
    }

    is_idle_ = false;
    auto end = std::chrono::steady_clock::now();
    auto duration = end - start_time_;
    idle_ns_ += duration.count();
  }


#ifndef DOCTEST_CONFIG_DISABLE
  friend TestLoadManager;
#endif
};

}}  // namespace triton::perfanalyzer
