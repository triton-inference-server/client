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

#include <thread>
#include "doctest.h"
#include "idle_timer.h"

namespace triton { namespace perfanalyzer {

TEST_CASE("idle_timer: basic usage")
{
  IdleTimer timer;
  CHECK(timer.GetIdleTime() == 0);
  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  timer.Stop();
  CHECK(timer.GetIdleTime() > 0);
  timer.Reset();
  CHECK(timer.GetIdleTime() == 0);
}

TEST_CASE("idle_timer: restart when inactive")
{
  IdleTimer timer;
  CHECK(timer.GetIdleTime() == 0);
  timer.Restart();
  CHECK(timer.GetIdleTime() == 0);
  CHECK_NOTHROW(timer.Start());
}

TEST_CASE("idle_timer: restart when active")
{
  IdleTimer timer;
  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  timer.Restart();
  CHECK(timer.GetIdleTime() > 0);
  CHECK_NOTHROW(timer.Stop());
}

TEST_CASE("idle_timer: reset when active")
{
  IdleTimer timer;
  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  timer.Stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  timer.Reset();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(timer.GetIdleTime() == 0);
  CHECK_NOTHROW(timer.Stop());
  CHECK(timer.GetIdleTime() > 0);
}

TEST_CASE("idle_timer: double start")
{
  IdleTimer timer;
  timer.Start();
  CHECK_THROWS_AS(timer.Start(), const std::exception&);
}

TEST_CASE("idle_timer: stop without start")
{
  IdleTimer timer;
  CHECK_THROWS_AS(timer.Stop(), const std::exception&);
}


}}  // namespace triton::perfanalyzer
