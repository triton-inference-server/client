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

#include <memory>

#include "ctx_id_tracker.h"
#include "doctest.h"

namespace triton { namespace perfanalyzer {


TEST_CASE("TKG STD")
{
  std::shared_ptr<ICtxIdTracker> tracker = std::make_shared<StdCtxIdTracker>();
  size_t count = 10;
  tracker->Reset(count);
  for (size_t i = 0; i < count; i++) {
    CHECK(tracker->Get() == i);
  }

  CHECK_FALSE(tracker->IsAvailable());
  tracker->Restore(7);
  tracker->Restore(13);
  CHECK(tracker->IsAvailable());
  CHECK(tracker->Get() == 7);
  CHECK(tracker->Get() == 13);
}

TEST_CASE("TKG CONC")
{
  std::shared_ptr<ICtxIdTracker> tracker = std::make_shared<ConcCtxIdTracker>();
  size_t count = 10;
  tracker->Reset(count);
  for (size_t i = 0; i < count; i++) {
    CHECK(tracker->Get() == 0);
  }
  CHECK_FALSE(tracker->IsAvailable());
  tracker->Restore(7);
  tracker->Restore(13);
  CHECK(tracker->IsAvailable());
  CHECK(tracker->Get() == 7);
  CHECK(tracker->Get() == 13);
}

TEST_CASE("TKG RAND")
{
  std::shared_ptr<ICtxIdTracker> tracker = std::make_shared<RandCtxIdTracker>();
  CHECK(tracker->IsAvailable());

  size_t count = 10;
  tracker->Reset(count);
  CHECK(tracker->IsAvailable());
  tracker->Restore(100);
  CHECK(tracker->IsAvailable());

  std::vector<size_t> result_count(10, 0);

  for (size_t i = 0; i < 1000; i++) {
    auto x = tracker->Get();
    REQUIRE((x < 10 && x >= 0));
    result_count[x]++;
  }

  // FIXME confirm distribution not perfect
}


}}  // namespace triton::perfanalyzer
