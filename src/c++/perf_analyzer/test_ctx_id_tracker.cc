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

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>

#include "ctx_id_tracker.h"
#include "doctest.h"

namespace triton { namespace perfanalyzer {

TEST_CASE("CtxIdTrackers: FIFO")
{
  std::shared_ptr<ICtxIdTracker> tracker = std::make_shared<FifoCtxIdTracker>();

  // Reset will load up context IDs 0-9 into the queue and return them in order
  // on consecutive Get calls
  size_t count = 10;
  CHECK_FALSE(tracker->IsAvailable());
  tracker->Reset(count);
  CHECK(tracker->IsAvailable());
  for (size_t i = 0; i < count; i++) {
    CHECK(tracker->Get() == i);
  }

  // Manually restoring values should be returned in-order
  CHECK_FALSE(tracker->IsAvailable());
  tracker->Restore(7);
  CHECK(tracker->IsAvailable());
  tracker->Restore(13);
  CHECK(tracker->Get() == 7);
  CHECK(tracker->Get() == 13);

  // A reset should throw away any values on the old list
  tracker->Reset(10);
  tracker->Reset(1);
  tracker->Get();
  CHECK(!tracker->IsAvailable());

  // Calling Get when not available should Throw
  CHECK_THROWS_AS(tracker->Get(), const std::exception&);
}

TEST_CASE("CtxIdTrackers: Conc")
{
  std::shared_ptr<ICtxIdTracker> tracker =
      std::make_shared<ConcurrencyCtxIdTracker>();

  // Reset will load up 10 instances of context IDs 0 into the queue and return
  // them in order on consecutive Get calls
  size_t count = 10;
  tracker->Reset(count);
  for (size_t i = 0; i < count; i++) {
    CHECK(tracker->Get() == 0);
  }

  // Manually restoring values should be returned in-order
  CHECK_FALSE(tracker->IsAvailable());
  tracker->Restore(7);
  tracker->Restore(13);
  CHECK(tracker->IsAvailable());
  CHECK(tracker->Get() == 7);
  CHECK(tracker->Get() == 13);

  // A reset should throw away any values on the old list
  tracker->Reset(10);
  tracker->Reset(1);
  tracker->Get();
  CHECK(!tracker->IsAvailable());

  // Calling Get when not available should Throw
  CHECK_THROWS_AS(tracker->Get(), const std::exception&);
}

TEST_CASE("CtxIdTrackers: Rand")
{
  std::shared_ptr<ICtxIdTracker> tracker = std::make_shared<RandCtxIdTracker>();

  // IsAvailable is always true for this class
  CHECK(tracker->IsAvailable());

  // Reset should define the bounds of random CTX id picking
  size_t count = 10;
  tracker->Reset(count);

  // Restore should have no impact on this class.
  tracker->Restore(9999);

  std::vector<size_t> result_count(10, 0);

  for (size_t i = 0; i < 1000; i++) {
    auto x = tracker->Get();
    REQUIRE((x < 10 && x >= 0));
    result_count[x]++;
  }

  double mean = std::accumulate(result_count.begin(), result_count.end(), 0.0) /
                result_count.size();
  double variance = 0;
  for (size_t i = 0; i < result_count.size(); i++) {
    variance += std::pow(result_count[i] - mean, 2);
  }
  variance /= result_count.size();

  // Confirm that the distrubution of the picked CTX IDs is random
  CHECK((variance > 10 && variance < 100));
}


}}  // namespace triton::perfanalyzer
