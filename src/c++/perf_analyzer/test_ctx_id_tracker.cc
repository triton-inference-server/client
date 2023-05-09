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
  size_t max;

  auto check_range_and_variance = [&]() {
    size_t num_trials = 1000;

    std::vector<size_t> results(max, 0);
    for (size_t i = 0; i < num_trials; i++) {
      auto x = tracker->Get();
      REQUIRE((x < max && x >= 0));
      results[x]++;
    }

    // Confirm that the distrubution of the picked CTX IDs is random
    double mean =
        std::accumulate(results.begin(), results.end(), 0.0) / results.size();
    double variance = 0;
    for (size_t i = 0; i < results.size(); i++) {
      variance += std::pow(results[i] - mean, 2);
    }
    variance /= results.size();
    CHECK((variance > 10 && variance < 100));
  };

  // IsAvailable is always true for this class
  CHECK(tracker->IsAvailable());

  // Reset should define the bounds of random CTX id picking
  max = 10;
  tracker->Reset(max);
  // Restore should have no impact on this class.
  tracker->Restore(9999);
  check_range_and_variance();


  // Reset should RE-define the bounds of random CTX id picking
  max = 5;
  tracker->Reset(max);
  check_range_and_variance();
}


}}  // namespace triton::perfanalyzer
