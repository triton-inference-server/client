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

#include "doctest.h"
#include "perf_utils.h"
#include "test_utils.h"

namespace triton { namespace perfanalyzer {

/// Helper class to test perf_utils.cc
///
class TestPerfUtils {
  public:
    /// Given a distributionType and request rate, confirm that request pattern matches
    /// what is expected.
    /// 
    static void TestDistribution(Distribution distribution_type, uint32_t request_rate) {
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

        std::chrono::nanoseconds ns_in_one_second = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)); 
        expected_avg = ns_in_one_second.count() / request_rate;
        
        if (distribution_type == CONSTANT) {
            expected_variance = 0;
        }
        else {
            // By definition, variance = mean for poisson
            expected_variance = expected_avg;
        }

        CHECK(avg == doctest::Approx(expected_avg).epsilon(0.005));
        CHECK(variance == doctest::Approx(expected_variance).epsilon(0.005));
    }


  private:

    static std::function<std::chrono::nanoseconds(std::mt19937&)> GetDistributionFunction(Distribution type, uint32_t request_rate) {
        std::function<std::chrono::nanoseconds(std::mt19937&)> distributionFunction;

        if (type == CONSTANT) {
            distributionFunction = ScheduleDistribution<CONSTANT>(request_rate);
        }
        else if (type == POISSON) {
            distributionFunction = ScheduleDistribution<POISSON>(request_rate);
        }
        else {
            throw std::invalid_argument("Unexpected distribution type");
        }
        return distributionFunction;
    }
};

/// Test all distributions across various request rates
///
TEST_CASE("test_distribution") {
    std::vector<Distribution> distTypes{CONSTANT, POISSON};
    std::vector<uint32_t> requestRates{10,100,1000,10000};

    for (auto dist : distTypes) {
        for (auto rate : requestRates) {
            TestPerfUtils::TestDistribution(dist, rate);
        }
    }
}

}}
