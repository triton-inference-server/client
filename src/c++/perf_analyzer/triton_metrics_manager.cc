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

#include "triton_metrics_manager.h"
#include <stdexcept>

namespace triton { namespace perfanalyzer {

TritonMetricsManager::TritonMetricsManager(
    std::shared_ptr<clientbackend::ClientBackend> client_backend,
    uint64_t triton_metrics_interval_ms)
    : client_backend_(client_backend),
      triton_metrics_interval_ms_(triton_metrics_interval_ms)
{
}

TritonMetricsManager::~TritonMetricsManager()
{
  StopQueryingTritonMetrics();
}

void
TritonMetricsManager::StartQueryingTritonMetrics()
{
  should_keep_querying_ = true;
  query_loop_future_ = std::async(
      &TritonMetricsManager::QueryTritonMetricsEveryNMilliseconds, this);
}

void
TritonMetricsManager::QueryTritonMetricsEveryNMilliseconds()
{
  while (should_keep_querying_) {
    const auto& start{std::chrono::system_clock::now()};

    TritonMetrics triton_metrics{};
    clientbackend::Error err{client_backend_->TritonMetrics(triton_metrics)};
    if (err.IsOk() == false) {
      throw std::runtime_error(err.Message());
    }
    triton_metrics_per_timestamp_.emplace_back(
        start, std::move(triton_metrics));

    const auto& end{std::chrono::system_clock::now()};
    const auto& duration{end - start};
    const auto& remainder{
        std::chrono::milliseconds(triton_metrics_interval_ms_) - duration};

    if (remainder < std::chrono::nanoseconds::zero()) {
      std::cerr << "Triton metrics endpoint latency ("
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       duration)
                       .count()
                << "ms) is larger than the querying interval ("
                << triton_metrics_interval_ms_
                << "ms). Please try a larger querying interval "
                   "via `--triton-metrics-interval`."
                << std::endl;
    }

    query_loop_cv_.wait_for(query_loop_lock_, remainder);
  }
}

void
TritonMetricsManager::CheckQueryingStatus()
{
  if (query_loop_future_.wait_for(std::chrono::seconds(0)) ==
      std::future_status::ready) {
    query_loop_future_.get();
  }
}

void
TritonMetricsManager::StopQueryingTritonMetrics()
{
  should_keep_querying_ = false;
  query_loop_cv_.notify_one();
  if (query_loop_future_.valid()) {
    query_loop_future_.get();
  }
}

}}  // namespace triton::perfanalyzer
