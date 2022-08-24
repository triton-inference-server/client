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

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include "client_backend/client_backend.h"
#include "triton_metrics.h"

namespace triton { namespace perfanalyzer {

// FIXME: pull all this stuff into implementation file :(
// FIXME: add testing for methods here
class TritonMetricsManager {
 public:
  TritonMetricsManager(
      std::shared_ptr<clientbackend::ClientBackend> client_backend,
      uint64_t triton_metrics_interval_ms)
      : client_backend_(client_backend),
        triton_metrics_interval_ms_(triton_metrics_interval_ms)
  {
  }

  ~TritonMetricsManager() { StopQueryingTritonMetrics(); }

  void StartQueryingTritonMetrics()
  {
    should_keep_querying_ = true;
    query_triton_metrics_main_loop_future_ = std::async(
        &TritonMetricsManager::QueryTritonMetricsEveryNMilliseconds, this);
  }

  void QueryTritonMetricsEveryNMilliseconds()
  {
    while (should_keep_querying_) {
      std::future<void> f{
          std::async(&TritonMetricsManager::QueryTritonMetrics, this)};
      query_triton_metrics_futures_.push_back(std::move(f));
      std::this_thread::sleep_for(
          std::chrono::milliseconds(triton_metrics_interval_ms_));
    }
  }

  void QueryTritonMetrics()
  {
    const auto& current_time{std::chrono::system_clock::now()};
    TritonMetrics triton_metrics{};
    client_backend_->TritonMetrics(triton_metrics);
    std::lock_guard<std::mutex> triton_metrics_per_timestamp_lock(
        triton_metrics_per_timestamp_mutex_);
    triton_metrics_per_timestamp_.emplace_back(
        current_time, std::move(triton_metrics));
  }

  void StopQueryingTritonMetrics()
  {
    should_keep_querying_ = false;
    for (const auto& f : query_triton_metrics_futures_) {
      f.wait();
    }
    query_triton_metrics_main_loop_future_.wait();
  }

 private:
  std::shared_ptr<clientbackend::ClientBackend> client_backend_{nullptr};
  uint64_t triton_metrics_interval_ms_{0};
  std::vector<std::pair<
      std::chrono::time_point<std::chrono::system_clock>, TritonMetrics>>
      triton_metrics_per_timestamp_{};
  std::mutex triton_metrics_per_timestamp_mutex_{};
  bool should_keep_querying_{false};
  std::future<void> query_triton_metrics_main_loop_future_{};
  std::vector<std::future<void>> query_triton_metrics_futures_{};
};

}}  // namespace triton::perfanalyzer
