// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace triton { namespace perfanalyzer {

// Holds the configuration for a worker thread
struct ThreadConfig {
  ThreadConfig(size_t thread_id) : thread_id_(thread_id) {}

  // ID of corresponding worker thread
  size_t thread_id_{0};

  // The concurrency level that the worker should produce
  // TPA-69: This is only used in concurrency mode and shouldn't be visible in
  // other modes
  size_t concurrency_{0};

  // The number of sequences owned by this worker
  // TPA-69: This is only used in request-rate mode and shouldn't be visible in
  // other modes
  uint32_t num_sequences_{1};

  // How many requests to generate before stopping. If 0, generate indefinitely
  size_t num_requests_{0};

  // The starting sequence stat index for this worker
  size_t seq_stat_index_offset_{0};

  // Whether or not the thread is issuing new inference requests
  bool is_paused_{false};
};


}}  // namespace triton::perfanalyzer
