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

#include <memory>

#include "concurrency_ctx_id_tracker.h"
#include "fifo_ctx_id_tracker.h"
#include "rand_ctx_id_tracker.h"

namespace triton { namespace perfanalyzer {

// Context ID tracker that is always available and returns random Context IDs
//
class CtxIdTrackerFactory {
 public:
  CtxIdTrackerFactory() = delete;

  /// Creates and returns a Context Id Tracker
  ///
  /// \param is_concurrency True if targetting Concurrency
  /// \param is_sequence_model True if the model is a sequence model
  /// \param serial_sequences True if in serial sequence mode
  ///
  static std::shared_ptr<ICtxIdTracker> CreateTracker(
      bool is_concurrency, bool is_sequence_model, bool serial_sequences)
  {
    if (is_concurrency) {
      if (is_sequence_model) {
        return std::make_shared<FifoCtxIdTracker>();
      } else {
        return std::make_shared<ConcurrencyCtxIdTracker>();
      }
    } else {
      if (is_sequence_model && serial_sequences) {
        return std::make_shared<FifoCtxIdTracker>();
      } else {
        return std::make_shared<RandCtxIdTracker>();
      }
    }
  }
};

}}  // namespace triton::perfanalyzer
