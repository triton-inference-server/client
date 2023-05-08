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

#include <queue>

/// Interface for object that tracks context IDs
///
class ICtxIdTracker {
 public:
  // Reset the tracker using the provided input count
  //
  virtual void Reset(size_t count) = 0;

  // Restore the given ID into the tracker
  //
  virtual void Restore(size_t id) = 0;

  // Pick and return a Ctx ID
  //
  virtual size_t Get() = 0;

  // Returns true if there are Ctx IDs available to Get.
  virtual bool IsAvailable() = 0;
};

// Base class for CtxIdTrackers that track available IDs via a queue
//
class BaseQueueCtxIdTracker : public ICtxIdTracker {
 public:
  BaseQueueCtxIdTracker() = default;

  void Restore(size_t id) override { free_ctx_ids_.push(id); }

  size_t Get() override
  {
    if (!IsAvailable()) {
      throw std::runtime_error("free ctx id list is empty");
    }

    size_t ctx_id = free_ctx_ids_.front();
    free_ctx_ids_.pop();
    return ctx_id;
  }

  bool IsAvailable() override { return free_ctx_ids_.size() > 0; }

 protected:
  std::queue<size_t> free_ctx_ids_;

  // Erase all entries in the tracking queue
  //
  void Clear()
  {
    std::queue<size_t> empty;
    std::swap(free_ctx_ids_, empty);
  }
};

// Context ID Tracker that reuses IDs in a roughly round-robin manner using a
// FIFO
//
class FifoCtxIdTracker : public BaseQueueCtxIdTracker {
 public:
  FifoCtxIdTracker() = default;
  void Reset(size_t count) override
  {
    Clear();

    for (size_t i = 0; i < count; ++i) {
      free_ctx_ids_.push(i);
    }
  }
};

// Context ID Tracker that always returns context 0, but ensures that only X
// requests are outstanding at a time
//
class ConcurrencyCtxIdTracker : public BaseQueueCtxIdTracker {
 public:
  ConcurrencyCtxIdTracker() = default;
  void Reset(size_t count) override
  {
    Clear();

    for (size_t i = 0; i < count; ++i) {
      free_ctx_ids_.push(0);
    }
  }
};

// Context ID tracker that is always available and returns random Context IDs
//
class RandCtxIdTracker : public ICtxIdTracker {
 public:
  RandCtxIdTracker() = default;

  void Reset(size_t count) override { max = count; }

  void Restore(size_t id) override{};

  size_t Get() override { return rand() % max; };

  bool IsAvailable() override { return true; };

 private:
  size_t max = 0;
};
