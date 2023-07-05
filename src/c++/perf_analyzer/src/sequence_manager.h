// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "client_backend/client_backend.h"
#include "data_loader.h"
#include "sequence_status.h"

namespace triton { namespace perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class NaggyMockSequenceManager;
#endif

/// Manages operations related to preparing requests to sequence models.
///
class SequenceManager {
 public:
  /// Constructs the sequence manager object. Involves initializing the
  /// distribution for randomly assigning input data streams to new sequences.
  /// \param start_sequence_id See associated data member description.
  /// \param sequence_id_range See associated data member description.
  /// \param sequence_length See associated data member description.
  /// \param sequence_length_specified See associated data member description.
  /// \param sequence_length_variation See associated data member description.
  /// \param using_json_data See associated data member description.
  /// \param data_loader See associated data member description.
  /// \return The constructed sequence manager object.
  ///
  SequenceManager(
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      const size_t sequence_length, const bool sequence_length_specified,
      const double sequence_length_variation, const bool using_json_data,
      std::shared_ptr<DataLoader> data_loader);

  /// Initializes the sequence statuses data structure.
  /// \param num_sequence_statuses The number of sequence status objects to
  /// create.
  ///
  void InitSequenceStatuses(size_t num_sequence_statuses);

  /// Gets a non-const reference to the mutex for the specified sequence status
  /// object.
  /// \param sequence_status_index The index of the sequence status object.
  /// \return A non-const reference to the mutex for the specified sequence
  /// status object.
  ///
  std::mutex& GetMutex(size_t sequence_status_index);

  /// Gets the data stream ID for the specified sequence status object.
  /// \param sequence_status_index The index of the sequence status object.
  /// \return The data stream ID for the specified sequence status object.
  ///
  const uint64_t GetDataStreamID(size_t sequence_status_index) const;

  /// Gets the remaining queries for the specified sequence status object.
  /// \param sequence_status_index The index of the sequence status object.
  /// \return The remaining queries for the specified sequence status object.
  ///
  const size_t GetRemainingQueries(size_t sequence_status_index) const;

  /// Sets the remaining queries for the specified sequence status object.
  /// \param sequence_status_index The index of the sequence status object.
  /// \param remaining_queries The new value of the remaining queries for the
  /// specified sequence status object.
  ///
  void SetRemainingQueries(
      size_t sequence_status_index, size_t remaining_queries);

  /// Decrements the remaining queries for the specified sequence status object.
  /// \param sequence_status_index The index of the sequence status object.
  ///
  void DecrementRemainingQueries(size_t sequence_status_index);

  /// Gets the number of sequence status objects in the sequence statuses data
  /// structure.
  /// \param sequence_status_index The index of the sequence status object.
  /// \return The number of sequence status objects in the sequence statuses
  /// data structure.
  ///
  const size_t GetNumSequenceStatuses() const;

  /// Sets options related to a single request to a sequence model.
  /// \param seq_stat_index The index for the sequence status object that is
  /// having its options set.
  /// \param options The options object for the request that is being prepared.
  ///
  virtual void SetInferSequenceOptions(
      const uint32_t seq_stat_index,
      std::unique_ptr<cb::InferOptions>& options);

  /// Gets the sequence length for the specified sequence status object.
  /// \param sequence_status_index The index of the sequence status object.
  /// \return The sequence length for the specified sequence status object.
  ///
  const size_t GetSequenceLength(size_t sequence_status_index) const;

 private:
  /// Initializes values for a sequence status object.
  /// \param seq_stat_index The index for the sequence status object that is
  /// being initialized.
  ///
  virtual void InitNewSequence(int seq_stat_index);

  /// Determines an appropriate next sequence ID for a renewed sequence status
  /// object.
  /// \param seq_stat_index The index for the sequence for which a request is
  /// being prepared.
  /// \return The potentially new sequence ID to be used by a renewed sequence
  /// status object.
  ///
  virtual uint64_t GetNextSeqId(int seq_stat_index);

  virtual uint64_t GetNewDataStreamId()
  {
    return distribution_(rng_generator_);
  }

  /// Generates a random sequence length based on a threshold.
  /// \param offset_ratio The offset ratio/threshold of the generated length.
  /// \return A random sequence length.
  ///
  virtual size_t GetRandomSequenceLength(double offset_ratio);

  /// Data structure holding sequence status objects
  ///
  std::vector<std::shared_ptr<SequenceStatus>> sequence_statuses_{};

  /// Current sequence id (for issuing new sequences)
  ///
  std::atomic<uint64_t> curr_seq_id_{0};

  /// Data loader to be used for various sequence operations.
  ///
  std::shared_ptr<DataLoader> data_loader_{nullptr};

  /// The starting sequence ID to be used for iterating through valid sequence
  /// IDs.
  ///
  const uint64_t start_sequence_id_{0};

  /// The maximum sequence ID to be used for iterating through valid sequence
  /// IDs.
  ///
  const uint64_t sequence_id_range_{0};

  /// The base length of new sequences.
  ///
  const size_t sequence_length_{0};

  /// Whether the user specified the sequence length.
  ///
  const bool sequence_length_specified_{false};

  /// The percentage variation in length of sequences using autogenerated data
  /// as input.
  ///
  const double sequence_length_variation_{0.0};

  /// Indicates whether to generate sequence request input data or read it from
  /// a JSON file.
  ///
  const bool using_json_data_{false};

  /// The distribution for randomly assigning new sequences a data stream in the
  /// input data JSON.
  ///
  std::uniform_int_distribution<uint64_t> distribution_;

  /// The random number generator for randomly assigning new sequences a data
  /// stream in the input data JSON.
  ///
  std::default_random_engine rng_generator_{};

#ifndef DOCTEST_CONFIG_DISABLE
  friend NaggyMockSequenceManager;

 public:
  SequenceManager() = default;
#endif
};

}}  // namespace triton::perfanalyzer
