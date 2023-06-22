// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <string>

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
namespace triton { namespace perfanalyzer {

const std::string SHA{STRINGIFY(GIT_SHA)};
const std::string RELEASE{STRINGIFY(PERF_ANALYZER_VERSION)};
const std::string VERSION{RELEASE + " (commit " + SHA + ")"};

constexpr static const uint32_t SUCCESS = 0;

constexpr static const uint32_t STABILITY_ERROR = 2;
constexpr static const uint32_t OPTION_ERROR = 3;

constexpr static const uint32_t GENERIC_ERROR = 99;

const double DELAY_PCT_THRESHOLD{1.0};

/// Different measurement modes possible.
enum MeasurementMode { TIME_WINDOWS = 0, COUNT_WINDOWS = 1 };

}}  // namespace triton::perfanalyzer
