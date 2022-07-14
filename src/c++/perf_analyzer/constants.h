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

#include <cstdint>


namespace triton { namespace perfanalyzer {

namespace pa = triton::perfanalyzer;

constexpr static const uint32_t SUCCESS = 0;

constexpr static const uint32_t EXIT_SIGNAL = 2;
constexpr static const uint32_t INVALID_USAGE = 3;
constexpr static const uint32_t LOAD_MANAGER_ERROR = 4;
constexpr static const uint32_t DATA_ERROR = 5;
constexpr static const uint32_t FILE_READ_ERROR = 6;
constexpr static const uint32_t INCONSISTENT_SETTING_ERROR = 7;
constexpr static const uint32_t OPTION_ERROR = 8;
constexpr static const uint32_t PARSE_ERROR = 9;
constexpr static const uint32_t CUDA_ERROR = 10;
constexpr static const uint32_t GPU_ERROR = 11;
constexpr static const uint32_t DATA_TYPE_ERROR = 12;


constexpr static const uint32_t DEFAULT_ERROR = 99;
// constexpr static const uint32_t UNSUPPORTED_ERROR = 53;
// constexpr static const uint32_t CREATION_ERROR = 54;
// constexpr static const uint32_t PROFILE_ERROR = 55;
// constexpr static const uint32_t TRITON_SERVER_ERROR = 56;


}}  // namespace triton::perfanalyzer
