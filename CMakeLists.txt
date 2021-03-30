# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritonclient LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_CC_HTTP "Enable C++ HTTP client libraries" OFF)
option(TRITON_ENABLE_CC_GRPC "Enable C++ GRPC client libraries" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)

set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core)

if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC)
  add_subdirectory(src/c++/library)
endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC

if(TRITON_ENABLE_CC_HTTP)
  add_custom_target(cc-http ALL)
  add_dependencies(cc-http httpclient httpclient_static)
endif() # TRITON_ENABLE_CC_HTTP

if(TRITON_ENABLE_CC_GRPC)
  add_custom_target(cc-grpc ALL)
  add_dependencies(cc-grpc grpcclient grpcclient_static)
endif() # TRITON_ENABLE_CC_GRPC

#add_subdirectory(src/c++/library)
#if(NOT ${TRITON_CLIENT_SKIP_EXAMPLES})
#  add_subdirectory(examples)
#  add_subdirectory(perf_analyzer)
#  # FIXME: Control building the client test with a
#  # separate cmake flag.
#  add_subdirectory(tests)
#endif() # TRITON_CLIENT_SKIP_EXAMPLES
