# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(perf-analyzer LANGUAGES C CXX)

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

set(TRITON_VERSION "0.0.0" CACHE STRING "Version for the clients")
set(PERF_ANALYZER_VERSION ${TRITON_VERSION} CACHE STRING "Build Version for Perf Analyzer")

#
# Perf Analyzer Options
#
option(TRITON_PACKAGE_PERF_ANALYZER "Include Perf Analyzer in pip wheel" ON)
option(TRITON_ENABLE_PERF_ANALYZER_C_API "Enable Performance Analyzer C API" ON)
option(TRITON_ENABLE_PERF_ANALYZER_TFS "Enable TensorFlow Serving support for Performance Analyzer" ON)
option(TRITON_ENABLE_PERF_ANALYZER_TS "Enable TorchServe support for Performance Analyzer" ON)
option(TRITON_ENABLE_PERF_ANALYZER_OPENAI "Enable OpenAI support for Performance Analyzer" ON)

#
# Client Options
#
option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" ON)
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" ON)
option(TRITON_ENABLE_PYTHON_HTTP "Build Python HTTP client libraries" OFF)
option(TRITON_ENABLE_PYTHON_GRPC "Build Python GRPC client libraries" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" ON)
option(TRITON_ENABLE_ZLIB "Include ZLIB library in build" ON)

#
# Github branch options
#
set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_CLIENT_REPO_TAG "fpetrini-migrate-pa" CACHE STRING "Tag for triton-inference-server/core repo")

#
# Install locations
#
set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/cc_clients/src/cc_clients-build/third-party/" CACHE STRING "Location of third-party build")
# If install prefix is default, it will install files deep in the build tree
set(CMAKE_INSTALL_PREFIX "/opt/tritonserver/" CACHE STRING "Triton clients install location")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(WIN32)
  message(FATAL_ERROR "perf_analyzer is not currently supported on Windows because "
          "is requires functionalities that are UNIX specific.")
endif()

if(NOT(${TRITON_ENABLE_CC_HTTP} AND ${TRITON_ENABLE_CC_GRPC}))
  message(FATAL_ERROR "perf_analyzer requires both http and grpc client libraries.")
endif()

#
# Dependencies
#
include(ExternalProject)
ExternalProject_Add(
  cc_clients
  PREFIX ${CMAKE_BINARY_DIR}/cc_clients
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/client.git
  GIT_TAG ${TRITON_CLIENT_REPO_TAG}
  GIT_SHALLOW ON
  CMAKE_CACHE_ARGS
      ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
      ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
      ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
      -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
      -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
      -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
      -DTRITON_ENABLE_CC_HTTP:BOOL=${TRITON_ENABLE_CC_HTTP}
      -DTRITON_ENABLE_CC_GRPC:BOOL=${TRITON_ENABLE_CC_GRPC}
      -DTRITON_ENABLE_PYTHON_HTTP:BOOL=${TRITON_ENABLE_PYTHON_HTTP}
      -DTRITON_ENABLE_PYTHON_HTTP:BOOL=${TRITON_ENABLE_PYTHON_GRPC}
      -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
      -DTRITON_ENABLE_ZLIB:BOOL=${TRITON_ENABLE_ZLIB}
      -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
      -DTRITON_THIRD_PARTY_INSTALL_PREFIX:STRING=${TRITON_THIRD_PARTY_INSTALL_PREFIX}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_INSTALL_PREFIX}
  INSTALL_COMMAND ""
)

ExternalProject_Add(
  perf_analyzer
  PREFIX ${CMAKE_BINARY_DIR}/perf_analyzer
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/perf_analyzer
  CMAKE_CACHE_ARGS
      ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
      ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
      ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
      -DTRITON_ENABLE_CC_HTTP:BOOL=${TRITON_ENABLE_CC_HTTP}
      -DTRITON_ENABLE_CC_GRPC:BOOL=${TRITON_ENABLE_CC_GRPC}
      -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
      -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
      -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
      -DTRITON_ENABLE_ZLIB:BOOL=${TRITON_ENABLE_ZLIB}
      -DTRITON_ENABLE_PERF_ANALYZER_C_API:BOOL=${TRITON_ENABLE_PERF_ANALYZER_C_API}
      -DTRITON_ENABLE_PERF_ANALYZER_TFS:BOOL=${TRITON_ENABLE_PERF_ANALYZER_TFS}
      -DTRITON_ENABLE_PERF_ANALYZER_TS:BOOL=${TRITON_ENABLE_PERF_ANALYZER_TS}
      -DTRITON_ENABLE_PERF_ANALYZER_OPENAI:BOOL=${TRITON_ENABLE_PERF_ANALYZER_OPENAI}
      -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
      -DTRITON_THIRD_PARTY_INSTALL_PREFIX:STRING=${TRITON_THIRD_PARTY_INSTALL_PREFIX}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_INSTALL_PREFIX}
  DEPENDS cc_clients
)

# FIXME: This is a WAR for now. We shouldn't have to re-clone the client repo
# Everything to build the wheel should be present in the cc_clients folder.
# PA needs to have its own build_wheel.py script.

# Build python client after perf_analyzer has been installed to a known location
# so that it can be packaged in the python client pip wheel.
if(TRITON_ENABLE_PYTHON_HTTP OR TRITON_ENABLE_PYTHON_GRPC)
  ExternalProject_Add(
    python_clients
    PREFIX ${CMAKE_BINARY_DIR}/python_clients
    GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/client.git
    GIT_TAG ${TRITON_CLIENT_REPO_TAG}
    GIT_SHALLOW ON
    CMAKE_CACHE_ARGS
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
        ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
        -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
        -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
        -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
        -DTRITON_ENABLE_CC_HTTP:BOOL=OFF
        -DTRITON_ENABLE_CC_GRPC:BOOL=OFF
        -DTRITON_ENABLE_PYTHON_HTTP:BOOL=${TRITON_ENABLE_PYTHON_HTTP}
        -DTRITON_ENABLE_PYTHON_HTTP:BOOL=${TRITON_ENABLE_PYTHON_GRPC}
        -DTRITON_PACKAGE_PERF_ANALYZER:BOOL=${TRITON_PACKAGE_PERF_ANALYZER}
        -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
        -DTRITON_ENABLE_ZLIB:BOOL=${TRITON_ENABLE_ZLIB}
        -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
        -DTRITON_THIRD_PARTY_INSTALL_PREFIX:STRING=${TRITON_THIRD_PARTY_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND ""
    DEPENDS perf_analyzer
)
endif()

