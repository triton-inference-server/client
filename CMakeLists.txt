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
option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" OFF)
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" OFF)
option(TRITON_ENABLE_PERF_ANALYZER "Enable Performance Analyzer" OFF)
option(TRITON_ENABLE_EXAMPLES "Include examples in build" OFF)
option(TRITON_ENABLE_TESTS "Include tests in build" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_THIRD_PARTY_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/third_party repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-third-party
  GIT_REPOSITORY https://github.com/triton-inference-server/third_party.git
  GIT_TAG ${TRITON_THIRD_PARTY_REPO_TAG}
  GIT_SHALLOW ON
)
set(TRITON_THIRD_PARTY_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third-party)
FetchContent_MakeAvailable(repo-third-party)

# Need to use ExternalProject for our builds so that we can get the
# correct dependencies between our components and the ExternalProject
# dependencies (found in the third_party repo)
if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
  include(ExternalProject)

  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cc-clients/install)
  else()
    set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  endif()

  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
  if (OPENSSL_ROOT_DIR)
    set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
  endif()

  set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "")
  if (CMAKE_TOOLCHAIN_FILE)
    set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "-DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}")
  endif()

  set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "")
  if (VCPKG_TARGET_TRIPLET)
    set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "-DVCPKG_TARGET_TRIPLET:STRING=${VCPKG_TARGET_TRIPLET}")
  endif()

  # Location where protobuf-config.cmake will be installed varies by
  # platform
  if (WIN32)
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/cmake")
  else()
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/lib/cmake/protobuf")
  endif()

  set(_cc_client_depends "")
  if(${TRITON_ENABLE_CC_HTTP})
    set(_cc_client_depends ${_cc_client_depends} curl)
  endif() # TRITON_ENABLE_CC_HTTP
  if(${TRITON_ENABLE_CC_GRPC})
    set(_cc_client_depends ${_cc_client_depends} grpc protobuf)
  endif() # TRITON_ENABLE_CC_GRPC

  ExternalProject_Add(cc-clients
    PREFIX cc-clients
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/c++"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cc-clients"
    CMAKE_CACHE_ARGS
      ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
      ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
      ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
      -DCURL_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/lib/cmake/CURL
      -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
      -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
      -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/lib/cmake/c-ares
      -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
      -DTRITON_ENABLE_CC_HTTP:BOOL=${TRITON_ENABLE_CC_HTTP}
      -DTRITON_ENABLE_CC_GRPC:BOOL=${TRITON_ENABLE_CC_GRPC}
      -DTRITON_ENABLE_PERF_ANALYZER:BOOL=${TRITON_ENABLE_PERF_ANALYZER}
      -DTRITON_ENABLE_EXAMPLES:BOOL=${TRITON_ENABLE_EXAMPLES}
      -DTRITON_ENABLE_TESTS:BOOL=${TRITON_ENABLE_TESTS}
      -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
    DEPENDS ${_cc_client_depends}
  )
endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER


# fixme
if(TRITON_ENABLE_PYTHON)
  add_subdirectory(src/python/library)
  if(TRITON_ENABLE_EXAMPLES)
    add_subdirectory(src/python/examples)
  endif() # TRITON_ENABLE_EXAMPLES
endif() # TRITON_ENABLE_PYTHON
