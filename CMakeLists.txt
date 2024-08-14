# Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Options
#
set(TRITON_VERSION "0.0.0" CACHE STRING "Version for the clients")
option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" OFF)
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" OFF)
option(TRITON_ENABLE_PYTHON_HTTP "Enable Python HTTP client libraries" OFF)
option(TRITON_ENABLE_PYTHON_GRPC "Enable Python GRPC client libraries" OFF)
option(TRITON_ENABLE_JAVA_HTTP "Enable JAVA HTTP client libraries" OFF)
option(TRITON_ENABLE_EXAMPLES "Include examples in build" OFF)
option(TRITON_ENABLE_TESTS "Include tests in build" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)
option(TRITON_ENABLE_ZLIB "Include ZLIB library in build" ON)

# Package Perf Analyzer with the Python Client -- Intended to be used by PA repo
option(TRITON_PACKAGE_PERF_ANALYZER "Include Perf Analyzer in pip wheel" OFF)

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_THIRD_PARTY_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/third_party repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT TRITON_ENABLE_PYTHON_GRPC)
  set(TRITON_COMMON_ENABLE_PROTOBUF_PYTHON OFF)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-third-party
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/third_party.git
  GIT_TAG ${TRITON_THIRD_PARTY_REPO_TAG}
  GIT_SHALLOW ON
)
set(TRITON_THIRD_PARTY_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third-party)
FetchContent_MakeAvailable(repo-third-party)

# Some libs are installed to ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib64 instead
# of ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib on Centos
set (LIB_DIR "lib")
# /etc/os-release does not exist on Windows
if(EXISTS "/etc/os-release")
  file(STRINGS /etc/os-release DISTRO REGEX "^NAME=")
  string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" DISTRO "${DISTRO}")
  message(STATUS "Distro Name: ${DISTRO}")
  if(DISTRO MATCHES "CentOS.*")
    set (LIB_DIR "lib64")
  endif()
endif()

# Need to use ExternalProject for our builds so that we can get the
# correct dependencies between our components and the ExternalProject
# dependencies (found in the third_party repo)
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
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/${LIB_DIR}/cmake/protobuf")
endif()

if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC)

  set(_cc_client_depends re2)
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
      -DCURL_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/${LIB_DIR}/cmake/CURL
      -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
      -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
      -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
      -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
      -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares
      -DGTEST_ROOT:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/googletest
      -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
      -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
      -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
      -DTRITON_ENABLE_CC_HTTP:BOOL=${TRITON_ENABLE_CC_HTTP}
      -DTRITON_ENABLE_CC_GRPC:BOOL=${TRITON_ENABLE_CC_GRPC}
      -DTRITON_ENABLE_EXAMPLES:BOOL=${TRITON_ENABLE_EXAMPLES}
      -DTRITON_ENABLE_TESTS:BOOL=${TRITON_ENABLE_TESTS}
      -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
      -DTRITON_ENABLE_ZLIB:BOOL=${TRITON_ENABLE_ZLIB}
      -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
    DEPENDS ${_cc_client_depends}
  )
endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC

if(TRITON_ENABLE_PYTHON_HTTP OR TRITON_ENABLE_PYTHON_GRPC)
  set(_py_client_depends re2)
  if(${TRITON_ENABLE_PYTHON_GRPC})
    set(_py_client_depends ${_py_client_depends} grpc protobuf)
  endif() # TRITON_ENABLE_PYTHON_GRPC

  ExternalProject_Add(python-clients
    PREFIX python-clients
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/python"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/python-clients"
    CMAKE_CACHE_ARGS
      ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
      ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
      ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
      -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
      -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
      -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
      -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares
      -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
      -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
      -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
      -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
      -DTRITON_VERSION:STRING=${TRITON_VERSION}
      -DTRITON_ENABLE_PYTHON_HTTP:BOOL=${TRITON_ENABLE_PYTHON_HTTP}
      -DTRITON_ENABLE_PYTHON_GRPC:BOOL=${TRITON_ENABLE_PYTHON_GRPC}
      -DTRITON_ENABLE_EXAMPLES:BOOL=${TRITON_ENABLE_EXAMPLES}
      -DTRITON_ENABLE_TESTS:BOOL=${TRITON_ENABLE_TESTS}
      -DTRITON_PACKAGE_PERF_ANALYZER:BOOL=${TRITON_PACKAGE_PERF_ANALYZER}
      -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
    DEPENDS ${_py_client_depends}
  )
endif() # TRITON_ENABLE_PYTHON_HTTP OR TRITON_ENABLE_PYTHON_GRPC

if(TRITON_ENABLE_JAVA_HTTP)
  ExternalProject_Add(java-clients
    PREFIX java-clients
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/java"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/java-clients"
    CMAKE_CACHE_ARGS
      ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
      ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
      ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
      -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
      -DTRITON_VERSION:STRING=${TRITON_VERSION}
      -DTRITON_ENABLE_JAVA_HTTP:BOOL=${TRITON_ENABLE_JAVA_HTTP}
      -DTRITON_ENABLE_EXAMPLES:BOOL=${TRITON_ENABLE_EXAMPLES}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
    INSTALL_COMMAND ""
  )

endif() # TRITON_ENABLE_JAVA_HTTP
