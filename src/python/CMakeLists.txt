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

project(python-clients LANGUAGES C CXX)

#
# Options
#
set(TRITON_VERSION "0.0.0" CACHE STRING "Version for the clients")
option(TRITON_ENABLE_PYTHON_HTTP "Enable Python HTTP client libraries" OFF)
option(TRITON_ENABLE_PYTHON_GRPC "Enable Python GRPC client libraries" OFF)
option(TRITON_ENABLE_PERF_ANALYZER "Enable Performance Analyzer" OFF)
option(TRITON_ENABLE_EXAMPLES "Include examples in build" OFF)
option(TRITON_ENABLE_TESTS "Include tests in build" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)

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

if(TRITON_ENABLE_PYTHON_GRPC)
  set(TRITON_COMMON_ENABLE_PROTOBUF ON)
  set(TRITON_COMMON_ENABLE_GRPC ON)
endif() # TRITON_ENABLE_PYTHON_GRPC

FetchContent_MakeAvailable(repo-common)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# Protobuf
#
if(TRITON_ENABLE_PYTHON_GRPC)
  set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
  find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "Using protobuf ${Protobuf_VERSION}")
  include_directories(${Protobuf_INCLUDE_DIRS})
endif() # TRITON_ENABLE_PYTHON_GRPC

#
# GRPC
#
if(TRITON_ENABLE_PYTHON_GRPC)
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")
  include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)
endif() # TRITON_ENABLE_PYTHON_GRPC

if(TRITON_ENABLE_PYTHON_HTTP OR TRITON_ENABLE_PYTHON_GRPC)
  add_subdirectory(library)
  if(TRITON_ENABLE_EXAMPLES)
    add_subdirectory(examples)
  endif() # TRITON_ENABLE_EXAMPLES
endif() # TRITON_ENABLE_PYTHON_HTTP OR TRITON_ENABLE_PYTHON_GRPC
