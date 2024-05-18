# Copyright 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required (VERSION 3.18)

if(${TRITON_ENABLE_ZLIB})
  find_package(ZLIB REQUIRED)
endif()

if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif()

set(
    TRITON_CLIENT_BACKEND_SRCS
    triton_client_backend.cc
)

set(
    TRITON_CLIENT_BACKEND_HDRS
    triton_client_backend.h
)

add_library(
    triton-client-backend-library  EXCLUDE_FROM_ALL OBJECT
    ${TRITON_CLIENT_BACKEND_SRCS}
    ${TRITON_CLIENT_BACKEND_HDRS}
)

target_link_libraries(
  triton-client-backend-library
  PUBLIC grpcclient_static
  PUBLIC gRPC::grpc++
  PUBLIC gRPC::grpc
)

target_link_libraries(
  triton-client-backend-library
  PUBLIC shm_utils_static
  PUBLIC httpclient_static
  PRIVATE CURL::libcurl
)

if(${TRITON_ENABLE_ZLIB})
  target_link_libraries(
    triton-client-backend-library
    PRIVATE ZLIB::ZLIB
  )
endif()

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-client-backend-library
    PRIVATE CUDA::cudart
  )
  # Need to enable this compilation flag, otherwise, ipc.h
  # will select mock-struct as opposed to the correct cuda
  # runtime header file.
  target_compile_definitions(
    triton-client-backend-library
    PUBLIC TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU
