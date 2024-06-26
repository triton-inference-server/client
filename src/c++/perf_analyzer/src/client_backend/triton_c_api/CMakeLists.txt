# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(FetchContent)
FetchContent_Declare(
    repo-core
    GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
    GIT_TAG ${TRITON_CORE_REPO_TAG}
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-core)

set(
    TRITON_C_API_CLIENT_BACKEND_SRCS
    triton_c_api_backend.cc
    shared_library.cc
    triton_loader.cc
    shared_memory_manager.cc
    scoped_defer.cc
)

set(
    TRITON_C_API_CLIENT_BACKEND_HDRS
    triton_c_api_backend.h
    shared_library.h
    shared_memory_manager.h
    triton_loader.h
    c_api_infer_results.h
    scoped_defer.h
)

add_library(
    triton-c-api-backend-library  EXCLUDE_FROM_ALL OBJECT
    ${TRITON_C_API_CLIENT_BACKEND_SRCS}
    ${TRITON_C_API_CLIENT_BACKEND_HDRS}
)

target_link_libraries(
    triton-c-api-backend-library
    PUBLIC grpcclient_static
    PUBLIC gRPC::grpc++
    PUBLIC gRPC::grpc
)

target_link_libraries(
    triton-c-api-backend-library
    PUBLIC httpclient_static
    PUBLIC triton-core-serverapi   # from repo-core
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-c-api-backend-library
    PRIVATE CUDA::cudart
  )
  # Need to enable this compilation flag, otherwise, ipc.h
  # will select mock-struct as opposed to the correct cuda
  # runtime header file.
  target_compile_definitions(
    triton-c-api-backend-library
    PUBLIC TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU

