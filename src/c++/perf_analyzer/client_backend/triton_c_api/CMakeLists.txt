# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

set(
    TRITON_LOCAL_CLIENT_BACKEND_SRCS
    triton_c_api_backend.cc
    shared_library.cc
    triton_loader.cc
)

set(
    TRITON_LOCAL_CLIENT_BACKEND_HDRS
    triton_c_api_backend.h
    shared_library.h
    triton_loader.h
    c_api_infer_results.h
)

add_library(
    triton-c-api-backend-library  EXCLUDE_FROM_ALL OBJECT
    ${TRITON_LOCAL_CLIENT_BACKEND_SRCS}
    ${TRITON_LOCAL_CLIENT_BACKEND_HDRS}
)

target_link_libraries(
  triton-c-api-backend-library
  PUBLIC grpcclient_static
  PUBLIC httpclient_static
  PUBLIC triton-core-serverstub  # from repo-core
  PUBLIC triton-core-serverapi   # from repo-core
)

target_include_directories(
  triton-c-api-backend-library
  PUBLIC /workspace/builddir/client/_deps/repo-core-src/include
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-client-backend-library
    PRIVATE CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU
