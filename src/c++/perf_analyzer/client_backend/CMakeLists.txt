# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# fixme
add_definitions(-DCURL_STATICLIB)

add_subdirectory(triton)

if(TRITON_ENABLE_PERF_ANALYZER_C_API)
  add_subdirectory(triton_c_api)
endif()

if(TRITON_ENABLE_PERF_ANALYZER_TFS)
  add_subdirectory(tensorflow_serving)
endif()

if(TRITON_ENABLE_PERF_ANALYZER_TS)
  add_subdirectory(torchserve)
endif()

set(
  CLIENT_BACKEND_SRCS
  client_backend.cc
)

set(
  CLIENT_BACKEND_HDRS
  client_backend.h
)

if(TRITON_ENABLE_PERF_ANALYZER_C_API)
  set(CAPI_LIBRARY $<TARGET_OBJECTS:triton-c-api-backend-library>)
  set(CAPI_TARGET_LINK_LIBRARY PUBLIC $<TARGET_PROPERTY:triton-c-api-backend-library,LINK_LIBRARIES>)
  set(CAPI_TARGET_INCLUDE_DIRECTORY PRIVATE $<TARGET_PROPERTY:triton-c-api-backend-library,INCLUDE_DIRECTORIES>)
endif()

if(TRITON_ENABLE_PERF_ANALYZER_TFS)
  set(TFS_LIBRARY $<TARGET_OBJECTS:tfs-client-backend-library>)
  set(TFS_TARGET_LINK_LIBRARY PUBLIC $<TARGET_PROPERTY:tfs-client-backend-library,LINK_LIBRARIES>)
  set(TFS_TARGET_INCLUDE_DIRECTORY PRIVATE $<TARGET_PROPERTY:tfs-client-backend-library,INCLUDE_DIRECTORIES>)
endif()

if(TRITON_ENABLE_PERF_ANALYZER_TS)
  set(TS_LIBRARY $<TARGET_OBJECTS:ts-client-backend-library>)
  set(TS_TARGET_LINK_LIBRARY PUBLIC $<TARGET_PROPERTY:ts-client-backend-library,LINK_LIBRARIES>)
  set(TS_TARGET_INCLUDE_DIRECTORY PRIVATE $<TARGET_PROPERTY:ts-client-backend-library,INCLUDE_DIRECTORIES>)
endif()

add_library(
  client-backend-library
  ${CLIENT_BACKEND_SRCS}
  ${CLIENT_BACKEND_HDRS}
  $<TARGET_OBJECTS:triton-client-backend-library>
  $<TARGET_OBJECTS:shm-utils-library>
  ${CAPI_LIBRARY}
  ${TFS_LIBRARY}
  ${TS_LIBRARY}
)

target_link_libraries(
  client-backend-library
  PUBLIC triton-common-json        # from repo-common
  PUBLIC $<TARGET_PROPERTY:triton-client-backend-library,LINK_LIBRARIES>
  ${CAPI_TARGET_LINK_LIBRARY}
  ${TFS_TARGET_LINK_LIBRARY}
  ${TS_TARGET_LINK_LIBRARY}
)

target_include_directories(
  client-backend-library
  PRIVATE $<TARGET_PROPERTY:triton-client-backend-library,INCLUDE_DIRECTORIES>
  ${CAPI_TARGET_INCLUDE_DIRECTORY}
  ${TFS_TARGET_INCLUDE_DIRECTORY}
  ${TS_TARGET_INCLUDE_DIRECTORY}
)
