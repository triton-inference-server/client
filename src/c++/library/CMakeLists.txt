# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

find_package(Threads REQUIRED)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

#
# common
#
add_library(
    client-common-library INTERFACE
)

target_include_directories(
  client-common-library
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

#
# json_utils
#
if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER OR TRITON_ENABLE_EXAMPLES)
  find_package(RapidJSON CONFIG REQUIRED)
  add_library(
      json-utils-library EXCLUDE_FROM_ALL OBJECT
      json_utils.h json_utils.cc
  )
  target_include_directories(
    json-utils-library
    PUBLIC ${RapidJSON_INCLUDE_DIRS}
  )
  target_link_libraries(
    json-utils-library
    PRIVATE:
      client-common-library
  )
endif()

#
# shm_utils
#
add_library(
    shm-utils-library EXCLUDE_FROM_ALL OBJECT
    shm_utils.h shm_utils.cc
)
target_link_libraries(
  shm-utils-library
  PRIVATE:
    client-common-library
)

if(TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
  #
  # libgrpcclient.so and libgrpcclient_static.a
  #
  configure_file(libgrpcclient.ldscript libgrpcclient.ldscript COPYONLY)

  # libgrpcclient object build
  set(
      REQUEST_SRCS
      grpc_client.cc common.cc
  )

  set(
      REQUEST_HDRS
      grpc_client.h common.h ipc.h
  )

  add_library(
      grpc-client-library EXCLUDE_FROM_ALL OBJECT
      ${REQUEST_SRCS} ${REQUEST_HDRS}
  )
  add_dependencies(
      grpc-client-library
      grpc-service-library proto-library
  )

  # libgrpcclient_static.a
  add_library(
      grpcclient_static STATIC
      $<TARGET_OBJECTS:grpc-client-library>
      $<TARGET_OBJECTS:grpc-service-library>
      $<TARGET_OBJECTS:proto-library>
  )
  add_library(
      TritonClient::grpcclient_static ALIAS grpcclient_static
  )

  target_include_directories(
      grpcclient_static
      PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_link_libraries(
      grpcclient_static
      PRIVATE gRPC::grpc++
      PRIVATE gRPC::grpc
      PUBLIC protobuf::libprotobuf
      PUBLIC Threads::Threads
  )

  # libgrpcclient.so
  add_library(
      grpcclient SHARED
      $<TARGET_OBJECTS:grpc-service-library>
      $<TARGET_OBJECTS:proto-library>
      $<TARGET_OBJECTS:grpc-client-library>
  )
  add_library(
      TritonClient::grpcclient ALIAS grpcclient
  )

  if (NOT WIN32 AND NOT TRITON_KEEP_TYPEINFO)
     set_target_properties(
       grpcclient
       PROPERTIES
         LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libgrpcclient.ldscript
         LINK_FLAGS "-Wl,--version-script=libgrpcclient.ldscript"
     )
  endif() # NOT WIN32 AND NOT TRITON_KEEP_TYPEINFO

  target_link_libraries(
      grpcclient
      PRIVATE gRPC::grpc++
      PRIVATE gRPC::grpc
      PUBLIC protobuf::libprotobuf
      PUBLIC Threads::Threads
  )

  foreach(_client_target grpc-client-library grpcclient_static grpcclient)
    target_compile_features(${_client_target} PRIVATE cxx_std_11)
    target_compile_options(
      ${_client_target} PRIVATE
      $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Wextra -Wno-unused-parameter -Werror>
      $<$<CXX_COMPILER_ID:MSVC>:/W0 /D_WIN32_WINNT=0x0A00 /EHsc>
    )

    set_target_properties(
      ${_client_target}
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(
      ${_client_target}
      PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<TARGET_PROPERTY:proto-library,INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:grpc-service-library,INCLUDE_DIRECTORIES>
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(TRITON_ENABLE_GPU)
      target_compile_definitions(
        ${_client_target}
          PUBLIC TRITON_ENABLE_GPU=1
      )
    endif() # TRITON_ENABLE_GPU

    if(TRITON_ENABLE_GPU)
      target_link_libraries(
        ${_client_target}
        PUBLIC CUDA::cudart
      )
    endif() # TRITON_ENABLE_GPU
  endforeach()

  if (TRITON_USE_THIRD_PARTY)
    if (NOT WIN32)
      install(
        DIRECTORY
          ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/curl/lib/
          ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/grpc/lib/
          ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/protobuf/lib/
          ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/c-ares/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING
          PATTERN "*\.a"
          PATTERN "CMakeFiles" EXCLUDE
          PATTERN "cmake" EXCLUDE
          PATTERN "gens" EXCLUDE
          PATTERN "libs" EXCLUDE
          PATTERN "third_party" EXCLUDE
      )
    else()
      install(
        DIRECTORY
          ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/curl/lib/
          ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/grpc/lib/
          ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/protobuf/lib/
          ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/c-ares/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING
          PATTERN "*\.lib"
          PATTERN "CMakeFiles" EXCLUDE
          PATTERN "cmake" EXCLUDE
          PATTERN "gens" EXCLUDE
          PATTERN "libs" EXCLUDE
          PATTERN "third_party" EXCLUDE
      )
    endif() # NOT WIN32
  
    install(
      DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/grpc/include/
        ${CMAKE_CURRENT_BINARY_DIR}/../../third-party/protobuf/include/
        DESTINATION include
    )
  endif()

  install(
      FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/grpc_client.h
      DESTINATION include
  )
endif() # TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER)
  #
  # libhttpclient.so and libhttpclient_static.a
  #
  configure_file(libhttpclient.ldscript libhttpclient.ldscript COPYONLY)

  # libhttpclient object build
  set(
      REQUEST_SRCS
      http_client.cc common.cc cencode.c
  )

  set(
      REQUEST_HDRS
      http_client.h common.h ipc.h cencode.h
  )

  add_library(
      http-client-library EXCLUDE_FROM_ALL OBJECT
      ${REQUEST_SRCS} ${REQUEST_HDRS}
  )

  if (NOT WIN32)
    set_property(
      SOURCE cencode.c
      PROPERTY COMPILE_FLAGS -Wno-implicit-fallthrough
    )
  endif() # NOT WIN32

  target_link_libraries(
    http-client-library
    PUBLIC
      triton-common-json        # from repo-common
  )

  # libhttpclient_static.a
  add_library(
      httpclient_static STATIC
      $<TARGET_OBJECTS:http-client-library>
  )
  add_library(
      TritonClient::httpclient_static ALIAS httpclient_static
  )

  target_link_libraries(
      httpclient_static
      PRIVATE triton-common-json
      PRIVATE CURL::libcurl
      PUBLIC Threads::Threads
  )

  # libhttpclient.so
  add_library(
      httpclient SHARED
      $<TARGET_OBJECTS:http-client-library>
  )
  add_library(
      TritonClient::httpclient ALIAS httpclient
  )

  if (NOT WIN32)
     set_target_properties(
       httpclient
       PROPERTIES
         LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libhttpclient.ldscript
         LINK_FLAGS "-Wl,--version-script=libhttpclient.ldscript"
     )
  endif() # NOT WIN32

  target_link_libraries(
      httpclient
      PRIVATE triton-common-json
      PRIVATE CURL::libcurl
      PUBLIC Threads::Threads
  )

  foreach(_client_target http-client-library httpclient_static httpclient)
    target_compile_features(${_client_target} PRIVATE cxx_std_11)
    target_compile_options(
      ${_client_target} PRIVATE
      $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Wextra -Wno-unused-parameter -Werror>
      $<$<CXX_COMPILER_ID:MSVC>:/W0 /D_WIN32_WINNT=0x0A00 /EHsc>
    )

    set_target_properties(
      ${_client_target}
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(
      ${_client_target}
      PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:CURL::libcurl,INTERFACE_INCLUDE_DIRECTORIES>
    )

    target_compile_definitions(
      ${_client_target}
        PRIVATE CURL_STATICLIB=1
    )
    if(TRITON_ENABLE_GPU)
      target_compile_definitions(
        ${_client_target}
          PUBLIC TRITON_ENABLE_GPU=1
      )
    endif() # TRITON_ENABLE_GPU

    if(TRITON_ENABLE_GPU)
      target_link_libraries(
        ${_client_target}
        PUBLIC CUDA::cudart
      )
    endif() # TRITON_ENABLE_GPU
  endforeach()

  install(
      FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/http_client.h
      DESTINATION include
  )
endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER

if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
  install(
      FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/common.h
      ${CMAKE_CURRENT_SOURCE_DIR}/ipc.h
      DESTINATION include
  )

  include(GNUInstallDirs)
  set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonClient)

  if(TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
    install(
      TARGETS
        grpcclient
        grpcclient_static
      EXPORT
        triton-client-targets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  endif() # TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

  if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER)
    install(
      TARGETS
        httpclient
        httpclient_static
      EXPORT
        triton-client-targets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER

  install(
    EXPORT
      triton-client-targets
    FILE
      TritonClientTargets.cmake
    NAMESPACE
      TritonClient::
    DESTINATION
      ${INSTALL_CONFIGDIR}
  )

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonClientConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TritonClientConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
  )

  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/TritonClientConfig.cmake
    DESTINATION
      ${INSTALL_CONFIGDIR}
  )

  #
  # Export from build tree
  #
  export(
    EXPORT
      triton-client-targets
    FILE
      ${CMAKE_CURRENT_BINARY_DIR}/TritonClientTargets.cmake
    NAMESPACE
      TritonClient::
  )

  export(PACKAGE TritonClient)

endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER
