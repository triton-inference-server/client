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

if(WIN32)
  message("perf_analyzer is not currently supported on Windows because "
          "is requires functionalities that are UNIX specific.")
else()

add_subdirectory(client_backend)

set(
  PERF_ANALYZER_SRCS
  main.cc
  perf_analyzer.cc
  model_parser.cc
  perf_utils.cc
  load_manager.cc
  data_loader.cc
  concurrency_manager.cc
  request_rate_manager.cc
  custom_load_manager.cc
  inference_profiler.cc
  report_writer.cc
  mpi_utils.cc
)

set(
  PERF_ANALYZER_HDRS
  perf_analyzer.h
  model_parser.h
  perf_utils.h
  load_manager.h
  data_loader.h
  concurrency_manager.h
  request_rate_manager.h
  custom_load_manager.h
  inference_profiler.h
  report_writer.h
  mpi_utils.h
  doctest.h
  constants.h
)

add_executable(
  perf_analyzer
  ${PERF_ANALYZER_SRCS}
  ${PERF_ANALYZER_HDRS}
  $<TARGET_OBJECTS:json-utils-library>
)
target_link_libraries(
  perf_analyzer
  PRIVATE
    client-backend-library
    -lb64
    ${CMAKE_DL_LIBS}
)

# If gpu is enabled then compile with CUDA dependencies
if(TRITON_ENABLE_GPU)
  target_compile_definitions(
    perf_analyzer
    PUBLIC TRITON_ENABLE_GPU=1
  )

  target_link_libraries(
    perf_analyzer
    PRIVATE CUDA::cudart
  )
endif()

if(TRITON_ENABLE_PERF_ANALYZER_C_API)
  target_compile_definitions(
    client-backend-library
    PUBLIC TRITON_ENABLE_PERF_ANALYZER_C_API=1
  )
endif()

if(TRITON_ENABLE_PERF_ANALYZER_TFS)
  target_compile_definitions(
    client-backend-library
    PUBLIC TRITON_ENABLE_PERF_ANALYZER_TFS=1
  )
endif()

if(TRITON_ENABLE_PERF_ANALYZER_TS)
  target_compile_definitions(
    client-backend-library
    PUBLIC TRITON_ENABLE_PERF_ANALYZER_TS=1
  )
endif()

install(
  TARGETS perf_analyzer
  RUNTIME DESTINATION bin
)

target_compile_definitions(perf_analyzer PUBLIC DOCTEST_CONFIG_DISABLE)

# Creating perf_client link to perf_analyzer binary for backwards compatibility.
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ./perf_analyzer perf_client
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin/)")
install(CODE "message(\"-- Created symlink: perf_client -> ./perf_analyzer\")")



set(PERF_ANALYZER_UNIT_TESTS_SRCS ${PERF_ANALYZER_SRCS})
list(REMOVE_ITEM PERF_ANALYZER_UNIT_TESTS_SRCS main.cc)
list(PREPEND PERF_ANALYZER_UNIT_TESTS_SRCS perf_analyzer_unit_tests.cc)
set(PERF_ANALYZER_UNIT_TESTS_HDRS ${PERF_ANALYZER_HDRS})

add_executable(
  perf_analyzer_unit_tests
  ${PERF_ANALYZER_UNIT_TESTS_SRCS}
  ${PERF_ANALYZER_UNIT_TESTS_HDRS}
  mock_inference_profiler.h
  test_inference_profiler.cc
  test_model_parser.cc
  $<TARGET_OBJECTS:json-utils-library>
)

target_link_libraries(
  perf_analyzer_unit_tests
  PRIVATE
    gmock
    client-backend-library
    -lb64
)

target_include_directories(
  perf_analyzer_unit_tests
  PRIVATE
    client_backend
)

install(
  TARGETS perf_analyzer_unit_tests
  RUNTIME DESTINATION bin
)

endif()
