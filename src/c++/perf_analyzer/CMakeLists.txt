# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(Git REQUIRED)

execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -n 1 --abbrev-commit --format=format:%h
  RESULT_VARIABLE RETURN_CODE
  OUTPUT_VARIABLE GIT_SHA)

if(NOT RETURN_CODE EQUAL "0")
  set(GIT_SHA "unknown")
endif()

set(
  PERF_ANALYZER_SRCS
  command_line_parser.cc
  perf_analyzer.cc
  model_parser.cc
  perf_utils.cc
  load_manager.cc
  data_loader.cc
  concurrency_manager.cc
  request_rate_manager.cc
  load_worker.cc
  concurrency_worker.cc
  request_rate_worker.cc
  custom_load_manager.cc
  infer_context.cc
  inference_profiler.cc
  report_writer.cc
  mpi_utils.cc
  metrics_manager.cc
  infer_data_manager_base.cc
  infer_data_manager.cc
  infer_data_manager_shm.cc
  sequence_manager.cc
)

set(
  PERF_ANALYZER_HDRS
  command_line_parser.h
  perf_analyzer.h
  model_parser.h
  perf_utils.h
  load_manager.h
  data_loader.h
  concurrency_manager.h
  request_rate_manager.h
  custom_load_manager.h
  iworker.h
  load_worker.h
  request_rate_worker.h
  concurrency_worker.h
  infer_context.h
  inference_profiler.h
  report_writer.h
  mpi_utils.h
  doctest.h
  constants.h
  metrics.h
  metrics_manager.h
  infer_data_manager_factory.h
  iinfer_data_manager.h
  infer_data_manager.h
  infer_data_manager_shm.h
  infer_data_manager_base.h
  infer_data.h
  sequence_manager.h
  sequence_status.h
  ictx_id_tracker.h
  concurrency_ctx_id_tracker.h
  fifo_ctx_id_tracker.h
  rand_ctx_id_tracker.h
  request_record.h
)

add_executable(
  perf_analyzer
  main.cc
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

target_compile_definitions(
  perf_analyzer
  PRIVATE
    PERF_ANALYZER_VERSION=${PERF_ANALYZER_VERSION}
    GIT_SHA=${GIT_SHA}
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
list(PREPEND PERF_ANALYZER_UNIT_TESTS_SRCS perf_analyzer_unit_tests.cc)
set(PERF_ANALYZER_UNIT_TESTS_HDRS ${PERF_ANALYZER_HDRS})

add_executable(
  perf_analyzer_unit_tests
  ${PERF_ANALYZER_UNIT_TESTS_SRCS}
  ${PERF_ANALYZER_UNIT_TESTS_HDRS}
  mock_inference_profiler.h
  mock_model_parser.h
  test_utils.h
  client_backend/mock_client_backend.h
  mock_concurrency_worker.h
  mock_data_loader.h
  mock_infer_context.h
  mock_infer_data_manager.h
  mock_request_rate_worker.h
  mock_sequence_manager.h
  test_dataloader.cc
  test_inference_profiler.cc
  test_command_line_parser.cc
  test_idle_timer.cc
  test_load_manager_base.h
  test_load_manager.cc
  test_model_parser.cc
  test_metrics_manager.cc
  test_perf_utils.cc
  test_report_writer.cc
  client_backend/triton/test_triton_client_backend.cc
  test_request_rate_manager.cc
  test_concurrency_manager.cc
  test_custom_load_manager.cc
  test_sequence_manager.cc
  test_infer_context.cc
  test_ctx_id_tracker.cc
  $<TARGET_OBJECTS:json-utils-library>
)

# -Wno-write-strings is needed for the unit tests in order to statically create
# input argv cases in the CommandLineParser unit test
#
set_target_properties(perf_analyzer_unit_tests
  PROPERTIES COMPILE_FLAGS "-Wno-write-strings")

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
