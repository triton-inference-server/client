cmake_minimum_required (VERSION 3.18)

project(pef_analyzer LANGUAGES C CXX)

#
# Perf Analyzer Options
#
option(TRITON_ENABLE_PERF_ANALYZER_C_API "Enable Performance Analyzer C API" OFF)
option(TRITON_ENABLE_PERF_ANALYZER_TFS "Enable TensorFlow Serving support for Performance Analyzer" OFF)
option(TRITON_ENABLE_PERF_ANALYZER_TS "Enable TorchServe support for Performance Analyzer" OFF)
option(TRITON_ENABLE_PERF_ANALYZER_OPENAI "Enable OpenAI support for Performance Analyzer" OFF)

#
# Client Options
#
option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" ON)
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" ON)
option(TRITON_ENABLE_PYTHON_HTTP "Build Python HTTP client libraries" OFF)
option(TRITON_ENABLE_PYTHON_GRPC "Build Python GRPC client libraries" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)
option(TRITON_ENABLE_ZLIB "Include ZLIB library in build" ON)

#
# Github branch options
#
set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

# Install locations
set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party-install" CACHE STRING "Location of third-party build")

# Add third party installation to the search path
set(CMAKE_PREFIX_PATH ${TRITON_THIRD_PARTY_INSTALL_PREFIX} ${CMAKE_PREFIX_PATH})

#
# Dependencies
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
endif() # TRITON_ENABLE_GPU

find_package(gRPC CONFIG REQUIRED)
include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)

set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
include_directories(${Protobuf_INCLUDE_DIRS})

add_subdirectory(client_backend)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/9406a60c7839052e4944ea4dbc8344762a89f9bd.zip
)
FetchContent_MakeAvailable(googletest)

find_package(Git REQUIRED)

execute_process(WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND "${GIT_EXECUTABLE}" log -n 1 --abbrev-commit --format=format:%h
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
  profile_data_collector.cc
  profile_data_exporter.cc
  periodic_concurrency_manager.cc
  periodic_concurrency_worker.cc
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
  profile_data_collector.h
  profile_data_exporter.h
  periodic_concurrency_manager.h
  periodic_concurrency_worker.h
  thread_config.h
)

add_executable(
  perf_analyzer
  main.cc
  ${PERF_ANALYZER_SRCS}
  ${PERF_ANALYZER_HDRS}
)

target_link_libraries(
  perf_analyzer
  PRIVATE
    httpclient_static
    grpcclient_static
    client-backend-library
    -lb64
    ${CMAKE_DL_LIBS}
)

target_compile_features(perf_analyzer PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})

set_target_properties(
  perf_analyzer
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
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

  target_compile_definitions(
    client-backend-library
    PUBLIC TRITON_ENABLE_GPU=1
  )

  target_link_libraries(
    perf_analyzer
    PRIVATE CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

if(TRITON_ENABLE_PERF_ANALYZER_C_API)
  target_compile_definitions(
    client-backend-library
    PUBLIC TRITON_ENABLE_PERF_ANALYZER_C_API=1
  )
endif() # TRITON_ENABLE_PERF_ANALYZER_C_API

if(TRITON_ENABLE_PERF_ANALYZER_TFS)
  target_compile_definitions(
    client-backend-library
    PUBLIC TRITON_ENABLE_PERF_ANALYZER_TFS=1
  )
endif() # TRITON_ENABLE_PERF_ANALYZER_TFS

if(TRITON_ENABLE_PERF_ANALYZER_TS)
  target_compile_definitions(
    client-backend-library
    PUBLIC TRITON_ENABLE_PERF_ANALYZER_TS=1
  )
endif() # TRITON_ENABLE_PERF_ANALYZER_TS

if(TRITON_ENABLE_PERF_ANALYZER_OPENAI)
  target_compile_definitions(
    client-backend-library
    PUBLIC TRITON_ENABLE_PERF_ANALYZER_OPENAI=1
  )
endif() # TRITON_ENABLE_PERF_ANALYZER_OPENAI

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
  mock_profile_data_collector.h
  mock_profile_data_exporter.h
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
  test_profile_data_collector.cc
  test_profile_data_exporter.cc
)

# -Wno-write-strings is needed for the unit tests in order to statically create
# input argv cases in the CommandLineParser unit test
#
set_target_properties(perf_analyzer_unit_tests
  PROPERTIES COMPILE_FLAGS "-Wno-write-strings")

target_include_directories(
  perf_analyzer_unit_tests
  PRIVATE
    client_backend
)

target_link_libraries(
  perf_analyzer_unit_tests
  PRIVATE
    gmock
    client-backend-library
    -lb64
)

install(
  TARGETS perf_analyzer_unit_tests
  RUNTIME DESTINATION bin
)

