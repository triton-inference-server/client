# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(java-clients LANGUAGES C CXX)

#
# Options
#
set(TRITON_VERSION "0.0.0" CACHE STRING "Version for the clients")
option(TRITON_ENABLE_JAVA_HTTP "Enable JAVA HTTP client libraries" OFF)
option(TRITON_ENABLE_EXAMPLES "Include examples in build" OFF)
option(TRITON_ENABLE_TESTS "Include tests in build" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(TRITON_ENABLE_JAVA_HTTP)
  find_package(Java REQUIRED)
  include(UseJava)
  set(PROJECT_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target")
  set(PROJECT_JAR "java-api-0.0.1.jar")
  set(EXAMPLES_PATH "src/main/java/triton/client")
  add_custom_command(
    OUTPUT "${PROJECT_TARGET_DIR}/${PROJECT_JAR}"
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/java
    COMMAND mvn clean install -Ddir=${PROJECT_TARGET_DIR}/examples
    COMMAND cp -r ${PROJECT_TARGET_DIR}/examples ${CMAKE_INSTALL_PREFIX}/java
    COMMAND cp -r ${PROJECT_TARGET_DIR}/${PROJECT_JAR} ${CMAKE_INSTALL_PREFIX}/java
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
)

add_custom_target(
    JavaProject ALL
    DEPENDS "${PROJECT_TARGET_DIR}/${PROJECT_JAR}"
)
endif() # TRITON_ENABLE_JAVA_HTTP
