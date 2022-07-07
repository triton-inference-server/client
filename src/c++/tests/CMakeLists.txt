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
  message("C++ tests are not currently supported on Windows because "
          "they require functionalities that are UNIX specific.")
else()

if(TRITON_ENABLE_CC_HTTP AND TRITON_ENABLE_CC_GRPC)
#
# client_timeout_test
#
add_executable(
  client_timeout_test
  client_timeout_test.cc
)

target_link_libraries(
  client_timeout_test
  PRIVATE
    grpcclient_static
    httpclient_static
)
install(
  TARGETS client_timeout_test
  RUNTIME DESTINATION bin
)

#
# memory_leak_test
#
add_executable(
  memory_leak_test
  memory_leak_test.cc
)

target_link_libraries(
  memory_leak_test
  PRIVATE
    grpcclient_static
    httpclient_static
)
install(
  TARGETS memory_leak_test
  RUNTIME DESTINATION bin
)

add_executable(
  cc_client_test
  cc_client_test.cc
)
target_include_directories(cc_client_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(
  cc_client_test
  PRIVATE
    grpcclient_static
    httpclient_static
    gtest
    ${GTEST_LIBRARY}
    ${GTEST_MAIN_LIBRARY}
)
install(
  TARGETS cc_client_test
  RUNTIME DESTINATION bin
)

endif() # TRITON_ENABLE_CC_HTTP AND TRITON_ENABLE_CC_GRPC

endif()
