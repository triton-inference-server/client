# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMakeDeps, CMakeToolchain

from conan import ConanFile


class TritonClientConan(ConanFile):
    name = "triton-client"
    version = "2.68.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "enable_gpu": [True, False],
        "enable_grpc": [True, False],
        "enable_http": [True, False],
    }
    default_options = {
        "enable_gpu": False,
        "enable_grpc": True,
        "enable_http": True,
    }

    def validate(self):
        if self.settings.os != "Linux":
            raise ConanInvalidConfiguration("triton-client only supports Linux")

    def requirements(self):
        self.requires("rapidjson/cci.20230929")
        self.requires("gtest/1.14.0")
        if self.options.enable_http:
            self.requires("libcurl/8.18.0")
        if self.options.enable_grpc:
            self.requires("grpc/1.54.3")
            self.requires("protobuf/3.21.12")
            self.requires("re2/20230301")

    def configure(self):
        if self.options.enable_http:
            self.options["libcurl"].shared = False
            self.options["libcurl"].with_ssl = "openssl"
        if self.options.enable_grpc:
            self.options["grpc"].shared = False
            self.options["grpc"].cpp_plugin = True

    def layout(self):
        # Flat generators folder so CMakePresets toolchainFile path resolves
        # as build/<preset>/conan/generators/conan_toolchain.cmake.
        self.folders.generators = "generators"

    def generate(self):
        tc = CMakeToolchain(self)
        ws = self.recipe_folder + "/.."
        # Point to the monorepo sibling so common targets (triton-common-json,
        # proto-py-library) are built via add_subdirectory without a network fetch.
        tc.variables["TRITON_COMMON_SOURCE_DIR"] = ws + "/common"
        tc.variables["TRITON_SKIP_THIRD_PARTY_FETCH"] = True
        tc.variables["TRITON_USE_THIRD_PARTY"] = False
        tc.variables["TRITON_ENABLE_GPU"] = self.options.enable_gpu
        tc.variables["TRITON_ENABLE_CC_HTTP"] = self.options.enable_http
        tc.variables["TRITON_ENABLE_CC_GRPC"] = self.options.enable_grpc
        tc.variables["TRITON_ENABLE_PYTHON_HTTP"] = self.options.enable_http
        tc.variables["TRITON_ENABLE_PYTHON_GRPC"] = self.options.enable_grpc
        tc.generate()
        CMakeDeps(self).generate()
