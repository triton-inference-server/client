// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "shared_library.h"
#include <dlfcn.h>
#include <iostream>

/// FIXME: Duplication of server/src/core/shared_library.cc
/// Separate shared_library to common library and delete this

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {

Error
OpenLibraryHandle(const std::string& path, void** handle)
{
  std::cout << "OpenLibraryHandle: " << path << std::endl;
  *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (*handle == nullptr) {
    return Error("unable to load backend library: " + std::string(dlerror()));
  }
  return Error::Success;
}

Error
CloseLibraryHandle(void* handle)
{
  if (handle != nullptr) {
    if (dlclose(handle) != 0) {
      return Error(
          "unable to unload backend library: " + std::string(dlerror()));
    }
  }
  return Error::Success;
}

Error
GetEntrypoint(
    void* handle, const std::string& name, const bool optional, void** befn)
{
  *befn = nullptr;
  dlerror();
  void* fn = dlsym(handle, name.c_str());
  const char* dlsym_error = dlerror();
  if (dlsym_error != nullptr) {
    if (optional) {
      return Error::Success;
    }

    std::string errstr(dlsym_error);  // need copy as dlclose overwrites
    return Error(
        "unable to find required entrypoint '" + name +
        "' in backend library: " + errstr);
  }

  if (fn == nullptr) {
    if (optional) {
      return Error::Success;
    }

    return Error(
        "unable to find required entrypoint '" + name + "' in backend library");
  }

  *befn = fn;
  return Error::Success;
}
}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
