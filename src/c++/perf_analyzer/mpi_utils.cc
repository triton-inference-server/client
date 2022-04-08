// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mpi_utils.h"

#include <dlfcn.h>
#include <iostream>
#include <stdexcept>

namespace triton { namespace perfanalyzer {

MPIDriver::MPIDriver(bool is_enabled) : is_enabled_(is_enabled)
{
  if (is_enabled_ == false) {
    return;
  }

  handle_ = dlopen("libmpi.so", RTLD_LAZY | RTLD_GLOBAL);

  if (handle_ == nullptr) {
    throw std::runtime_error(
        "Unable to load MPI library. If you are trying to run with "
        "MPI / multiple models, check that 'libmpi.so' is on "
        "`LD_LIBRARY_PATH` environment variable path.");
  }

  CheckMPIImpl();
}

bool
MPIDriver::IsMPIRun()
{
  if (is_enabled_ == false) {
    return false;
  }

  if (MPIInitialized() == false) {
    throw std::runtime_error("Must call MPI_Init() before calling IsMPIRun().");
  }

  return MPICommSizeWorld() > 1;
}

void
MPIDriver::MPIInit(int* argc, char*** argv)
{
  if (is_enabled_ == false) {
    return;
  }

  int (*MPI_Init)(
      int*, char***){(int (*)(int*, char***))dlsym(handle_, "MPI_Init")};
  if (MPI_Init == nullptr) {
    throw std::runtime_error("Unable to obtain address of `MPI_Init` symbol.");
  }

  MPI_Init(argc, argv);
}

int
MPIDriver::MPICommSizeWorld()
{
  if (is_enabled_ == false) {
    return -1;
  }

  int world_size{1};

  int (*MPI_Comm_size)(
      void*, int*){(int (*)(void*, int*))dlsym(handle_, "MPI_Comm_size")};
  if (MPI_Comm_size == nullptr) {
    throw std::runtime_error(
        "Unable to obtain address of `MPI_Comm_size` symbol.");
  }

  MPI_Comm_size(MPICommWorld(), &world_size);

  return world_size;
}

void
MPIDriver::MPIBarrierWorld()
{
  if (is_enabled_ == false) {
    return;
  }

  int (*MPI_Barrier)(void*){(int (*)(void*))dlsym(handle_, "MPI_Barrier")};
  if (MPI_Barrier == nullptr) {
    throw std::runtime_error(
        "Unable to obtain address of `MPI_Barrier` symbol.");
  }

  MPI_Barrier(MPICommWorld());
}

int
MPIDriver::MPICommRankWorld()
{
  if (is_enabled_ == false) {
    return -1;
  }

  int rank{0};

  int (*MPI_Comm_rank)(
      void*, int*){(int (*)(void*, int*))dlsym(handle_, "MPI_Comm_rank")};
  if (MPI_Comm_rank == nullptr) {
    throw std::runtime_error(
        "Unable to obtain address of `MPI_Comm_rank` symbol.");
  }

  MPI_Comm_rank(MPICommWorld(), &rank);

  return rank;
}

void
MPIDriver::MPIBcastIntWorld(void* buffer, int count, int root)
{
  if (is_enabled_ == false) {
    return;
  }

  int (*MPI_Bcast)(void*, int, void*, int, void*){
      (int (*)(void*, int, void*, int, void*))dlsym(handle_, "MPI_Bcast")};
  if (MPI_Bcast == nullptr) {
    throw std::runtime_error("Unable to obtain address of `MPI_Bcast` symbol.");
  }

  MPI_Bcast(buffer, count, MPIInt(), root, MPICommWorld());
}

void
MPIDriver::MPIFinalize()
{
  if (is_enabled_ == false) {
    return;
  }

  int (*MPI_Finalize)(){(int (*)())dlsym(handle_, "MPI_Finalize")};
  if (MPI_Finalize == nullptr) {
    throw std::runtime_error(
        "Unable to obtain address of `MPI_Finalize` symbol.");
  }

  MPI_Finalize();
}

bool
MPIDriver::MPIInitialized()
{
  if (is_enabled_ == false) {
    return false;
  }

  int (*MPI_Initialized)(int*){
      (int (*)(int*))dlsym(handle_, "MPI_Initialized")};
  if (MPI_Initialized == nullptr) {
    throw std::runtime_error(
        "Unable to obtain address of `MPI_Initialized` symbol.");
  }

  int initialized{0};
  MPI_Initialized(&initialized);
  return initialized != 0;
}

void*
MPIDriver::MPICommWorld()
{
  if (is_enabled_ == false) {
    return nullptr;
  }

  void* MPI_COMM_WORLD{dlsym(handle_, "ompi_mpi_comm_world")};
  if (MPI_COMM_WORLD == nullptr) {
    throw std::runtime_error(
        "Unable to obtain address of `ompi_mpi_comm_world` symbol.");
  }

  return MPI_COMM_WORLD;
}

void*
MPIDriver::MPIInt()
{
  if (is_enabled_ == false) {
    return nullptr;
  }

  void* MPI_INT{dlsym(handle_, "ompi_mpi_int")};
  if (MPI_INT == nullptr) {
    throw std::runtime_error(
        "Unable to obtain address of `ompi_mpi_int` symbol.");
  }

  return MPI_INT;
}

void
MPIDriver::CheckMPIImpl()
{
  if (is_enabled_ == false) {
    return;
  }

  int (*MPI_Get_library_version)(char*, int*){
      (int (*)(char*, int*))dlsym(handle_, "MPI_Get_library_version")};
  if (MPI_Get_library_version == nullptr) {
    throw std::runtime_error(
        "Unable to obtain address of `MPI_Get_library_version` symbol.");
  }

  std::string version;
  version.resize(MPIVersionStringMaximumLength);
  int resultlen{0};
  MPI_Get_library_version(&version[0], &resultlen);

  if (version.find("Open MPI") != 0) {
    throw std::runtime_error(
        "Perf Analyzer only supports Open MPI. Please uninstall your current "
        "implementation of MPI and install Open MPI.");
  }
}

}}  // namespace triton::perfanalyzer
