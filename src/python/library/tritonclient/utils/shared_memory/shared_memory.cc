// Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
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
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <errno.h>
#include <fcntl.h>

#include <cstring>
#include <iostream>

#include "shared_memory.h"
#include "shared_memory_handle.h"

#define TRITON_SHM_FILE_ROOT "C:\\triton_shm\\"

//==============================================================================
// SharedMemoryControlContext
namespace {

void*
SharedMemoryHandleCreate(
    std::string triton_shm_name, void* shm_addr, std::string shm_key,
    std::unique_ptr<ShmFile>&& shm_file, size_t offset, size_t byte_size)
{
  SharedMemoryHandle* handle = new SharedMemoryHandle();
  handle->triton_shm_name_ = triton_shm_name;
  handle->base_addr_ = shm_addr;
  handle->shm_key_ = shm_key;
  handle->platform_handle_ = std::move(shm_file);
  handle->offset_ = offset;
  handle->byte_size_ = byte_size;
  return static_cast<void*>(handle);
}

int
SharedMemoryRegionMap(
    ShmFile* shm_file, size_t offset, size_t byte_size, void** shm_addr)
{
#ifdef _WIN32
  // The MapViewOfFile function takes a high-order and low-order DWORD (4 bytes
  // each) for offset. 'size_t' can either be 4 or 8 bytes depending on the
  // operating system. To handle both cases agnostically, we cast 'offset' to
  // uint64 to ensure we have a known size and enough space to perform our
  // logical operations.
  uint64_t upperbound_offset = (uint64_t)offset;
  DWORD high_order_offset = (upperbound_offset >> 32) & 0xFFFFFFFF;
  DWORD low_order_offset = upperbound_offset & 0xFFFFFFFF;
  // map shared memory to process address space
  *shm_addr = MapViewOfFile(
      shm_file->shm_mapping_handle_,  // handle to map object
      FILE_MAP_ALL_ACCESS,            // read/write permission
      high_order_offset,              // offset (high-order DWORD)
      low_order_offset,               // offset (low-order DWORD)
      byte_size);

  if (*shm_addr == NULL) {
    CloseHandle(shm_file->shm_mapping_handle_);
    return -1;
  }
  // For Windows, we cannot close the shared memory handle here. When all
  // handles are closed, the system will free the section of the paging
  // file the shared memory object uses. Instead, we close on error or when
  // we are destroying the shared memory object.
  return 0;
#else
  // map shared memory to process address space
  *shm_addr =
      mmap(NULL, byte_size, PROT_WRITE, MAP_SHARED, shm_file->shm_fd_, offset);
  if (*shm_addr == MAP_FAILED) {
    return -1;
  }

  return 0;
#endif
}

#ifdef _WIN32
int
SharedMemoryCreateBackingFile(const char* shm_key, HANDLE* backing_file_handle)
{
  LPCSTR backing_file_directory(TRITON_SHM_FILE_ROOT);
  bool success = CreateDirectory(backing_file_directory, NULL);
  if (!success && GetLastError() != ERROR_ALREADY_EXISTS) {
    return -1;
  }
  LPCSTR backing_file_path =
      std::string(TRITON_SHM_FILE_ROOT + std::string(shm_key)).c_str();
  *backing_file_handle = CreateFile(
      backing_file_path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL,
      OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (*backing_file_handle == INVALID_HANDLE_VALUE) {
    return -1;
  }
  return 0;
}

int
SharedMemoryDeleteBackingFile(const char* key, HANDLE backing_file_handle)
{
  CloseHandle(backing_file_handle);
  LPCSTR backing_file_path =
      std::string(TRITON_SHM_FILE_ROOT + std::string(key)).c_str();
  if (!DeleteFile(backing_file_path)) {
    return -1;
  }
}
#endif

}  // namespace

TRITONCLIENT_DECLSPEC int
SharedMemoryRegionCreate(
    const char* triton_shm_name, const char* shm_key, size_t byte_size,
    void** shm_handle)
{
#ifdef _WIN32
  HANDLE backing_file_handle;
  int err = SharedMemoryCreateBackingFile(shm_key, &backing_file_handle);
  if (err == -1) {
    return -7;
  }
  // The CreateFileMapping function takes a high-order and low-order DWORD (4
  // bytes each) for size. 'size_t' can either be 4 or 8 bytes depending on the
  // operating system. To handle both cases agnostically, we cast 'byte_size' to
  // uint64 to ensure we have a known size and enough space to perform our
  // logical operations.
  uint64_t upperbound_size = (uint64_t)byte_size;
  DWORD high_order_size = (upperbound_size >> 32) & 0xFFFFFFFF;
  DWORD low_order_size = upperbound_size & 0xFFFFFFFF;

  HANDLE win_handle = CreateFileMapping(
      backing_file_handle,  // use backing file
      NULL,                 // default security
      PAGE_READWRITE,       // read/write access
      high_order_size,      // maximum object size (high-order DWORD)
      low_order_size,       // maximum object size (low-order DWORD)
      shm_key);             // name of mapping object

  if (win_handle == NULL) {
    LPCSTR backing_file_path =
        std::string(TRITON_SHM_FILE_ROOT + std::string(shm_key)).c_str();
    // Cleanup backing file on failure
    SharedMemoryDeleteBackingFile(shm_key, backing_file_handle);
    return -8;
  }

  std::unique_ptr<ShmFile> shm_file =
      std::make_unique<ShmFile>(backing_file_handle, win_handle);
  // get base address of shared memory region
  void* shm_addr = nullptr;
  err = SharedMemoryRegionMap(shm_file.get(), 0, byte_size, &shm_addr);
  if (err == -1) {
    SharedMemoryDeleteBackingFile(shm_key, backing_file_handle);
    return -4;
  }
#else
  // get shared memory region descriptor
  int shm_fd = shm_open(shm_key, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return -2;
  }

  // extend shared memory object as by default it's initialized with size 0
  int res = ftruncate(shm_fd, byte_size);
  if (res == -1) {
    return -3;
  }

  std::unique_ptr<ShmFile> shm_file = std::make_unique<ShmFile>(shm_fd);
  // get base address of shared memory region
  void* shm_addr = nullptr;
  int err = SharedMemoryRegionMap(shm_file.get(), 0, byte_size, &shm_addr);
  if (err == -1) {
    return -4;
  }
#endif
  // create a handle for the shared memory region
  *shm_handle = SharedMemoryHandleCreate(
      std::string(triton_shm_name), shm_addr, std::string(shm_key),
      std::move(shm_file), 0, byte_size);
  return 0;
}

TRITONCLIENT_DECLSPEC int
SharedMemoryRegionSet(
    void* shm_handle, size_t offset, size_t byte_size, const void* data)
{
  void* shm_addr = static_cast<SharedMemoryHandle*>(shm_handle)->base_addr_;
  char* shm_addr_offset = static_cast<char*>(shm_addr);
  std::memcpy(shm_addr_offset + offset, data, byte_size);
  return 0;
}

TRITONCLIENT_DECLSPEC int
GetSharedMemoryHandleInfo(
    void* shm_handle, char** shm_addr, const char** shm_key, void* shm_file,
    size_t* offset, size_t* byte_size)
{
  SharedMemoryHandle* handle = static_cast<SharedMemoryHandle*>(shm_handle);
  ShmFile* file = static_cast<ShmFile*>(shm_file);
  *shm_addr = static_cast<char*>(handle->base_addr_);
  *shm_key = handle->shm_key_.c_str();
  *offset = handle->offset_;
  *byte_size = handle->byte_size_;
#ifdef _WIN32
  file->backing_file_handle_ = handle->platform_handle_->shm_mapping_handle_;
  file->shm_mapping_handle_ = handle->platform_handle_->shm_mapping_handle_;
#else
  file->shm_fd_ = handle->platform_handle_->shm_fd_;
#endif
  return 0;
}

TRITONCLIENT_DECLSPEC int
SharedMemoryRegionDestroy(void* shm_handle)
{
  SharedMemoryHandle* handle = static_cast<SharedMemoryHandle*>(shm_handle);
  void* shm_addr = static_cast<char*>(handle->base_addr_);

#ifdef _WIN32
  bool success = UnmapViewOfFile(shm_addr);
  if (!success) {
    return -6;
  }
  CloseHandle(handle->platform_handle_->shm_mapping_handle_);
  int err = SharedMemoryDeleteBackingFile(
      handle->shm_key_.c_str(), handle->platform_handle_->backing_file_handle_);
  if (err == -1) {
    return -9;
  }
#else
  int status = munmap(shm_addr, handle->byte_size_);
  if (status == -1) {
    return -6;
  }

  int shm_fd = shm_unlink(handle->shm_key_.c_str());
  if (shm_fd == -1) {
    return -5;
  }
  close(handle->platform_handle_->shm_fd_);
#endif  // _WIN32

  // FIXME: Investigate use of smart pointers for this
  // allocation instead
  delete handle;

  return 0;
}
//==============================================================================
