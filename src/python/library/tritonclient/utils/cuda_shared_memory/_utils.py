# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

from cuda import cuda as cuda_driver
from cuda import cudart


def call_cuda_function(function, *argv):
    res = function(*argv)
    err = res[0]
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            _, bytes = cudart.cudaGetErrorString(err)
            raise Exception(bytes)
    elif isinstance(err, cuda_driver.CUresult):
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            _, bytes = cuda_driver.cuGetErrorString(err)
            raise Exception(bytes)
    if len(res) > 1:
        return res[1:] if len(res) > 2 else res[1]
    return None


class CudaSharedMemoryException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : c_void_p
        Pointer to an Error that should be used to initialize the exception.

    """

    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        return msg


class CudaSharedMemoryRegion:
    def __init__(
        self,
        triton_shm_name: str,
        cuda_shm_handle: cudart.cudaIpcMemHandle_t,
        base_addr: Any,
        byte_size: int,
        device_id: int,
    ) -> None:
        self._triton_shm_name = triton_shm_name
        self._cuda_shm_handle = cuda_shm_handle
        self._base_addr = base_addr
        self._byte_size = byte_size
        self._device_id = device_id
        # [FXIME] C implementation has below which is not relevant, need to
        # revisit when applying similar change to system shared memory utils
        # and check on whether the "handles" can be unified.
        # handle->offset_ = 0;
        # handle->shm_key_ = "";
        # handle->shm_fd_ = 0;

    def __del__(self):
        # __init__() may fail, don't attempt to release
        # uninitialized resource.
        if not hasattr(self, "_base_addr"):
            return
        prev_device = None
        try:
            prev_device = call_cuda_function(cudart.cudaGetDevice)
            call_cuda_function(cudart.cudaSetDevice, self._device_id)
            call_cuda_function(cudart.cudaFree, self._base_addr)
        finally:
            if prev_device is not None:
                maybe_set_device(prev_device)


class CudaStream:
    def __init__(self, device_id):
        prev_device = None
        try:
            prev_device = call_cuda_function(cudart.cudaGetDevice)
            call_cuda_function(cudart.cudaSetDevice, device_id)
            self._stream = call_cuda_function(cudart.cudaStreamCreate)
        finally:
            if prev_device is not None:
                maybe_set_device(prev_device)

    def __del__(self):
        # __init__() may fail, don't attempt to release
        # uninitialized resource.
        if not hasattr(self, "_stream") or self._stream is None:
            return
        # [FIXME] __del__ is not the best place for releasing resources
        call_cuda_function(cudart.cudaStreamDestroy, self._stream)
        self._stream = None


def maybe_set_device(device_id):
    device = call_cuda_function(cuda_driver.cuDeviceGet, device_id)
    _, active = call_cuda_function(cuda_driver.cuDevicePrimaryCtxGetState, device)
    if active:
        call_cuda_function(cudart.cudaSetDevice, device_id)
