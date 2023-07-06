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

from cuda import cudart
from typing import Any


# [WIP] replace in _raise_if_error
def _raise_errno_if_cuda_err(err, errno):
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            _raise_if_error(errno)


def _raise_if_error(errno):
    """
    Raise CudaSharedMemoryException if 'err' is non-success.
    Otherwise return nothing.
    """
    if errno.value != 0:
        ex = CudaSharedMemoryException(errno)
        raise ex
    return


def _raise_error(msg):
    ex = CudaSharedMemoryException(msg)
    raise ex


class CudaSharedMemoryException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : c_void_p
        Pointer to an Error that should be used to initialize the exception.

    """

    def __init__(self, err):
        self.err_code_map = {
            -1:
                "unable to set device successfully",
            -2:
                "unable to create cuda shared memory handle",
            -3:
                "unable to set values in cuda shared memory",
            -4:
                "unable to free GPU device memory",
            -5:
                "failed to read cuda shared memory results",
            -6:
                "unable to read device attributes",
            -7:
                "device or platform does not support unified virtual addressing",
            -8:
                "unable to manage CUDA stream"
        }
        self._msg = None
        if type(err) == str:
            self._msg = err
        elif err.value != 0 and err.value in self.err_code_map:
            self._msg = self.err_code_map[err.value]

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        return msg


class CudaSharedMemoryHandle:

    def __init__(self, triton_shm_name: str,
                 cuda_shm_handle: cudart.cudaIpcMemHandle_t, base_addr: Any,
                 byte_size: int, device_id: int) -> None:
        self._triton_shm_name = triton_shm_name
        self._cuda_shm_handle = cuda_shm_handle
        self._base_addr = base_addr
        self._byte_size = byte_size
        self._device_id = device_id
        # [FXIME] C implementation has below which is not relevant?
        # handle->offset_ = 0;
        # handle->shm_key_ = "";
        # handle->shm_fd_ = 0;

    def __del__(self):
        # __init__() may fail, don't attempt to release
        # uninitialized resource.
        if not hasattr(self, "_base_addr"):
            return
        err, prev_device = cudart.cudaGetDevice()
        _raise_errno_if_cuda_err(err, -1)
        _raise_errno_if_cuda_err(cudart.cudaSetDevice(self._device_id), -1)
        try:
            err = cudart.cudaFree(self._base_addr)
            _raise_errno_if_cuda_err(err, -4)
        finally:
            # Don't raise error again which may overwrite the actual error
            cudart.cudaSetDevice(prev_device)


class CudaStream:

    def __init__(self):
        err, stream = cudart.cudaStreamCreate()
        _raise_errno_if_cuda_err(err, -8)
        self._stream = stream

    def __del__(self):
        # __init__() may fail, don't attempt to release
        # uninitialized resource.
        if not hasattr(self, "_stream"):
            return
        err = cudart.cudaStreamDestroy(self._stream)
        _raise_errno_if_cuda_err(err, -8)
