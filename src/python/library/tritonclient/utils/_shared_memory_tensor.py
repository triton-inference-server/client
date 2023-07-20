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

import collections
import ctypes
from typing import Any

from . import _dlpack


class SharedMemoryTensor:
    """An object of SharedMemoryTensor class is a view of the shared memory
    region that follows DLPack specification. This object should be considered
    invalidated if there is modification on the corresponding shared memory
    region.

    https://dmlc.github.io/dlpack/latest/python_spec.html

    """

    def __init__(
        self,
        dtype: str,
        shape: collections.abc.Iterable,
        shm_addr: Any,
        offset: int,
        byte_size: int,
        device_id: int,
    ) -> None:
        self._dtype = dtype
        self._shape = shape
        self._shm_addr = shm_addr
        self._offset = offset
        self._byte_size = byte_size
        self._device_id = device_id
        if device_id != -1:
            self._dl_device = (_dlpack.DLDeviceType.kDLCUDA, device_id)
        else:
            self._dl_device = (_dlpack.DLDeviceType.kDLCPU, 0)

    def __dlpack__(self, stream=None):
        context = _dlpack.DataViewContext(self._shape)
        size = ctypes.c_size_t(ctypes.sizeof(_dlpack.DLManagedTensor))
        dl_managed_tensor = _dlpack.DLManagedTensor.from_address(
            ctypes.pythonapi.PyMem_RawMalloc(size)
        )
        dl_managed_tensor.dl_tensor.data = self._shm_addr
        dl_managed_tensor.dl_tensor.device = self._dl_device
        dl_managed_tensor.dl_tensor.dtype = _dlpack.triton_to_dlpack_dtype(self._dtype)
        dl_managed_tensor.dl_tensor.ndim = len(self._shape)
        dl_managed_tensor.dl_tensor.shape = context._shape
        dl_managed_tensor.dl_tensor.strides = context._strides
        dl_managed_tensor.dl_tensor.byte_offset = self._offset
        dl_managed_tensor.manager_ctx = context.as_manager_ctx()
        dl_managed_tensor.deleter = _dlpack.managed_tensor_deleter
        pycapsule = ctypes.pythonapi.PyCapsule_New(
            ctypes.byref(dl_managed_tensor),
            _dlpack.c_str_dltensor,
            _dlpack.pycapsule_deleter,
        )
        return pycapsule

    def __dlpack_device__(self):
        return self._dl_device
