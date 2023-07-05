# Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from ctypes import *
import numpy as np
import pkg_resources
import struct

import ctypes
from .. import _dlpack
from .._shared_memory_tensor import SharedMemoryTensor


class _utf8(object):

    @classmethod
    def from_param(cls, value):
        if value is None:
            return None
        elif isinstance(value, bytes):
            return value
        else:
            return value.encode('utf8')


_ccudashm_lib = "ccudashm" if os.name == 'nt' else 'libccudashm.so'
_ccudashm_path = pkg_resources.resource_filename(
    'tritonclient.utils.cuda_shared_memory', _ccudashm_lib)
_ccudashm = cdll.LoadLibrary(_ccudashm_path)

_ccudashm_shared_memory_region_create = _ccudashm.CudaSharedMemoryRegionCreate
_ccudashm_shared_memory_region_create.restype = c_int
_ccudashm_shared_memory_region_create.argtypes = [
    _utf8, c_uint64, c_uint64, POINTER(c_void_p)
]
_ccudashm_get_raw_handle = _ccudashm.CudaSharedMemoryGetRawHandle
_ccudashm_get_raw_handle.restype = c_int
_ccudashm_get_raw_handle.argtypes = [c_void_p, POINTER(c_char_p)]
_ccudashm_shared_memory_region_set = _ccudashm.CudaSharedMemoryRegionSet
_ccudashm_shared_memory_region_set.restype = c_int
_ccudashm_shared_memory_region_set.argtypes = [
    c_void_p, c_uint64, c_uint64, c_void_p, c_int
]
_cshm_get_shared_memory_handle_info = _ccudashm.GetCudaSharedMemoryHandleInfo
_cshm_get_shared_memory_handle_info.restype = c_int
_cshm_get_shared_memory_handle_info.argtypes = [
    c_void_p,
    POINTER(c_void_p),
    POINTER(c_uint64),
    POINTER(c_uint64),
    POINTER(c_int)
]

_cshm_cuda_shared_memory_allocate_and_read_to_host_buffer = _ccudashm.CudaSharedMemoryAllocateAndReadToHostBuffer
_cshm_cuda_shared_memory_allocate_and_read_to_host_buffer.restype = c_int
_cshm_cuda_shared_memory_allocate_and_read_to_host_buffer.argtypes = [
    c_void_p, POINTER(c_char_p)
]
_cshm_cuda_shared_memory_release_host_buffer = _ccudashm.CudaSharedMemoryReleaseHostBuffer
_cshm_cuda_shared_memory_release_host_buffer.restype = c_int
_cshm_cuda_shared_memory_release_host_buffer.argtypes = [c_char_p]

_ccudashm_shared_memory_region_destroy = _ccudashm.CudaSharedMemoryRegionDestroy
_ccudashm_shared_memory_region_destroy.restype = c_int
_ccudashm_shared_memory_region_destroy.argtypes = [c_void_p]

_ccudashm_stream_create = _ccudashm.CudaStreamCreate
_ccudashm_stream_create.restype = c_int
_ccudashm_stream_create.argtypes = [POINTER(c_void_p)]

_ccudashm_stream_destroy = _ccudashm.CudaStreamDestroy
_ccudashm_stream_destroy.restype = c_int
_ccudashm_stream_destroy.argtypes = [c_void_p]

_ccudashm_stream_synchronize = _ccudashm.CudaStreamSynchronize
_ccudashm_stream_synchronize.restype = c_int
_ccudashm_stream_synchronize.argtypes = [c_void_p]

allocated_shm_regions = []
# Internally managed stream for properly synchronizing on DLPack data,
# the stream will be created / destroyed according to 'allocated_shm_regions'
# and be reused throughout the process. May revisit for stream pool if
# asynchronous write on CUDA shared memory region is requested
_dlpack_stream = c_void_p()


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


def create_shared_memory_region(triton_shm_name, byte_size, device_id):
    """Creates a shared memory region with the specified name and size.

    Parameters
    ----------
    triton_shm_name : str
        The unique name of the cuda shared memory region to be created.
    byte_size : int
        The size in bytes of the cuda shared memory region to be created.
    device_id : int
        The GPU device ID of the cuda shared memory region to be created.
    Returns
    -------
    cuda_shm_handle : c_void_p
        The handle for the cuda shared memory region.

    Raises
    ------
    CudaSharedMemoryException
        If unable to create the cuda shared memory region on the specified device.
    """

    cuda_shm_handle = c_void_p()
    _raise_if_error(
        c_int(
            _ccudashm_shared_memory_region_create(triton_shm_name, byte_size,
                                                  device_id,
                                                  byref(cuda_shm_handle))))
    allocated_shm_regions.append(cuda_shm_handle)
    global _dlpack_stream
    if not bool(_dlpack_stream):
        _raise_if_error(c_int(_ccudashm_stream_create(byref(_dlpack_stream))))
    return cuda_shm_handle


def get_raw_handle(cuda_shm_handle):
    """Returns the underlying raw serialized cudaIPC handle in base64 encoding.

    Parameters
    ----------
    cuda_shm_handle : c_void_p
        The handle for the cuda shared memory region.

    Returns
    -------
    bytes
        The raw serialized cudaIPC handle of underlying cuda shared memory
        in base64 encoding.

    """
    craw_handle = c_char_p()
    _raise_if_error(
        c_int(_ccudashm_get_raw_handle(cuda_shm_handle, byref(craw_handle))))

    return craw_handle.value


def set_shared_memory_region(cuda_shm_handle, input_values):
    """Copy the contents of the numpy array into the cuda shared memory region.

    Parameters
    ----------
    cuda_shm_handle : c_void_p
        The handle for the cuda shared memory region.
    input_values : list
        The list of numpy arrays to be copied into the shared memory region.

    Raises
    ------
    CudaSharedMemoryException
        If unable to set values in the cuda shared memory region.
    """

    if not isinstance(input_values, (list, tuple)):
        _raise_error("input_values must be specified as a numpy array")
    for input_value in input_values:
        if not isinstance(input_value, (np.ndarray,)):
            _raise_error(
                "input_values must be specified as a list/tuple of numpy arrays"
            )

    offset_current = 0
    for input_value in input_values:
        input_value = np.ascontiguousarray(input_value).flatten()
        if input_value.dtype == np.object_:
            input_value = input_value.item()
            byte_size = np.dtype(np.byte).itemsize * len(input_value)
            _raise_if_error(
                c_int(_ccudashm_shared_memory_region_set(cuda_shm_handle, c_uint64(offset_current), \
                    c_uint64(byte_size), cast(input_value, c_void_p), -1)))
        else:
            byte_size = input_value.size * input_value.itemsize
            _raise_if_error(
                c_int(_ccudashm_shared_memory_region_set(cuda_shm_handle, c_uint64(offset_current), \
                    c_uint64(byte_size), input_value.ctypes.data_as(c_void_p), -1)))
        offset_current += byte_size
    return


def get_contents_as_numpy(cuda_shm_handle, datatype, shape):
    """Generates a numpy array using the data stored in the cuda shared memory
    region specified with the handle.

    Parameters
    ----------
    cuda_shm_handle : c_void_p
        The handle for the cuda shared memory region.
    datatype : np.dtype
        The datatype of the array to be returned.
    shape : list
        The list of int describing the shape of the array to be returned.

    Returns
    -------
    np.array
        The numpy array generated using contents from the specified shared
        memory region.
    """
    offset = c_uint64()
    byte_size = c_uint64()
    host_addr = c_char_p()
    device_id = c_int()
    try:
        shm_addr = c_void_p()
        _raise_if_error(
            c_int(
                _cshm_get_shared_memory_handle_info(cuda_shm_handle,
                                                    byref(shm_addr),
                                                    byref(offset),
                                                    byref(byte_size),
                                                    byref(device_id))))
        # Numpy can only read from host buffer.
        # [FIXME] explore Python way for CUDA memory copy
        _raise_if_error(
            c_int(
                _cshm_cuda_shared_memory_allocate_and_read_to_host_buffer(
                    cuda_shm_handle, byref(host_addr))))
        start_pos = offset.value
        if (datatype != np.object_) and (datatype != np.bytes_):
            requested_byte_size = np.prod(shape) * np.dtype(datatype).itemsize
            cval_len = start_pos + requested_byte_size
            if byte_size.value < cval_len:
                _raise_error(
                    "The size of the shared memory region is unsufficient to provide numpy array with requested size"
                )
            if cval_len == 0:
                result = np.empty(shape, dtype=datatype)
            else:
                val_buf = cast(host_addr, POINTER(c_byte * cval_len))[0]
                val = np.frombuffer(val_buf, dtype=datatype, offset=start_pos)

                # Reshape the result to the appropriate shape. This copy is only
                # needed as the temporary CPU buffer is cleared later by
                # _cshm_cuda_shared_memory_release_host_buffer
                result = np.reshape(np.copy(val), shape)
        else:
            str_offset = start_pos
            val_buf = cast(host_addr, POINTER(c_byte * byte_size.value))[0]
            ii = 0
            strs = list()
            while (ii % np.prod(shape) != 0) or (ii == 0):
                l = struct.unpack_from("<I", val_buf, str_offset)[0]
                str_offset += 4
                sb = struct.unpack_from("<{}s".format(l), val_buf,
                                        str_offset)[0]
                str_offset += l
                strs.append(sb)
                ii += 1

            val = np.array(strs, dtype=object)

            # Reshape the result to the appropriate shape.
            result = np.reshape(val, shape)

    finally:
        c_int(_cshm_cuda_shared_memory_release_host_buffer(host_addr))
        return result


def set_shared_memory_region_from_dlpack(cuda_shm_handle, input_values):
    # this function basically is an implementation of 'from_dlpack'
    offset_current = 0
    for input_value in input_values:
        # Knowing the implementation detail of how shared memory region is
        # set (cudaMemcpy). There is no need to transfer ownership of
        # 'dl_managed_tensor': the data has been copied out when dlpack
        # capsule is out of scope.
        stream = _dlpack_stream.value if bool(_dlpack_stream) else None
        dlcapsule = _dlpack.get_dlpack_capsule(input_value, stream)

        ptr = ctypes.pythonapi.PyCapsule_GetPointer(dlcapsule,
                                                    _dlpack.c_str_dltensor)
        dmt = _dlpack.DLManagedTensor.from_address(ptr)
        if not _dlpack.is_contiguous_data(
                dmt.dl_tensor.ndim, dmt.dl_tensor.shape, dmt.dl_tensor.strides):
            _raise_error(
                "DLPack tensor is not contiguous. Only contiguous DLPack tensors that are stored in C-Order are supported."
            )

        # Write to shared memory region
        byte_size = _dlpack.get_byte_size(dmt.dl_tensor.dtype,
                                          dmt.dl_tensor.ndim,
                                          dmt.dl_tensor.shape)
        # apply offset to the data pointer ('data' pointer is implicitly converted to int)
        data_ptr = dmt.dl_tensor.data + dmt.dl_tensor.byte_offset
        if dmt.dl_tensor.device == _dlpack.DLDeviceType.kDLCUDA:
            device_id = dmt.dl_tensor.device.device_id
        else:
            device_id = -1

        _raise_if_error(c_int(_ccudashm_stream_synchronize(_dlpack_stream)))
        _raise_if_error(
            c_int(_ccudashm_shared_memory_region_set(cuda_shm_handle, c_uint64(offset_current), \
                c_uint64(byte_size), cast(data_ptr, c_void_p), device_id)))

        offset_current += byte_size
    return


def as_shared_memory_tensor(cuda_shm_handle, datatype, shape):
    offset = c_uint64()
    byte_size = c_uint64()
    shm_addr = c_void_p()
    device_id = c_int()
    _raise_if_error(
        c_int(
            _cshm_get_shared_memory_handle_info(cuda_shm_handle,
                                                byref(shm_addr), byref(offset),
                                                byref(byte_size),
                                                byref(device_id))))

    return SharedMemoryTensor(datatype, shape, shm_addr, offset, byte_size,
                              device_id)


def allocated_shared_memory_regions():
    """Return all cuda shared memory regions that were allocated but not freed.

    Returns
    -------
    list
        The list of cuda shared memory handles corresponding to the allocated regions.
    """

    return allocated_shm_regions


def destroy_shared_memory_region(cuda_shm_handle):
    """Close a cuda shared memory region with the specified handle.

    Parameters
    ----------
    cuda_shm_handle : c_void_p
        The handle for the cuda shared memory region.

    Raises
    ------
    CudaSharedMemoryException
        If unable to close the cuda shared memory region and free the device memory.
    """

    _raise_if_error(
        c_int(_ccudashm_shared_memory_region_destroy(cuda_shm_handle)))
    allocated_shm_regions.remove(cuda_shm_handle)

    global _dlpack_stream
    if not allocated_shm_regions and bool(_dlpack_stream):
        _raise_if_error(c_int(_ccudashm_stream_destroy(_dlpack_stream)))
        _dlpack_stream = c_void_p()
    return


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
