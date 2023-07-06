#!/usr/bin/env python3
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

# Check for dependency before other import so other imports can assume
# the module is available (drop "try ... except .."")
try:
    from cuda import cudart
except ModuleNotFoundError as error:
    raise RuntimeError(
        'The installation does not include CUDA shared memory support. Specify \'cudashm\' or \'all\' while installing the tritonclient package to include the support'
    ) from error

import os
import numpy as np
import pkg_resources

import base64

import ctypes
from .. import _dlpack
from .._shared_memory_tensor import SharedMemoryTensor
from ._utils import CudaSharedMemoryHandle, CudaStream, _raise_errno_if_cuda_err, _raise_if_error, _raise_error


class _utf8(object):
    @classmethod
    def from_param(cls, value):
        if value is None:
            return None
        elif isinstance(value, bytes):
            return value
        else:
            return value.encode("utf8")


allocated_shm_regions = []
# Internally managed stream for properly synchronizing on DLPack data,
# the stream will be created / destroyed according to 'allocated_shm_regions'
# and be reused throughout the process. May revisit for stream pool if
# asynchronous write on CUDA shared memory region is requested
_dlpack_stream = None


# Helper function to retrieve internally managed CUDA stream
def _get_or_create_global_cuda_stream():
    global _dlpack_stream
    if _dlpack_stream is None:
        _dlpack_stream = CudaStream()
    return _dlpack_stream._stream


def _support_uva(shm_device_id, ext_device_id):
    err, support_uva = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrUnifiedAddressing, shm_device_id)
    _raise_errno_if_cuda_err(err, -6)
    if (support_uva != 0) and (ext_device_id != -1):
        err, support_uva = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrUnifiedAddressing, ext_device_id)
        _raise_errno_if_cuda_err(err, -6)
    if support_uva == 0:
        _raise_if_error(-7)


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
    cuda_shm_handle : CudaSharedMemoryHandle
        The handle for the cuda shared memory region.

    Raises
    ------
    CudaSharedMemoryException
        If unable to create the cuda shared memory region on the specified device.
    """
    err, prev_device = cudart.cudaGetDevice()
    _raise_errno_if_cuda_err(err, -1)
    _raise_errno_if_cuda_err(cudart.cudaSetDevice(device_id), -1)
    try:
        err, device_ptr = cudart.cudaMalloc(byte_size)
        _raise_errno_if_cuda_err(err, -2)
        err, cuda_shm_handle = cudart.cudaIpcGetMemHandle(device_ptr)
        _raise_errno_if_cuda_err(err, -2)
        triton_shm_handle = CudaSharedMemoryHandle(triton_shm_name,
                                                   cuda_shm_handle, device_ptr,
                                                   byte_size, device_id)
        allocated_shm_regions.append(triton_shm_handle)
    finally:
        # Don't raise error again which may overwrite the actual error
        cudart.cudaSetDevice(prev_device)

    return triton_shm_handle


def get_raw_handle(cuda_shm_handle):
    """Returns the underlying raw serialized cudaIPC handle in base64 encoding.

    Parameters
    ----------
    cuda_shm_handle : CudaSharedMemoryHandle
        The handle for the cuda shared memory region.

    Returns
    -------
    bytes
        The raw serialized cudaIPC handle of underlying cuda shared memory
        in base64 encoding.

    """
    # 'reserved' in shared memory handle is not well documented but experiment
    # showed that it is the equivalent handle used in
    # cudaIpcOpenMemHandle (C API)
    return base64.b64encode(cuda_shm_handle._cuda_shm_handle.reserved)


def set_shared_memory_region(cuda_shm_handle, input_values):
    """Copy the contents of the numpy array into the cuda shared memory region.

    Parameters
    ----------
    cuda_shm_handle : CudaSharedMemoryHandle
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

    _support_uva(cuda_shm_handle._device_id, -1)
    stream = _get_or_create_global_cuda_stream()

    offset_current = 0
    for input_value in input_values:
        input_value = np.ascontiguousarray(input_value).flatten()
        # When the input array is in type "BYTES" (np.object_ is
        # the numpy equivalent), expect the array has been serialized
        # via 'tritonclient.utils.serialize_byte_tensor'.
        if input_value.dtype == np.object_:
            input_value = input_value.item()
            byte_size = np.dtype(np.byte).itemsize * len(input_value)
            err = cudart.cudaMemcpyAsync(
                cuda_shm_handle._base_addr + offset_current, input_value,
                byte_size, cudart.cudaMemcpyKind.cudaMemcpyDefault, stream)
            _raise_errno_if_cuda_err(err, -3)
        else:
            byte_size = input_value.size * input_value.itemsize
            err = cudart.cudaMemcpyAsync(
                cuda_shm_handle._base_addr + offset_current,
                input_value.ctypes.data, byte_size,
                cudart.cudaMemcpyKind.cudaMemcpyDefault, stream)
            _raise_errno_if_cuda_err(err, -3)
        offset_current += byte_size
    _raise_errno_if_cuda_err(cudart.cudaStreamSynchronize(stream), -8)
    return


def get_contents_as_numpy(cuda_shm_handle, datatype, shape):
    """Generates a numpy array using the data stored in the cuda shared memory
    region specified with the handle.

    Parameters
    ----------
    cuda_shm_handle : CudaSharedMemoryHandle
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
    _support_uva(cuda_shm_handle._device_id, -1)
    stream = _get_or_create_global_cuda_stream()

    # Numpy can only read from host buffer.
    host_buffer = (ctypes.c_char * cuda_shm_handle._byte_size)()
    err = cudart.cudaMemcpyAsync(host_buffer, cuda_shm_handle._base_addr,
                                 cuda_shm_handle._byte_size,
                                 cudart.cudaMemcpyKind.cudaMemcpyDefault,
                                 stream)
    _raise_errno_if_cuda_err(err, -5)
    # Sync to ensure the host buffer is ready
    _raise_errno_if_cuda_err(cudart.cudaStreamSynchronize(stream), -8)
    start_pos = 0  # was 'handle->offset_'
    if (datatype != np.object_) and (datatype != np.bytes_):
        requested_byte_size = np.prod(shape) * np.dtype(datatype).itemsize
        cval_len = start_pos + requested_byte_size
        if cuda_shm_handle._byte_size < cval_len:
            _raise_error(
                "The size of the shared memory region is unsufficient to provide numpy array with requested size"
            )
        if cval_len == 0:
            result = np.empty(shape, dtype=datatype)
        else:
            val_buf = ctypes.cast(host_buffer,
                                  ctypes.POINTER(ctypes.c_byte * cval_len))[0]
            val = np.frombuffer(val_buf, dtype=datatype, offset=start_pos)

            # Reshape the result to the appropriate shape. This copy is only
            # needed as the temporary CPU buffer is cleared later by
            # _cshm_cuda_shared_memory_release_host_buffer
            result = np.reshape(np.copy(val), shape)
    else:
        str_offset = start_pos
        val_buf = ctypes.cast(
            host_buffer,
            ctypes.POINTER(ctypes.c_byte * cuda_shm_handle._byte_size))[0]
        ii = 0
        strs = list()
        while (ii % np.prod(shape) != 0) or (ii == 0):
            l = struct.unpack_from("<I", val_buf, str_offset)[0]
            str_offset += 4
            sb = struct.unpack_from("<{}s".format(l), val_buf, str_offset)[0]
            str_offset += l
            strs.append(sb)
            ii += 1

        val = np.array(strs, dtype=object)

        # Reshape the result to the appropriate shape.
        result = np.reshape(val, shape)

    return result


def set_shared_memory_region_from_dlpack(cuda_shm_handle, input_values):
    stream = _get_or_create_global_cuda_stream()
    # this function basically is an implementation of 'from_dlpack'
    offset_current = 0
    for input_value in input_values:
        # Knowing the implementation detail of how shared memory region is
        # set (cudaMemcpy). There is no need to transfer ownership of
        # 'dl_managed_tensor': the data has been copied out when dlpack
        # capsule is out of scope.
        dlcapsule = _dlpack.get_dlpack_capsule(input_value, stream.getPtr())
        dmt = _dlpack.get_managed_tensor(dlcapsule)
        if not _dlpack.is_contiguous_data(
            dmt.dl_tensor.ndim, dmt.dl_tensor.shape, dmt.dl_tensor.strides
        ):
            _raise_error(
                "DLPack tensor is not contiguous. Only contiguous DLPack tensors that are stored in C-Order are supported."
            )
        if dmt.dl_tensor.device == _dlpack.DLDeviceType.kDLCUDA:
            device_id = dmt.dl_tensor.device.device_id
        else:
            device_id = -1
        _support_uva(cuda_shm_handle._device_id, device_id)

        # Write to shared memory region
        byte_size = _dlpack.get_byte_size(
            dmt.dl_tensor.dtype, dmt.dl_tensor.ndim, dmt.dl_tensor.shape
        )
        # apply offset to the data pointer ('data' pointer is implicitly converted to int)
        data_ptr = dmt.dl_tensor.data + dmt.dl_tensor.byte_offset

        err = cudart.cudaMemcpyAsync(
            cuda_shm_handle._base_addr + offset_current, data_ptr, byte_size,
            cudart.cudaMemcpyKind.cudaMemcpyDefault, stream)
        _raise_errno_if_cuda_err(err, -3)
        _raise_errno_if_cuda_err(cudart.cudaStreamSynchronize(stream), -8)

        offset_current += byte_size
    return


def as_shared_memory_tensor(cuda_shm_handle, datatype, shape):
    return SharedMemoryTensor(datatype, shape, cuda_shm_handle._base_addr, 0,
                              cuda_shm_handle._byte_size,
                              cuda_shm_handle._device_id)


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
    cuda_shm_handle : CudaSharedMemoryHandle
        The handle for the cuda shared memory region.

    Raises
    ------
    CudaSharedMemoryException
        If unable to close the cuda shared memory region and free the device memory.
    """
    allocated_shm_regions.remove(cuda_shm_handle)
    del cuda_shm_handle
    return
