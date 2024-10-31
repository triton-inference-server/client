#!/usr/bin/env python3

# Copyright 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import struct
import warnings
from multiprocessing import shared_memory as mpshm

import numpy as np

_key_mapping = {}


class SharedMemoryRegion:
    def __init__(
        self,
        triton_shm_name: str,
        shm_key: str,
    ) -> None:
        self._triton_shm_name = triton_shm_name
        self._shm_key = shm_key
        self._mpsm_handle = None


def create_shared_memory_region(triton_shm_name, shm_key, byte_size, create_only=False):
    """Return a handle of the system shared memory region with the specified name and size.

    Parameters
    ----------
    triton_shm_name : str
        The unique name of the shared memory region to be created.
    shm_key : str
        The unique key of the shared memory object.
    byte_size : int
        The size in bytes of the shared memory region to be created.
    create_only : bool
        Whether a shared memory region must be created. If False and
        a shared memory region of the same name exists, a handle to that
        shared memory region will be returned and user must be aware that
        the previously allocated shared memory size can be different from
        the size requested.

    Returns
    -------
    shm_handle : SharedMemoryRegion
        The handle for the system shared memory region.

    Raises
    ------
    SharedMemoryException
        If unable to create the shared memory region.
    """
    shm_handle = SharedMemoryRegion(triton_shm_name, shm_key)
    # Check whether the region exists before creating it
    if not create_only:
        try:
            shm_handle._mpsm_handle = mpshm.SharedMemory(shm_key)
            if shm_key not in _key_mapping:
                _key_mapping[shm_key] = {
                    "needs_unlink": False,
                    "active_handle_count": 0,
                }
            _key_mapping[shm_key]["active_handle_count"] += 1
        except FileNotFoundError:
            # File not found means the shared memory region has not been created,
            # suppress the exception and attempt to create the region.
            pass
    if shm_handle._mpsm_handle is None:
        try:
            shm_handle._mpsm_handle = mpshm.SharedMemory(
                shm_key, create=True, size=byte_size
            )
        except Exception as ex:
            raise SharedMemoryException(
                "unable to create the shared memory region"
            ) from ex
        if shm_key not in _key_mapping:
            _key_mapping[shm_key] = {"needs_unlink": False, "active_handle_count": 0}
        _key_mapping[shm_key]["needs_unlink"] = True
        _key_mapping[shm_key]["active_handle_count"] += 1

    if byte_size > shm_handle._mpsm_handle.size:
        warnings.warn(
            f"reusing shared memory region with key '{shm_key}', region size is {shm_handle._mpsm_handle.size} instead of requested {byte_size}"
        )

    return shm_handle


def set_shared_memory_region(shm_handle, input_values, offset=0):
    """Copy the contents of the numpy array into the system shared memory region.

    Parameters
    ----------
    shm_handle : SharedMemoryRegion
        The handle for the system shared memory region.
    input_values : list
        The list of numpy arrays to be copied into the shared memory region.
    offset : int
        The offset, in bytes, into the region where you want the array copied.
        The default value is 0.

    Raises
    ------
    SharedMemoryException
        If unable to mmap or set values in the system shared memory region.
    """

    if not isinstance(input_values, (list, tuple)):
        raise SharedMemoryException(
            "input_values must be specified as a list/tuple of numpy arrays"
        )
    for input_value in input_values:
        if not isinstance(input_value, np.ndarray):
            raise SharedMemoryException(
                "each element of input_values must be a numpy array"
            )

    try:
        for input_value in input_values:
            # numpy array of object type is "syntactic sugar" for the API, should
            # be handled by accessing its item and treat as Python object
            if input_value.dtype == np.object_:
                byte_size = len(input_value.item())
                shm_handle._mpsm_handle.buf[offset : offset + byte_size] = (
                    input_value.item()
                )
                offset += byte_size
            else:
                shm_tensor_view = np.ndarray(
                    input_value.shape,
                    input_value.dtype,
                    buffer=shm_handle._mpsm_handle.buf[offset:],
                )
                shm_tensor_view[:] = input_value[:]
                offset += input_value.nbytes
    except Exception as ex:
        raise SharedMemoryException("unable to set the shared memory region") from ex


def get_contents_as_numpy(shm_handle, datatype, shape, offset=0):
    """Generates a numpy array using the data stored in the system shared memory
    region specified with the handle.

    Parameters
    ----------
    shm_handle : SharedMemoryRegion
        The handle for the system shared memory region.
    datatype : np.dtype
        The datatype of the array to be returned.
    shape : list
        The list of int describing the shape of the array to be returned.
    offset : int
        The offset, in bytes, into the region where you want the array extracted.
        The default value is 0.

    Returns
    -------
    np.array
        The numpy array generated using the contents of the specified shared
        memory region.
    """
    if (datatype != np.object_) and (datatype != np.bytes_):
        result = np.ndarray(
            shape, datatype, buffer=shm_handle._mpsm_handle.buf[offset:]
        )
    else:
        str_offset = offset
        val_buf = shm_handle._mpsm_handle.buf
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


def mapped_shared_memory_regions():
    """Return all system shared memory regions that were mapped but not unmapped/destroyed.

    Returns
    -------
    list
        The list of mapped system shared memory regions.
    """

    return list(_key_mapping.keys())


def destroy_shared_memory_region(shm_handle):
    """Release the handle, unlink a system shared memory region with the specified handle
    if it is the last managed handle.

    Parameters
    ----------
    shm_handle : SharedMemoryRegion
        The handle for the system shared memory region.

    Raises
    ------
    SharedMemoryException
        If unable to unlink the shared memory region.
    """
    # It is safer to remove the shared memory key from the list before
    # deleting the shared memory region because if the deletion should
    # fail, a re-attempt could result in a segfault. Secondarily, if we
    # fail to delete a region, we should not report it back to the user
    # as a valid memory region.
    shm_handle._mpsm_handle.close()
    _key_mapping[shm_handle._shm_key]["active_handle_count"] -= 1
    if _key_mapping[shm_handle._shm_key]["active_handle_count"] == 0:
        try:
            if _key_mapping[shm_handle._shm_key]["needs_unlink"]:
                shm_handle._mpsm_handle.unlink()
        finally:
            _key_mapping.pop(shm_handle._shm_key)


class SharedMemoryException(Exception):
    """Exception type for shared memory related error."""

    pass
