#!/usr/bin/env python3

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
import numpy as np
from tritonclient.utils import (
    np_to_triton_dtype,
    raise_error,
    serialize_bf16_tensor,
    serialize_byte_tensor,
    triton_to_np_dtype,
)


class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    shape : list
        The shape of the associated input.
    datatype : str
        The datatype of the associated input.
    """

    def __init__(self, name, shape, datatype):
        self._name = name
        self._shape = shape
        self._datatype = datatype
        self._parameters = {}
        self._data = None
        self._raw_data = None

    def name(self):
        """Get the name of input associated with this object.

        Returns
        -------
        str
            The name of input
        """
        return self._name

    def datatype(self):
        """Get the datatype of input associated with this object.

        Returns
        -------
        str
            The datatype of input
        """
        return self._datatype

    def shape(self):
        """Get the shape of input associated with this object.

        Returns
        -------
        list
            The shape of input
        """
        return self._shape

    def set_shape(self, shape):
        """Set the shape of input.

        Parameters
        ----------
        shape : list
            The shape of the associated input.

        Returns
        -------
        InferInput
            The updated input
        """
        self._shape = shape
        return self

    def set_data_from_numpy(self, input_tensor, binary_data=True):
        """Set the tensor data from the specified numpy array for
        input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format
        binary_data : bool
            Indicates whether to set data for the input in binary format
            or explicit tensor within JSON. The default value is True,
            which means the data will be delivered as binary data in the
            HTTP body after the JSON object.

        Returns
        -------
        InferInput
            The updated input

        Raises
        ------
        InferenceServerException
            If failed to set data for the tensor.
        """
        if not isinstance(input_tensor, (np.ndarray,)):
            raise_error("input_tensor must be a numpy array")
        # DLIS-3986: Special handling for bfloat16 until Numpy officially supports it
        if self._datatype == "BF16":
            if input_tensor.dtype != triton_to_np_dtype(self._datatype):
                raise_error(
                    "got unexpected datatype {} from numpy array, expected {} for BF16 type".format(
                        input_tensor.dtype, triton_to_np_dtype(self._datatype)
                    )
                )
        else:
            dtype = np_to_triton_dtype(input_tensor.dtype)
            if self._datatype != dtype:
                raise_error(
                    "got unexpected datatype {} from numpy array, expected {}".format(
                        dtype, self._datatype
                    )
                )
        valid_shape = True
        if len(self._shape) != len(input_tensor.shape):
            valid_shape = False
        else:
            for i in range(len(self._shape)):
                if self._shape[i] != input_tensor.shape[i]:
                    valid_shape = False
        if not valid_shape:
            raise_error(
                "got unexpected numpy array shape [{}], expected [{}]".format(
                    str(input_tensor.shape)[1:-1], str(self._shape)[1:-1]
                )
            )

        self._parameters.pop("shared_memory_region", None)
        self._parameters.pop("shared_memory_byte_size", None)
        self._parameters.pop("shared_memory_offset", None)

        if not binary_data:
            self._parameters.pop("binary_data_size", None)
            self._raw_data = None
            if self._datatype == "BF16":
                raise_error(
                    "BF16 inputs must be sent as binary data over HTTP. Please set binary_data=True"
                )
            if self._datatype == "BYTES":
                self._data = []
                try:
                    if input_tensor.size > 0:
                        for obj in np.nditer(
                            input_tensor, flags=["refs_ok"], order="C"
                        ):
                            # We need to convert the object to string using utf-8,
                            # if we want to use the binary_data=False. JSON requires
                            # the input to be a UTF-8 string.
                            if input_tensor.dtype == np.object_:
                                if type(obj.item()) == bytes:
                                    self._data.append(str(obj.item(), encoding="utf-8"))
                                else:
                                    self._data.append(str(obj.item()))
                            else:
                                self._data.append(str(obj.item(), encoding="utf-8"))
                except UnicodeDecodeError:
                    raise_error(
                        f'Failed to encode "{obj.item()}" using UTF-8. Please use binary_data=True, if'
                        " you want to pass a byte array."
                    )
            else:
                self._data = [val.item() for val in input_tensor.flatten()]
        else:
            self._data = None
            if self._datatype == "BYTES":
                serialized_output = serialize_byte_tensor(input_tensor)
                if serialized_output.size > 0:
                    self._raw_data = serialized_output.item()
                else:
                    self._raw_data = b""
            elif self._datatype == "BF16":
                serialized_output = serialize_bf16_tensor(input_tensor)
                if serialized_output.size > 0:
                    self._raw_data = serialized_output.item()
                else:
                    self._raw_data = b""
            else:
                self._raw_data = input_tensor.tobytes()
            self._parameters["binary_data_size"] = len(self._raw_data)
        return self

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Set the tensor data from the specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region holding tensor data.
        byte_size : int
            The size of the shared memory region holding tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        Returns
        -------
        InferInput
            The updated input
        """
        self._data = None
        self._raw_data = None
        self._parameters.pop("binary_data_size", None)

        self._parameters["shared_memory_region"] = region_name
        self._parameters["shared_memory_byte_size"] = byte_size
        if offset != 0:
            self._parameters["shared_memory_offset"] = offset
        return self

    def _get_binary_data(self):
        """Returns the raw binary data if available

        Returns
        -------
        bytes
            The raw data for the input tensor
        """
        return self._raw_data

    def _get_tensor(self):
        """Retrieve the underlying input as json dict.

        Returns
        -------
        dict
            The underlying tensor specification as dict
        """
        tensor = {"name": self._name, "shape": self._shape, "datatype": self._datatype}
        if self._parameters:
            tensor["parameters"] = self._parameters

        if (
            self._parameters.get("shared_memory_region") is None
            and self._raw_data is None
        ):
            if self._data is not None:
                tensor["data"] = self._data
        return tensor
