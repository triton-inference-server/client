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
from tritonclient.grpc import service_pb2
from tritonclient.utils import *

from ._utils import raise_error


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
        self._input = service_pb2.ModelInferRequest().InferInputTensor()
        self._input.name = name
        self._input.ClearField("shape")
        self._input.shape.extend(shape)
        self._input.datatype = datatype
        self._raw_content = None

    def name(self):
        """Get the name of input associated with this object.

        Returns
        -------
        str
            The name of input
        """
        return self._input.name

    def datatype(self):
        """Get the datatype of input associated with this object.

        Returns
        -------
        str
            The datatype of input
        """
        return self._input.datatype

    def shape(self):
        """Get the shape of input associated with this object.

        Returns
        -------
        list
            The shape of input
        """
        return self._input.shape

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
        self._input.ClearField("shape")
        self._input.shape.extend(shape)
        return self

    def set_data_from_numpy(self, input_tensor):
        """Set the tensor data from the specified numpy array for
        input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format

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
        if self._input.datatype == "BF16":
            if input_tensor.dtype != triton_to_np_dtype(self._input.datatype):
                raise_error(
                    "got unexpected datatype {} from numpy array, expected {} for BF16 type".format(
                        input_tensor.dtype, triton_to_np_dtype(self._input.datatype)
                    )
                )
        else:
            dtype = np_to_triton_dtype(input_tensor.dtype)
            if self._input.datatype != dtype:
                raise_error(
                    "got unexpected datatype {} from numpy array, expected {}".format(
                        dtype, self._input.datatype
                    )
                )
        valid_shape = True
        if len(self._input.shape) != len(input_tensor.shape):
            valid_shape = False
        for i in range(len(self._input.shape)):
            if self._input.shape[i] != input_tensor.shape[i]:
                valid_shape = False
        if not valid_shape:
            raise_error(
                "got unexpected numpy array shape [{}], expected [{}]".format(
                    str(input_tensor.shape)[1:-1], str(self._input.shape)[1:-1]
                )
            )

        self._input.parameters.pop("shared_memory_region", None)
        self._input.parameters.pop("shared_memory_byte_size", None)
        self._input.parameters.pop("shared_memory_offset", None)

        if self._input.datatype == "BYTES":
            serialized_output = serialize_byte_tensor(input_tensor)
            if serialized_output.size > 0:
                self._raw_content = serialized_output.item()
            else:
                self._raw_content = b""
        elif self._input.datatype == "BF16":
            serialized_output = serialize_bf16_tensor(input_tensor)
            if serialized_output.size > 0:
                self._raw_content = serialized_output.item()
            else:
                self._raw_content = b""
        else:
            self._raw_content = input_tensor.tobytes()
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
        self._input.ClearField("contents")
        self._raw_content = None

        self._input.parameters["shared_memory_region"].string_param = region_name
        self._input.parameters["shared_memory_byte_size"].int64_param = byte_size
        if offset != 0:
            self._input.parameters["shared_memory_offset"].int64_param = offset
        return self

    def _get_tensor(self):
        """Retrieve the underlying InferInputTensor message.
        Returns
        -------
        protobuf message
            The underlying InferInputTensor protobuf message.
        """
        return self._input

    def _get_content(self):
        """Retrieve the contents for this tensor in raw bytes.
        Returns
        -------
        bytes
            The associated contents for this tensor in raw bytes.
        """
        return self._raw_content
