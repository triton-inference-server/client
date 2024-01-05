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
from tritonclient.grpc import service_pb2

from ._utils import raise_error


class InferRequestedOutput:
    """An object of :py:class:`InferRequestedOutput` class is used to describe a
    requested output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of output tensor to associate with this object
    class_count : int
        The number of classifications to be requested. The default
        value is 0 which means the classification results are not
        requested.
    """

    def __init__(self, name, class_count=0):
        self._output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        self._output.name = name
        if class_count != 0:
            self._output.parameters["classification"].int64_param = class_count

    def name(self):
        """Get the name of output associated with this object.

        Returns
        -------
        str
            The name of output
        """
        return self._output.name

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Marks the output to return the inference result in
        specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region to hold tensor data.
        byte_size : int
            The size of the shared memory region to hold tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        Raises
        ------
        InferenceServerException
            If failed to set shared memory for the tensor.
        """
        if "classification" in self._output.parameters:
            raise_error("shared memory can't be set on classification output")

        self._output.parameters["shared_memory_region"].string_param = region_name
        self._output.parameters["shared_memory_byte_size"].int64_param = byte_size
        if offset != 0:
            self._output.parameters["shared_memory_offset"].int64_param = offset

    def unset_shared_memory(self):
        """Clears the shared memory option set by the last call to
        :py:meth:`InferRequestedOutput.set_shared_memory()`. After call to this
        function requested output will no longer be returned in a
        shared memory region.
        """

        self._output.parameters.pop("shared_memory_region", None)
        self._output.parameters.pop("shared_memory_byte_size", None)
        self._output.parameters.pop("shared_memory_offset", None)

    def _get_tensor(self):
        """Retrieve the underlying InferRequestedOutputTensor message.
        Returns
        -------
        protobuf message
            The underlying InferRequestedOutputTensor protobuf message.
        """
        return self._output
