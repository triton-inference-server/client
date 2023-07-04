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
from tritonclient.utils import raise_error


class InferRequestedOutput:
    """An object of InferRequestedOutput class is used to describe a
    requested output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of output tensor to associate with this object.
    binary_data : bool
        Indicates whether to return result data for the output in
        binary format or explicit tensor within JSON. The default
        value is True, which means the data will be delivered as
        binary data in the HTTP body after JSON object. This field
        will be unset if shared memory is set for the output.
    class_count : int
        The number of classifications to be requested. The default
        value is 0 which means the classification results are not
        requested.
    """

    def __init__(self, name, binary_data=True, class_count=0):
        self._name = name
        self._parameters = {}
        if class_count != 0:
            self._parameters["classification"] = class_count
        self._binary = binary_data
        self._parameters["binary_data"] = binary_data

    def name(self):
        """Get the name of output associated with this object.

        Returns
        -------
        str
            The name of output
        """
        return self._name

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

        """
        if "classification" in self._parameters:
            raise_error("shared memory can't be set on classification output")
        if self._binary:
            self._parameters["binary_data"] = False

        self._parameters["shared_memory_region"] = region_name
        self._parameters["shared_memory_byte_size"] = byte_size
        if offset != 0:
            self._parameters["shared_memory_offset"] = offset

    def unset_shared_memory(self):
        """Clears the shared memory option set by the last call to
        InferRequestedOutput.set_shared_memory(). After call to this
        function requested output will no longer be returned in a
        shared memory region.
        """

        self._parameters["binary_data"] = self._binary
        self._parameters.pop("shared_memory_region", None)
        self._parameters.pop("shared_memory_byte_size", None)
        self._parameters.pop("shared_memory_offset", None)

    def _get_tensor(self):
        """Retrieve the underlying input as json dict.

        Returns
        -------
        dict
            The underlying tensor as a dict
        """
        tensor = {"name": self._name}
        if self._parameters:
            tensor["parameters"] = self._parameters
        return tensor
