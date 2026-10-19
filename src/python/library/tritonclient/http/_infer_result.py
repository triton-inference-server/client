#!/usr/bin/env python3

# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import gzip
import io
import zlib

import numpy as np
import rapidjson as json
from tritonclient.utils import deserialize_bytes_tensor, raise_error, triton_to_np_dtype

_DEFAULT_MAX_RESPONSE_SIZE = 32 * 1024 * 1024


class InferResult:
    """An object of :py:class:`InferResult` class holds the response of
    an inference request and provide methods to retrieve
    inference results.

    Parameters
    ----------
    response : geventhttpclient.response.HTTPSocketPoolResponse
        The inference response from the server
    verbose : bool
        If True generate verbose output. Default value is False.
    max_response_size : int
        Maximum decompressed response size in bytes. Default value is 32 MiB.
    """

    def __init__(self, response, verbose, max_response_size=_DEFAULT_MAX_RESPONSE_SIZE):
        header_length = response.get("Inference-Header-Content-Length")

        # Internal class that simulate the interface of 'response'
        class DecompressedResponse:
            def __init__(self, decompressed_data):
                self.decompressed_data_ = decompressed_data
                self.offset_ = 0

            def read(self, length=-1):
                if length == -1:
                    return self.decompressed_data_[self.offset_ :]
                else:
                    prev_offset = self.offset_
                    self.offset_ += length
                    return self.decompressed_data_[prev_offset : self.offset_]

        content_encoding = response.get("Content-Encoding")
        if content_encoding in ("gzip", "deflate"):
            if type(max_response_size) is not int or max_response_size <= 0:
                raise_error("max_response_size must be a positive integer")
            # Read one byte beyond the limit to detect oversized responses.
            if content_encoding == "gzip":
                with gzip.GzipFile(
                    fileobj=io.BytesIO(response.read()), mode="rb"
                ) as stream:
                    decompressed = stream.read(max_response_size + 1)
            else:
                stream = zlib.decompressobj()
                decompressed = stream.decompress(response.read(), max_response_size + 1)
                if len(decompressed) <= max_response_size and not stream.eof:
                    raise_error("Incomplete deflate response")
            if len(decompressed) > max_response_size:
                raise_error(
                    f"Decompressed response exceeds max_response_size "
                    f"({max_response_size} bytes)"
                )
            response = DecompressedResponse(decompressed)
        if header_length is None:
            content = response.read()
            if verbose:
                print(content)
            try:
                self._result = json.loads(content)
            except UnicodeDecodeError as e:
                raise_error(
                    f"Failed to encode using UTF-8. Please use binary_data=True, if"
                    f" you want to pass a byte array. UnicodeError: {e}"
                )
        else:
            header_length = int(header_length)
            content = response.read(length=header_length)
            if verbose:
                print(content)
            self._result = json.loads(content)

            # Maps the output name to the index in buffer for quick retrieval
            self._output_name_to_buffer_map = {}
            # Read the remaining data off the response body.
            self._buffer = response.read()
            buffer_index = 0
            for output in self._result["outputs"]:
                parameters = output.get("parameters")
                if parameters is not None:
                    this_data_size = parameters.get("binary_data_size")
                    if this_data_size is not None:
                        self._output_name_to_buffer_map[output["name"]] = buffer_index
                        buffer_index = buffer_index + this_data_size

    @classmethod
    def from_response_body(
        cls,
        response_body,
        verbose=False,
        header_length=None,
        content_encoding=None,
        max_response_size=_DEFAULT_MAX_RESPONSE_SIZE,
    ):
        """A class method to construct :py:class:`InferResult` object
        from a given 'response_body'.

        Parameters
        ----------
        response_body : bytes
            The inference response from the server
        verbose : bool
            If True generate verbose output. Default value is False.
        header_length : int
            The length of the inference header if the header does not occupy
            the whole response body. Default value is None.
        content_encoding : string
            The encoding of the response body if it is compressed.
            Default value is None.
        max_response_size : int
            Maximum decompressed response size in bytes. Default value is 32 MiB.

        Returns
        -------
        InferResult
            The InferResult object generated from the response body
        """

        # Internal class that simulate the interface of 'response'
        class Response:
            def __init__(self, response_body, header_length, content_encoding):
                self.response_body_ = response_body
                self.offset_ = 0
                self.parameters_ = {
                    "Inference-Header-Content-Length": header_length,
                    "Content-Encoding": content_encoding,
                }

            def get(self, key):
                return self.parameters_.get(key)

            def read(self, length=-1):
                if length == -1:
                    return self.response_body_[self.offset_ :]
                else:
                    prev_offset = self.offset_
                    self.offset_ += length
                    return self.response_body_[prev_offset : self.offset_]

        return cls(
            Response(response_body, header_length, content_encoding),
            verbose,
            max_response_size,
        )

    def as_numpy(self, name):
        """Get the tensor data for output associated with this object
        in numpy format

        Parameters
        ----------
        name : str
            The name of the output tensor whose result is to be retrieved.

        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """
        if self._result.get("outputs") is not None:
            for output in self._result["outputs"]:
                if output["name"] == name:
                    datatype = output["datatype"]
                    has_binary_data = False
                    parameters = output.get("parameters")
                    if parameters is not None:
                        this_data_size = parameters.get("binary_data_size")
                        if this_data_size is not None:
                            has_binary_data = True
                            if this_data_size != 0:
                                start_index = self._output_name_to_buffer_map[name]
                                end_index = start_index + this_data_size
                                if datatype == "BYTES":
                                    # String results contain a 4-byte string length
                                    # followed by the actual string characters. Hence,
                                    # need to decode the raw bytes to convert into
                                    # array elements.
                                    np_array = deserialize_bytes_tensor(
                                        self._buffer[start_index:end_index]
                                    )
                                else:
                                    np_array = np.frombuffer(
                                        self._buffer[start_index:end_index],
                                        dtype=triton_to_np_dtype(datatype),
                                    )
                            else:
                                np_array = np.empty(
                                    0, dtype=triton_to_np_dtype(datatype)
                                )
                    if not has_binary_data:
                        np_array = np.array(
                            output["data"], dtype=triton_to_np_dtype(datatype)
                        )
                    np_array = np_array.reshape(output["shape"])
                    return np_array
        return None

    def get_output(self, name):
        """Retrieves the output tensor corresponding to the named output.

        Parameters
        ----------
        name : str
            The name of the tensor for which Output is to be
            retrieved.

        Returns
        -------
        Dict
            If an output tensor with specified name is present in
            the infer response then returns it as a json dict,
            otherwise returns None.
        """
        for output in self._result["outputs"]:
            if output["name"] == name:
                return output

        return None

    def get_response(self):
        """Retrieves the complete response

        Returns
        -------
        dict
            The underlying response dict.
        """
        return self._result
