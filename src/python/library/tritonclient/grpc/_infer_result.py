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

import rapidjson as json
from tritonclient.utils import *
from google.protobuf.json_format import MessageToJson


class InferResult:
    """An object of InferResult class holds the response of
    an inference request and provide methods to retrieve
    inference results.

    Parameters
    ----------
    result : protobuf message
        The ModelInferResponse returned by the server
    """

    def __init__(self, result):
        self._result = result

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
        index = 0
        for output in self._result.outputs:
            if output.name == name:
                shape = []
                for value in output.shape:
                    shape.append(value)

                datatype = output.datatype
                if index < len(self._result.raw_output_contents):
                    if datatype == 'BYTES':
                        # String results contain a 4-byte string length
                        # followed by the actual string characters. Hence,
                        # need to decode the raw bytes to convert into
                        # array elements.
                        np_array = deserialize_bytes_tensor(
                            self._result.raw_output_contents[index])
                    elif datatype == "BF16":
                        np_array = deserialize_bf16_tensor(
                            self._result.raw_output_contents[index])
                    else:
                        np_array = np.frombuffer(
                            self._result.raw_output_contents[index],
                            dtype=triton_to_np_dtype(datatype))
                elif len(output.contents.bytes_contents) != 0:
                    np_array = np.array(output.contents.bytes_contents,
                                        copy=False)
                else:
                    np_array = np.empty(0)
                np_array = np_array.reshape(shape)
                return np_array
            else:
                index += 1
        return None

    def get_output(self, name, as_json=False):
        """Retrieves the InferOutputTensor corresponding to the
        named ouput.

        Parameters
        ----------
        name : str
            The name of the tensor for which Output is to be
            retrieved.
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        protobuf message or dict
            If a InferOutputTensor with specified name is present in
            ModelInferResponse then returns it as a protobuf messsage
            or dict, otherwise returns None.
        """
        for output in self._result.outputs:
            if output.name == name:
                if as_json:
                    MessageToJson(output, preserving_proto_field_name=True)
                else:
                    return output

        return None

    def get_response(self, as_json=False):
        """Retrieves the complete ModelInferResponse as a
        json dict object or protobuf message

        Parameters
        ----------
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        protobuf message or dict
            The underlying ModelInferResponse as a protobuf message or dict.
        """
        if as_json:
            return json.loads(
                MessageToJson(self._result, preserving_proto_field_name=True))
        else:
            return self._result
