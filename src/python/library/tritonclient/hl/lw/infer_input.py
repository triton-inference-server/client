# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import numpy as np
    
def raise_error(message):
    """Raise an InferenceServerException with the specified message."""
    raise Exception(message)

def triton_to_np_dtype(dtype):
    """Converts a Triton dtype to a numpy dtype."""
    if dtype == "BOOL":
        return bool
    elif dtype == "INT8":
        return np.int8
    elif dtype == "INT16":
        return np.int16
    elif dtype == "INT32":
        return np.int32
    elif dtype == "INT64":
        return np.int64
    elif dtype == "UINT8":
        return np.uint8
    elif dtype == "UINT16":
        return np.uint16
    elif dtype == "UINT32":
        return np.uint32
    elif dtype == "UINT64":
        return np.uint64
    elif dtype == "FP16":
        return np.float16
    elif dtype == "FP32" or dtype == "BF16":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "BYTES":
        return np.object_
    return None

def serialize_byte_tensor(input_tensor):
    """
    Serializes a bytes tensor into a flat numpy array of length prepended
    bytes. The numpy array should use dtype of np.object. For np.bytes,
    numpy will remove trailing zeros at the end of byte sequence and because
    of this it should be avoided.

    Parameters
    ----------
    input_tensor : np.array
        The bytes tensor to serialize.

    Returns
    -------
    serialized_bytes_tensor : np.array
        The 1-D numpy array of type uint8 containing the serialized bytes in row-major form.

    Raises
    ------
    InferenceServerException
        If unable to serialize the given tensor.
    """

    if input_tensor.size == 0:
        return np.empty([0], dtype=np.object_)

    # If the input is a tensor of string/bytes objects, then must flatten those into
    # a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in row-major
    # order.

    if (input_tensor.dtype != np.object_) and (input_tensor.dtype.type != np.bytes_):
        raise_error("cannot serialize bytes tensor: invalid datatype")

    flattened_ls = []
    # 'C' order is row-major.
    for obj in np.nditer(input_tensor, flags=["refs_ok"], order="C"):
        # If directly passing bytes to BYTES type,
        # don't convert it to str as Python will encode the
        # bytes which may distort the meaning
        if input_tensor.dtype == np.object_:
            if type(obj.item()) == bytes:
                s = obj.item()
            else:
                s = str(obj.item()).encode("utf-8")
        else:
            s = obj.item()
        flattened_ls.append(struct.pack("<I", len(s)))
        flattened_ls.append(s)
    flattened = b"".join(flattened_ls)
    flattened_array = np.asarray(flattened, dtype=np.object_)
    if not flattened_array.flags["C_CONTIGUOUS"]:
        flattened_array = np.ascontiguousarray(flattened_array, dtype=np.object_)
    return flattened_array

def np_to_triton_dtype(np_dtype):
    """Converts a numpy dtype to a Triton dtype."""
    if np_dtype == bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None


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

        binary_data=False

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

    def to_dict(self):
        return {
            "name": self.name(),
            "shape": self.shape(),
            "datatype": self.datatype(),
            "data": self._get_tensor()["data"],
        }

class InferRequestedOutput():
    def __init__(self, name):
        self.name = name
