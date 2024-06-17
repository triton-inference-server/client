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

import json
import logging
import threading

import numpy as np
import requests
from tritonclient.hl.lw.infer_input import InferInput

_LOGGER = logging.getLogger(__name__)


class InferOutput:
    """Output from the inference server."""

    def __init__(self, name):
        """Initialize the output."""
        self.name = name


class InferResponse:
    """Response from the inference server."""

    class Parameter:
        """Parameter for the response."""

        def __init__(self, parameter):
            """Initialize the parameter."""
            self.bool_param = parameter

    class Parameters:
        """Parameters for the response."""

        def __init__(self, parameters):
            """Initialize the parameters."""
            self.parameters = parameters

        def get(self, key):
            """Get the key."""
            return InferResponse.Parameter(self.parameters.get(key))

    def __init__(self, outputs=None, parameters=None):
        """Initialize the response."""
        if outputs is None:
            outputs = {"outputs": []}
        self._rest_outputs = outputs
        self.outputs = [
            InferOutput(response["name"]) for response in outputs["outputs"]
        ]
        self.parameters = InferResponse.Parameters(parameters)

    def get_response(self):
        """Get the response."""
        return self

    def as_numpy(self, name):
        """Get the response as numpy."""
        for response in self._rest_outputs["outputs"]:
            if response["name"] == name:
                return np.array(response["data"])
        return None


def _array_to_first_element(arr):
    """
    Convert a NumPy array to its first element if it contains only one element.
    Raise a ValueError if the array contains more than one element.

    Parameters:
    arr (np.ndarray): The input NumPy array.

    Returns:
    The first element of the array if it contains only one element.

    Raises:
    ValueError: If the array contains more than one element.
    """
    if arr.size == 1:
        return arr.item()
    else:
        raise ValueError("Array contains more than one element")


def _process_chunk(chunk):
    """Process the chunk of data received."""
    # Decode the byte string to a regular string
    chunk_str = chunk.decode("utf-8")

    # Strip the "data: " prefix
    if chunk_str.startswith("data: "):
        chunk_str = chunk_str[len("data: ") :]

    # Load the JSON string into a Python dictionary
    chunk_json = json.loads(chunk_str)
    return chunk_json


class InferenceServerClient:
    """Client to perform http communication with the Triton."""

    def __init__(self, url, **kwargs):
        self.url = url
        self._stream = None
        self._callback = None
        self._event = threading.Event()
        self._exception = None

    def is_server_ready(self):
        """Check if the server is ready."""
        response = requests.get("http://" + self.url + "/v2/health/live")
        return response.status_code == 200

    def is_server_live(self):
        """Check if the server is ready."""
        response = requests.get("http://" + self.url + "/v2/health/ready")
        return response.status_code == 200

    def is_model_ready(self, model_name, model_version):
        """Check if the model is ready."""
        model_version = model_version if model_version else "1"
        request = (
            "http://"
            + self.url
            + "/v2/models/{}/versions/{}/ready".format(model_name, model_version)
        )
        response = requests.get(request)

        return response.status_code == 200

    def get_model_config(self, model_name, model_version):
        """Get the model configuration."""
        model_version = model_version if model_version else "1"
        request = (
            "http://"
            + self.url
            + "/v2/models/{}/versions/{}/config".format(model_name, model_version)
        )
        response = requests.get(request)

        return response.json()

    def infer(
        self,
        model_name,
        model_version,
        inputs,
        headers,
        outputs,
        request_id,
        parameters,
    ):
        """Perform inference."""
        model_version = model_version if model_version else "1"
        request = (
            "http://"
            + self.url
            + "/v2/models/{}/versions/{}/infer".format(model_name, model_version)
        )

        inputs_for_json = [input_value.to_dict() for input_value in inputs]

        ## TODO: Support setting outputs, request_id, parameters and headers
        response = requests.post(request, json={"inputs": inputs_for_json})

        return InferResponse(response.json())

    def start_stream(self, callback):
        """Start the stream."""
        self._callback = callback

    def process_chunk(self, chunk, error):
        """Process the chunk of data received."""
        if self._callback:
            self._callback(chunk, error)

    def stream_request(self, model_name, model_version, inputs_for_stream):
        request_url = f"http://{self.url}/v2/models/{model_name}/generate_stream"
        headers = {"Content-Type": "application/json"}

        try:
            with requests.post(
                request_url, json=inputs_for_stream, headers=headers, stream=True
            ) as response:
                if response.status_code != 200:
                    _LOGGER.debug(f"Request failed with status code: {response}")
                    self._exception = Exception(
                        f"Request failed with status code: {response.status_code}"
                    )
                    self._event.set()
                    return

                self._event.set()  # Signal that the first response was received successfully

                try:
                    for line in response.iter_lines():
                        if line:  # Filter out keep-alive new lines
                            response_json = _process_chunk(line)
                            outputs = []
                            for key, value in response_json.items():
                                if key not in ["model_name", "model_version"]:
                                    outputs.append({"name": key, "data": [value]})
                            outputs_struct = {"outputs": outputs}
                            response = InferResponse(
                                outputs_struct,
                                parameters={"triton_final_response": False},
                            )
                            self.process_chunk(response, None)
                except Exception as e:
                    _LOGGER.debug(
                        f"Some error occurred while processing the response: {e}"
                    )
                    self.process_chunk(None, e)
            response = InferResponse(
                outputs=None, parameters={"triton_final_response": True}
            )
            self.process_chunk(response, None)
        except Exception as e:
            _LOGGER.debug(f"Some error occurred while processing the response: {e}")
            self._exception = e
            self._event.set()

    def async_stream_infer(
        self,
        model_name,
        model_version,
        inputs,
        outputs,
        request_id,
        enable_empty_final_response,
        **kwargs,
    ):
        """Perform inference."""
        model_version = model_version if model_version else "1"
        inputs_for_stream = {
            input_value.name(): _array_to_first_element(input_value._np_data)
            for input_value in inputs
        }

        self._event.clear()
        self._exception = None
        self._stream_thread = threading.Thread(
            target=self.stream_request,
            args=(model_name, model_version, inputs_for_stream),
        )
        self._stream_thread.start()
        self._event.wait()  # Block until the first 200 response or error is returned

        if isinstance(self._exception, Exception):
            raise self._exception
        else:
            raise Exception("An unknown error occurred")

    def close(self):
        """Close the stream and join the thread."""
        if self._stream is not None:
            self._stream.join()

    def close(self):
        """Close the client."""
        pass


class InferRequestedOutput:
    def __init__(self, name):
        self.name = name
