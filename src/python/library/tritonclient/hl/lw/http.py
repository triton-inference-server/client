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

from tritonclient.hl.lw.infer_input import InferInput

class InferOutput():
    """Output from the inference server."""
    def __init__(self, name):
        """Initialize the output."""
        self.name = name


class InferResponse():
    """Response from the inference server."""

    def __init__(self, outputs):
        """Initialize the response."""
        self._rest_outputs = outputs
        self.outputs = [InferOutput(response['name'])  for response in outputs['outputs']]

    def get_response(self):
        """Get the response."""
        return self
    
    def as_numpy(self, name):
        """Get the response as numpy."""
        for response in self._rest_outputs['outputs']:
            if response['name'] == name:
                return np.array(response['data'])
        return None

class InferenceServerClient():
    """Client to perform http communication with the Triton.
    """

    def __init__(self, url, **kwargs):
        self.url = url

    def is_server_ready(self):
        """Check if the server is ready.
        """
        response = requests.get("http://" + self.url + "/v2/health/live")
        return response.status_code == 200
    
    def is_server_live(self):
        """Check if the server is ready.
        """
        response = requests.get("http://" + self.url + "/v2/health/ready")
        return response.status_code == 200
    
    def is_model_ready(self, model_name, model_version):
        """Check if the model is ready.
        """
        model_version = model_version if model_version else "1"
        request = "http://" + self.url + "/v2/models/{}/versions/{}/ready".format(model_name, model_version)
        response = requests.get(request)

        return response.status_code == 200
    
    def get_model_config(self, model_name, model_version):
        """Get the model configuration.
        """
        model_version = model_version if model_version else "1"
        request = "http://" + self.url + "/v2/models/{}/versions/{}/config".format(model_name, model_version)
        response = requests.get(request)

        return response.json()
    
        # In [13]: import requests
        #     ...: import json
        #     ...: 
        #     ...: # Define the server URL
        #     ...: server_url = "http://localhost:8000/v2/models/identity/versions/1/infer"
        #     ...: 
        #     ...: # Prepare the input data
        #     ...: input_string = "Hello Triton Inference Server!"
        #     ...: 
        #     ...: # Triton requires the data to be in a specific format
        #     ...: inputs = [
        #     ...:     {
        #     ...:         "name": "input",
        #     ...:         "shape": [1, 1],  # Adjust the shape to include the batch dimension
        #     ...:         "datatype": "BYTES",
        #     ...:         "data": [input_string]
        #     ...:     }
        #     ...: ]
        #     ...: 
        #     ...: # Prepare the request payload
        #     ...: payload = {
        #     ...:     "inputs": inputs
        #     ...: }
        #     ...: 
        #     ...: # Send the request
        #     ...: response = requests.post(server_url, json=payload)
        #     ...: 
        #     ...: # Check the response status
        #     ...: if response.status_code == 200:
        #     ...:     result = response.json()
        #     ...:     print("Inference result:", result)
        #     ...: else:
        #     ...:     print("Failed to get inference result:", response.status_code, response.text)
        #     ...: 
        # Inference result: {'model_name': 'identity', 'model_version': '1', 'outputs': [{'name': 'output', 'datatype': 'BYTES', 'shape': [1, 1], 'data': ['Hello Triton Inference Server!']}]}
        # In [14]:   

    def infer(self, model_name, model_version, inputs, headers, outputs, request_id, parameters):
        """Perform inference.
        """
        model_version = model_version if model_version else "1"
        request = "http://" + self.url + "/v2/models/{}/versions/{}/infer".format(model_name, model_version)
        print(request)

        inputs_for_json = [input_value.to_dict() for input_value in inputs]

        print(inputs_for_json)
                
        ## TODO: Support setting outputs, request_id, parameters and headers
        response = requests.post(request, json={"inputs": inputs_for_json})

        return InferResponse(response.json())
    

class InferRequestedOutput():
    def __init__(self, name):
        self.name = name
