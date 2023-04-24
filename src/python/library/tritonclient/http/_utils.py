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
from tritonclient.utils import InferenceServerException, raise_error
from urllib.parse import quote_plus
import struct


def _get_error(response):
    """
    Returns the InferenceServerException object if response
    indicates the error. If no error then return None
    """
    if response.status_code != 200:
        body = response.read()
        try:
            error_response = json.loads(body)
            return InferenceServerException(msg=error_response["error"],
                                            status=str(response.status_code))
        except json.JSONDecodeError:
            return InferenceServerException(msg=body,
                                            status=str(response.status_code))
    else:
        return None


def _raise_if_error(response):
    """
    Raise InferenceServerException if received non-Success
    response from the server
    """
    error = _get_error(response)
    if error is not None:
        raise error


def _get_query_string(query_params):
    params = []
    for key, value in query_params.items():
        if isinstance(value, list):
            for item in value:
                params.append("%s=%s" %
                              (quote_plus(key), quote_plus(str(item))))
        else:
            params.append("%s=%s" % (quote_plus(key), quote_plus(str(value))))
    if params:
        return "&".join(params)
    return ''


def _get_inference_request(inputs, request_id, outputs, sequence_id,
                           sequence_start, sequence_end, priority, timeout,
                           custom_parameters):
    infer_request = {}
    parameters = {}
    if request_id != "":
        infer_request['id'] = request_id
    if sequence_id != 0 and sequence_id != "":
        parameters['sequence_id'] = sequence_id
        parameters['sequence_start'] = sequence_start
        parameters['sequence_end'] = sequence_end
    if priority != 0:
        parameters['priority'] = priority
    if timeout is not None:
        parameters['timeout'] = timeout

    infer_request['inputs'] = [
        this_input._get_tensor() for this_input in inputs
    ]
    if outputs:
        infer_request['outputs'] = [
            this_output._get_tensor() for this_output in outputs
        ]
    else:
        # no outputs specified so set 'binary_data_output' True in the
        # request so that all outputs are returned in binary format
        parameters['binary_data_output'] = True

    if custom_parameters:
        for key, value in custom_parameters.items():
            if key == 'sequence_id' or key == 'sequence_start' or key == 'sequence_end' or key == 'priority' or key == 'binary_data_output':
                raise_error(
                    f'Parameter "{key}" is a reserved parameter and cannot be specified.'
                )
            else:
                parameters[key] = value

    if parameters:
        infer_request['parameters'] = parameters

    request_body = json.dumps(infer_request)
    json_size = len(request_body)
    binary_data = None
    for input_tensor in inputs:
        raw_data = input_tensor._get_binary_data()
        if raw_data is not None:
            if binary_data is not None:
                binary_data += raw_data
            else:
                binary_data = raw_data

    if binary_data is not None:
        request_body = struct.pack(
            '{}s{}s'.format(len(request_body), len(binary_data)),
            request_body.encode(), binary_data)
        return request_body, json_size

    return request_body, None
