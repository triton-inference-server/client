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
import struct
from urllib.parse import quote_plus

import rapidjson as json
from tritonclient.utils import InferenceServerException, raise_error


def _get_error(response):
    """
    Returns the :py:class:`InferenceServerException` object if response
    indicates the error. If no error then return None
    """
    if response.status_code != 200:
        body = None
        try:
            body = response.read().decode("utf-8")
            error_response = (
                json.loads(body)
                if len(body)
                else {"error": "client received an empty response from the server."}
            )
            return InferenceServerException(
                msg=error_response["error"], status=str(response.status_code)
            )
        except Exception as e:
            return InferenceServerException(
                msg=f"an exception occurred in the client while decoding the response: {e}",
                status=str(response.status_code),
                debug_details=body,
            )
    else:
        return None


def _raise_if_error(response):
    """
    Raise :py:class:`InferenceServerException` if received non-Success
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
                params.append("%s=%s" % (quote_plus(key), quote_plus(str(item))))
        else:
            params.append("%s=%s" % (quote_plus(key), quote_plus(str(value))))
    if params:
        return "&".join(params)
    return ""


def _get_inference_request(
    inputs,
    request_id,
    outputs,
    sequence_id,
    sequence_start,
    sequence_end,
    priority,
    timeout,
    custom_parameters,
):
    infer_request = {}
    parameters = {}
    if request_id != "":
        infer_request["id"] = request_id
    if sequence_id != 0 and sequence_id != "":
        parameters["sequence_id"] = sequence_id
        parameters["sequence_start"] = sequence_start
        parameters["sequence_end"] = sequence_end
    if priority != 0:
        parameters["priority"] = priority
    if timeout is not None:
        parameters["timeout"] = timeout

    infer_request["inputs"] = [this_input._get_tensor() for this_input in inputs]
    if outputs:
        infer_request["outputs"] = [
            this_output._get_tensor() for this_output in outputs
        ]
    else:
        # no outputs specified so set 'binary_data_output' True in the
        # request so that all outputs are returned in binary format
        parameters["binary_data_output"] = True

    if custom_parameters:
        for key, value in custom_parameters.items():
            if (
                key == "sequence_id"
                or key == "sequence_start"
                or key == "sequence_end"
                or key == "priority"
                or key == "binary_data_output"
            ):
                raise_error(
                    f'Parameter "{key}" is a reserved parameter and cannot be specified.'
                )
            else:
                parameters[key] = value

    if parameters:
        infer_request["parameters"] = parameters

    request_json = json.dumps(infer_request)
    json_size = len(request_json)

    request_body = [request_json.encode()]
    for input_tensor in inputs:
        raw_data = input_tensor._get_binary_data()
        if raw_data is not None:
            request_body.append(raw_data)

    if len(request_body) == 1:
        # The request body constitutes the whole request
        return request_body[0], None
    else:
        return b"".join(request_body), json_size
