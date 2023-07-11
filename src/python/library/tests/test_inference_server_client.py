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
try:
    from geventhttpclient.response import HTTPSocketResponse
except ModuleNotFoundError as error:
    raise RuntimeError(
        "The installation does not include http support. Specify 'http' or 'all' while installing the tritonclient package to include the support"
    ) from error
import unittest
from unittest.mock import MagicMock, patch

import rapidjson
from tritonclient.http import *
from tritonclient.http._client import _raise_if_error
from tritonclient.utils import *

json_error_response = """{
                          "error":"foo",
                          "status_code":"404"
                          }"""


class TestInferenceServerClient(unittest.TestCase):
    """
    Testing the various methods in InferenceServerClient to ensure proper functionality
    """

    @patch(
        "geventhttpclient.response.HTTPSocketResponse._read_headers",
        MagicMock(return_value=None),
    )
    def setUp(self) -> None:
        self.response = HTTPSocketResponse("dummy_sock")

    @patch(
        "tritonclient.http.InferenceServerClient._get",
        MagicMock(return_value={"status_code": 200}),
    )
    def test_get_method_success(self):
        """
        Verify that a 200 return code on a get call is a success and does not throw an error
        """

        url = "dummy_url"
        self.client = InferenceServerClient(url)
        self.assertEqual(self.client._get(url, None, None)["status_code"], 200)

    @patch(
        "geventhttpclient.response.HTTPSocketResponse.get_code",
        MagicMock(return_value=400),
    )
    @patch(
        "geventhttpclient.response.HTTPSocketResponse.read",
        MagicMock(return_value=json_error_response),
    )
    def test_get_method_failure(self):
        """
        Verify that non 200 return codes are raising an exception in the client
        """

        with self.assertRaises(InferenceServerException):
            _raise_if_error(self.response)

    @patch(
        "tritonclient.http.InferenceServerClient._post",
        MagicMock(return_value={"status_code": 200}),
    )
    def test_post_method_success(self):
        """
        Verify that a 200 return code on a post call is a success and does not throw an error
        """

        url = "dummy_url"
        body = "dummy_body"
        self.client = InferenceServerClient(url)
        self.assertEqual(self.client._post(url, body, None, None)["status_code"], 200)

    @patch(
        "geventhttpclient.response.HTTPSocketResponse.get_code",
        MagicMock(return_value=404),
    )
    @patch(
        "geventhttpclient.response.HTTPSocketResponse.read",
        MagicMock(return_value="error_string"),
    )
    def test_error_plain_text(self):
        """
        Verify that a non JSON response returns an InferenceServerException and not a JSONDecodeError
        """

        with self.assertRaises(InferenceServerException):
            _raise_if_error(self.response)
