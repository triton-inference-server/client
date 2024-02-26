#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import ANY, MagicMock, Mock, patch

from tritonclient.http import InferenceServerException

from .mock_base import MockBase


class MockTritonClientMethods(MockBase):
    """
    Mocks the tritonclient module functions
    used in model_analyzer/triton/client
    Provides functions to check operation.
    """

    TEST_MODEL_CONFIG = {}

    def __init__(self):
        client_attrs = {
            "load_model": MagicMock(),
            "unload_model": MagicMock(),
            "is_model_ready": MagicMock(return_value=True),
            "is_server_ready": MagicMock(return_value=True),
            "get_model_config": MagicMock(),
        }
        mock_http_client = Mock(**client_attrs)
        mock_grpc_client = Mock(**client_attrs)
        self.patcher_http_client = patch(
            "model_analyzer.triton.client.http_client.httpclient.InferenceServerClient",
            Mock(return_value=mock_http_client),
        )
        self.patcher_grpc_client = patch(
            "model_analyzer.triton.client.grpc_client.grpcclient.InferenceServerClient",
            Mock(return_value=mock_grpc_client),
        )
        super().__init__()
        self._fill_patchers()

    def start(self):
        """
        start the patchers
        """

        self.http_mock = self.patcher_http_client.start()
        self.grpc_mock = self.patcher_grpc_client.start()

    def _fill_patchers(self):
        """
        Fills the patcher list for destruction
        """

        self._patchers.append(self.patcher_http_client)
        self._patchers.append(self.patcher_grpc_client)

    def assert_created_grpc_client_with_args(
        self,
        url,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None,
    ):
        """
        Assert that the correct InferServerClient was
        indeed constructed with the specified url and SSL options
        """

        self.grpc_mock.assert_called_with(
            url=url,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )

    def assert_created_http_client_with_args(
        self, url, ssl_options={}, ssl=False, ssl_context_factory=ANY, insecure=True
    ):
        """
        Assert that the correct InferServerClient was
        indeed constructed with the specified url  and SSL options
        """

        self.http_mock.assert_called_with(
            url=url,
            ssl_options=ssl_options,
            ssl=ssl,
            ssl_context_factory=ssl_context_factory,
            insecure=insecure,
        )

    def assert_grpc_client_waited_for_server_ready(self):
        """
        Assert that the correct InferServerClient
        indeed called is_server_ready
        """

        self.grpc_mock.return_value.is_server_ready.assert_called()

    def assert_http_client_waited_for_server_ready(self):
        """
        Assert that the correct InferServerClient
        indeed called is_server_ready
        """

        self.http_mock.return_value.is_server_ready.assert_called()

    def assert_grpc_client_waited_for_model_ready(self, model_name):
        """
        Assert that the correct InferServerClient
        indeed called is_model_ready with correct model
        """

        self.grpc_mock.return_value.is_model_ready.assert_called_with(model_name)

    def assert_http_client_waited_for_model_ready(self, model_name):
        """
        Assert that the correct InferServerClient
        indeed called is_model_ready with correct model
        """

        self.http_mock.return_value.is_model_ready.assert_called_with(model_name)

    def raise_exception_on_load(self):
        """
        Set load_model to throw
        InferenceServerException
        """

        self.grpc_mock.return_value.load_model.side_effect = InferenceServerException(
            ""
        )
        self.http_mock.return_value.load_model.side_effect = InferenceServerException(
            ""
        )

    def raise_exception_on_unload(self):
        """
        Set unload_model to throw
        InferenceServerException
        """

        self.grpc_mock.return_value.unload_model.side_effect = Exception
        self.http_mock.return_value.unload_model.side_effect = Exception

    def raise_exception_on_wait_for_server_ready(self):
        """
        Set is_server_ready to throw
        InferenceServerException
        """

        self.grpc_mock.return_value.is_server_ready.side_effect = Exception
        self.http_mock.return_value.is_server_ready.side_effect = Exception

    def raise_exception_on_wait_for_model_ready(self):
        """
        Set is_model_ready to throw
        InferenceServerException
        """

        self.grpc_mock.return_value.is_model_ready.side_effect = Exception
        self.http_mock.return_value.is_model_ready.side_effect = Exception

    def set_server_not_ready(self):
        """
        Set is_server_ready's return value to False
        """

        self.grpc_mock.return_value.is_server_ready.return_value = False
        self.http_mock.return_value.is_server_ready.return_value = False

    def set_model_not_ready(self):
        """
        Set is_model_ready's return value to False
        """

        self.grpc_mock.return_value.is_model_ready.return_value = False
        self.http_mock.return_value.is_model_ready.return_value = False

    def set_model_config(self, model_config_dict):
        """
        Sets the value returned by client.get_model_config
        """

        self.grpc_mock.return_value.get_model_config.return_value = model_config_dict
        self.http_mock.return_value.get_model_config.return_value = model_config_dict

    def reset(self):
        """
        Reset the client mocks
        """

        self.grpc_mock.return_value.load_model.side_effect = None
        self.http_mock.return_value.load_model.side_effect = None
        self.grpc_mock.return_value.unload_model.side_effect = None
        self.http_mock.return_value.unload_model.side_effect = None
        self.grpc_mock.return_value.is_server_ready.side_effect = None
        self.http_mock.return_value.is_server_ready.side_effect = None
        self.grpc_mock.return_value.is_model_ready.side_effect = None
        self.http_mock.return_value.is_model_ready.side_effect = None
        self.grpc_mock.return_value.is_server_ready.return_value = True
        self.http_mock.return_value.is_server_ready.return_value = True
        self.grpc_mock.return_value.is_model_ready.return_value = True
        self.http_mock.return_value.is_model_ready.return_value = True
        self.grpc_mock.return_value.get_model_config.return_value = None
        self.http_mock.return_value.get_model_config.return_value = None
