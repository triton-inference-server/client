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

import os
import tempfile
import unittest
from unittest.mock import patch

from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.triton.client.client_factory import TritonClientFactory
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory

from .common import test_result_collector as trc
from .mocks.mock_client import MockTritonClientMethods
from .mocks.mock_server_docker import MockServerDockerMethods

# Test parameters
MODEL_REPOSITORY_PATH = "test_repo"
TRITON_IMAGE = "test_image"
CONFIG_TEST_ARG = "url"
GRPC_URL = "test_grpc_url"
HTTP_URL = "test_http_url"
TEST_MODEL_NAME = "test_model"


class TestTritonClientMethods(trc.TestResultCollector):
    def setUp(self):
        # GPUs
        gpus = [GPUDevice("TEST_DEVICE_NAME", 0, "TEST_PCI_BUS_ID", "TEST_UUID")]

        # Mocks
        self.server_docker_mock = MockServerDockerMethods()
        self.tritonclient_mock = MockTritonClientMethods()
        self.server_docker_mock.start()
        self.tritonclient_mock.start()

        # Create server config
        self.server_config = TritonServerConfig()
        self.server_config["model-repository"] = MODEL_REPOSITORY_PATH
        self.server_config["model-control-mode"] = "explicit"

        # Set CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Create and start the server
        self.server = TritonServerFactory.create_server_docker(
            image=TRITON_IMAGE, config=self.server_config, gpus=gpus
        )

    def test_create_client(self):
        # Create GRPC client
        TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(url=GRPC_URL)

        # Create HTTP client
        TritonClientFactory.create_http_client(server_url=HTTP_URL)
        self.tritonclient_mock.assert_created_http_client_with_args(url=HTTP_URL)

    def test_wait_for_server_ready(self):
        # For reuse
        def _test_with_client(self, client):
            with self.assertRaises(
                TritonModelAnalyzerException,
                msg="Expected Exception trying" " wait for server ready",
            ):
                self.tritonclient_mock.raise_exception_on_wait_for_server_ready()
                client.wait_for_server_ready(num_retries=1, sleep_time=0.1)
            self.tritonclient_mock.reset()

            with self.assertRaises(
                TritonModelAnalyzerException,
                msg="Expected Exception on" " server not ready",
            ):
                self.tritonclient_mock.set_server_not_ready()
                client.wait_for_server_ready(num_retries=1, sleep_time=0.1)

            self.tritonclient_mock.reset()
            client.wait_for_server_ready(num_retries=1, sleep_time=0.1)

        # HTTP client
        client = TritonClientFactory.create_http_client(server_url=HTTP_URL)
        self.tritonclient_mock.assert_created_http_client_with_args(HTTP_URL)
        _test_with_client(self, client)
        self.tritonclient_mock.assert_http_client_waited_for_server_ready()

        # GRPC client
        client = TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(GRPC_URL)
        _test_with_client(self, client)
        self.tritonclient_mock.assert_grpc_client_waited_for_server_ready()

    def test_wait_for_server_ready_with_invalid_argument(self):
        """
        Tests that we detect when an invalid argument is passed to the server
        """
        log_file = tempfile.NamedTemporaryFile()
        log_file.write(b"Unexpected argument: UNKNOWN_CMD")

        client = TritonClientFactory.create_http_client(server_url=HTTP_URL)
        self.tritonclient_mock.raise_exception_on_wait_for_server_ready()

        with self.assertRaises(TritonModelAnalyzerException):
            client.wait_for_server_ready(
                num_retries=1, sleep_time=0.1, log_file=log_file
            )

    def test_wait_for_model_ready(self):
        # For reuse
        def _test_with_client(self, client):
            client.wait_for_model_ready(
                model_name=TEST_MODEL_NAME, num_retries=1, sleep_time=0.1
            )
            self.tritonclient_mock.reset()

            self.tritonclient_mock.set_model_not_ready()
            self.assertTrue(
                client.wait_for_model_ready(
                    model_name=TEST_MODEL_NAME, num_retries=2, sleep_time=0.1
                ),
                -1,
            )
            self.tritonclient_mock.reset()
            client.wait_for_model_ready(
                model_name=TEST_MODEL_NAME, num_retries=1, sleep_time=0.1
            )

        # HTTP client
        client = TritonClientFactory.create_http_client(server_url=HTTP_URL)
        self.tritonclient_mock.assert_created_http_client_with_args(HTTP_URL)
        _test_with_client(self, client)
        self.tritonclient_mock.assert_http_client_waited_for_model_ready(
            model_name=TEST_MODEL_NAME
        )

        # GRPC client
        client = TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(GRPC_URL)
        _test_with_client(self, client)
        self.tritonclient_mock.assert_grpc_client_waited_for_model_ready(
            model_name=TEST_MODEL_NAME
        )

    def test_load_unload_model(self):
        # Create client
        client = TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(GRPC_URL)

        # Start the server and wait till it is ready
        self.server.start()
        client.wait_for_server_ready(num_retries=1, sleep_time=0.1)

        # Try to load a dummy model and expect error
        self.tritonclient_mock.raise_exception_on_load()
        self.assertTrue(client.load_model("dummy"), -1)

        self.tritonclient_mock.reset()

        # Load the test model
        client.load_model(TEST_MODEL_NAME)
        client.wait_for_model_ready(TEST_MODEL_NAME, num_retries=1, sleep_time=0.1)

        # Try to unload a dummy model and expect error
        self.tritonclient_mock.raise_exception_on_unload()
        self.assertTrue(client.unload_model("dummy"), -1)

        self.tritonclient_mock.reset()

        # Unload the test model
        client.unload_model(TEST_MODEL_NAME)

        self.server.stop()

    def test_get_model_config(self):
        # Create client
        client = TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(GRPC_URL)

        # Start the server and wait till it is ready
        self.server.start()
        client.wait_for_server_ready(num_retries=1, sleep_time=0.1)

        # Set model config, and try to get it
        test_model_config_dict = {"config": "test_config"}
        self.tritonclient_mock.set_model_config(test_model_config_dict)
        self.assertEqual(
            client.get_model_config(TEST_MODEL_NAME, num_retries=1), "test_config"
        )

    def tearDown(self):
        # In case test raises exception
        if self.server is not None:
            self.server.stop()
        self.server_docker_mock.stop()
        self.tritonclient_mock.stop()
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
