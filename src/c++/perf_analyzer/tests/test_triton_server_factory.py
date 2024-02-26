#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from unittest.mock import MagicMock, patch

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.triton.server.server_factory import TritonServerFactory

from .common import test_result_collector as trc
from .mocks.mock_os import MockOSMethods


class TestTritonServerFactory(trc.TestResultCollector):
    def setUp(self):
        # Mock path validation
        self.mock_os = MockOSMethods(
            mock_paths=[
                "model_analyzer.triton.server.server_factory",
                "model_analyzer.config.input.config_utils",
            ]
        )
        self.mock_os.start()

    def _test_get_server_handle_helper(
        self, launch_mode, expect_local, expect_docker, expect_config
    ):
        """
        Test get_server_handle() calls the correct Triton Server create function with the
        correct Triton Server Config options

        Parameters:
        ----------
        launch_mode: str
            which launch mode to use
        expect_local: bool
            If true, assert that create_server_local() is called
        expect_docker: bool
            If true, assert that create_server_docker() is called
        expect_config: bool
            If true, verify that certain Triton Server args are passed to the create function as expected
            If false, verify that those args are NOT passed
        """

        config = ConfigCommandProfile()
        config.model_repository = "/fake_model_repository"
        config.triton_launch_mode = launch_mode
        config.triton_http_endpoint = "fake_address:2345"
        config.triton_grpc_endpoint = "fake_address:4567"
        config.monitoring_interval = 0.5

        expected_http_port = "2345"
        expected_grpc_port = "4567"
        # Convert seconds to ms
        expected_metrics_interval_ms = int(config.monitoring_interval * 1000)

        with patch(
            "model_analyzer.triton.server.server_factory.TritonServerFactory.create_server_local"
        ) as mocked_local, patch(
            "model_analyzer.triton.server.server_factory.TritonServerFactory.create_server_docker"
        ) as mocked_docker, patch(
            "model_analyzer.triton.server.server_factory.TritonServerFactory._validate_triton_install_path"
        ), patch(
            "model_analyzer.triton.server.server_factory.TritonServerFactory._validate_triton_server_path"
        ):
            _ = TritonServerFactory.get_server_handle(config, MagicMock(), False)
            self.assertEqual(mocked_local.call_count, expect_local)
            self.assertEqual(mocked_docker.call_count, expect_docker)

            if expect_local:
                _, kwargs = mocked_local.call_args
            else:
                _, kwargs = mocked_docker.call_args
            triton_config = kwargs["config"]

            if expect_config:
                self.assertEqual(triton_config["http-port"], expected_http_port)
                self.assertEqual(triton_config["grpc-port"], expected_grpc_port)
                self.assertEqual(
                    triton_config["metrics-interval-ms"], expected_metrics_interval_ms
                )

            else:
                self.assertEqual(triton_config["http-port"], None)
                self.assertEqual(triton_config["grpc-port"], None)
                self.assertEqual(triton_config["metrics-interval-ms"], None)

    def test_get_server_handle_remote(self):
        self._test_get_server_handle_helper(
            launch_mode="remote", expect_local=1, expect_docker=0, expect_config=False
        )

    def test_get_server_handle_c_api(self):
        self._test_get_server_handle_helper(
            launch_mode="c_api", expect_local=1, expect_docker=0, expect_config=False
        )

    def test_get_server_handle_local(self):
        self._test_get_server_handle_helper(
            launch_mode="local", expect_local=1, expect_docker=0, expect_config=True
        )

    def test_get_server_handle_docker(self):
        self._test_get_server_handle_helper(
            launch_mode="docker", expect_local=0, expect_docker=1, expect_config=True
        )

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
