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

import unittest
from unittest.mock import patch

from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory

from .common import test_result_collector as trc
from .mocks.mock_gpu_device_factory import MockGPUDeviceFactory
from .mocks.mock_os import MockOSMethods
from .mocks.mock_server_docker import MockServerDockerMethods
from .mocks.mock_server_local import MockServerLocalMethods

# Test parameters
MODEL_REPOSITORY_PATH = "test_repo"
TRITON_LOCAL_BIN_PATH = "test_bin_path/tritonserver"
TRITON_DOCKER_BIN_PATH = "tritonserver"
TRITON_IMAGE = "test_image"
CONFIG_TEST_ARG = "exit-on-error"
CLI_TO_STRING_TEST_ARGS = {
    "allow-grpc": True,
    "min-supported-compute-capability": 7.5,
    "metrics-port": 8000,
    "model-repository": MODEL_REPOSITORY_PATH,
}


class TestTritonServerMethods(trc.TestResultCollector):
    def setUp(self):
        # GPUs for this test
        self._sys_gpus = [
            GPUDevice("TEST_DEVICE_NAME", 0, "TEST_BUS_ID0", "TEST_UUID0"),
            GPUDevice("TEST_DEVICE_NAME", 1, "TEST_BUS_ID1", "TEST_UUID1"),
        ]

        # Mock
        self.gpu_device_factory_mock = MockGPUDeviceFactory(self._sys_gpus)
        self.server_docker_mock = MockServerDockerMethods()
        self.server_local_mock = MockServerLocalMethods()
        self.os_mock = MockOSMethods(
            mock_paths=[
                "model_analyzer.triton.server.server_local",
                "tests.mocks.mock_server_local",
            ]
        )
        self.gpu_device_factory_mock.start()
        self.server_docker_mock.start()
        self.server_local_mock.start()
        self.os_mock.start()

        # server setup
        self.server = None

    def test_server_config(self):
        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Check config initializations
        self.assertIsNone(
            server_config[CONFIG_TEST_ARG],
            msg="Server config had unexpected initial" f"value for {CONFIG_TEST_ARG}",
        )
        # Set value
        server_config[CONFIG_TEST_ARG] = True

        # Test get again
        self.assertTrue(
            server_config[CONFIG_TEST_ARG], msg=f"{CONFIG_TEST_ARG} was not set"
        )

        # Try to set an unsupported config argument, expect failure
        with self.assertRaises(
            TritonModelAnalyzerException,
            msg="Expected exception on trying to set"
            "unsupported argument in Triton server"
            "config",
        ):
            server_config["dummy"] = 1

        # Reset test arg
        server_config[CONFIG_TEST_ARG] = None

        # Finally set a couple of args and then check the cli string
        for arg, value in CLI_TO_STRING_TEST_ARGS.items():
            server_config[arg] = value

        cli_string = server_config.to_cli_string()
        for argstring in cli_string.split():
            # Parse the created string
            arg, value = argstring.split("=")
            arg = arg[2:]

            # Make sure each parsed arg was in test dict
            self.assertIn(
                arg,
                CLI_TO_STRING_TEST_ARGS,
                msg=f"CLI string contained unknown argument: {arg}",
            )

            # Make sure parsed value is the one from dict, check type too
            test_value = CLI_TO_STRING_TEST_ARGS[arg]
            self.assertEqual(
                test_value,
                type(test_value)(value),
                msg=f"CLI string contained unknown value: {value}",
            )

    def _test_create_server(self, gpus):
        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Run for both types of environments
        self.server = TritonServerFactory.create_server_docker(
            image=TRITON_IMAGE, config=server_config, gpus=gpus
        )

        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=gpus
        )

        # Try to create a server without specifying model repository and expect
        # error
        server_config["model-repository"] = None
        with self.assertRaises(
            AssertionError,
            msg="Expected AssertionError for trying to create"
            "server without specifying model repository.",
        ):
            self.server = TritonServerFactory.create_server_docker(
                image=TRITON_IMAGE, config=server_config, gpus=gpus
            )
        with self.assertRaises(
            AssertionError,
            msg="Expected AssertionError for trying to create"
            "server without specifying model repository.",
        ):
            self.server = TritonServerFactory.create_server_local(
                path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=gpus
            )

    def test_triton_server_ssl_options(self):
        server_config = TritonServerConfig()

        triton_server_flags = {
            "grpc-use-ssl": "1",
            "grpc-use-ssl-mutual": "1",
            "grpc-server-cert": "a",
            "grpc-server-key": "b",
            "grpc-root-cert": "c",
        }
        server_config.update_config(triton_server_flags)

        expected_cli_str = (
            f"--grpc-use-ssl=1 --grpc-use-ssl-mutual=1 "
            f"--grpc-server-cert=a --grpc-server-key=b --grpc-root-cert=c"
        )
        self.assertEqual(server_config.to_cli_string(), expected_cli_str)

    def test_create_server_no_gpu(self):
        self._test_create_server(gpus=[])

    def test_create_server_select_gpu(self):
        self._test_create_server(gpus=self._sys_gpus[:1])

    def _test_start_stop_gpus(self, gpus):
        device_requests = [device.device_id() for device in gpus]

        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Create server in docker, start , wait, and stop
        self.server = TritonServerFactory.create_server_docker(
            image=TRITON_IMAGE, config=server_config, gpus=gpus
        )

        # Start server check that mocked api is called
        self.server.start()
        self.server_docker_mock.assert_server_process_start_called_with(
            f"{TRITON_DOCKER_BIN_PATH} {server_config.to_cli_string()}",
            MODEL_REPOSITORY_PATH,
            TRITON_IMAGE,
            device_requests,
            gpus,
            8000,
            8001,
            8002,
        )

        self.server_docker_mock.raise_exception_on_container_run()
        with self.assertRaises(TritonModelAnalyzerException):
            self.server.start()
        self.server_docker_mock.stop_raise_exception_on_container_run()

        # Stop container and check api calls
        self.server.stop()
        self.server_docker_mock.assert_server_process_terminate_called()

        # Create local server which runs triton as a subprocess
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=gpus
        )

        # Check that API functions are called
        self.server.start()

        self.server_local_mock.assert_server_process_start_called_with(
            cmd=[TRITON_LOCAL_BIN_PATH, "--model-repository", MODEL_REPOSITORY_PATH],
            gpus=gpus,
            stdout=self.server._log_file,
        )

        self.server.stop()
        self.server_local_mock.assert_server_process_terminate_called()

    def test_start_stop_gpus_no_gpu(self):
        self._test_start_stop_gpus(gpus=[])

    def test_start_stop_gpus_select_gpu(self):
        self._test_start_stop_gpus(gpus=self._sys_gpus[:1])

    def start_stop_docker_args(self):
        device_requests, gpu_uuids = self._find_correct_gpu_settings(self._sys_gpus)

        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Create mounts and labels
        mounts = ["/host/path:/dest/path:ro", "/another/host/path:/some/dest/path:rw"]
        labels = {"RUNNER_ID": "TEST_RUNNER_ID"}

        environment = {"VARIABLE": "VALUE"}
        # Create server in docker, start , wait, and stop
        self.server = TritonServerFactory.create_server_docker(
            image=TRITON_IMAGE,
            config=server_config,
            gpus=self._sys_gpus,
            mounts=mounts,
            labels=labels,
        )

        # Start server check that mocked api is called
        self.server.start(env=environment)
        self.server_docker_mock.assert_server_process_start_called_with(
            f"{TRITON_DOCKER_BIN_PATH} {server_config.to_cli_string()}",
            MODEL_REPOSITORY_PATH,
            TRITON_IMAGE,
            device_requests,
            gpu_uuids,
            8000,
            8001,
            8002,
            mounts,
            labels,
        )

        # Stop container and check api calls
        self.server.stop()
        self.server_docker_mock.assert_server_process_terminate_called()

    def _test_get_logs(self, gpus):
        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Check docker server logs
        self.server = TritonServerFactory.create_server_docker(
            image=TRITON_IMAGE, config=server_config, gpus=gpus
        )
        self.server.start()
        self.server.stop()
        self.server_docker_mock.assert_server_process_terminate_called()
        self.assertEqual(self.server.logs(), "Triton Server Test Log")

        # Create local server logs
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=gpus
        )
        self.server.start()
        self.server.stop()
        self.server_local_mock.assert_server_process_terminate_called()
        self.assertEqual(self.server.logs(), "Triton Server Test Log")

    def _test_cpu_stats(self, gpus):
        device_requests = [device.device_id() for device in gpus]

        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Test local server cpu_stats
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=gpus
        )
        self.server.start()
        _, _ = self.server.cpu_stats()
        self.server_local_mock.assert_cpu_stats_called()
        self.server.stop()

        # Test docker server cpu stats
        self.server = TritonServerFactory.create_server_docker(
            image=TRITON_IMAGE, config=server_config, gpus=gpus
        )
        self.server.start()

        # The following needs to be called as it resets exec_run return value
        self.server_docker_mock.assert_server_process_start_called_with(
            f"{TRITON_DOCKER_BIN_PATH} {server_config.to_cli_string()}",
            MODEL_REPOSITORY_PATH,
            TRITON_IMAGE,
            device_requests,
            gpus,
            8000,
            8001,
            8002,
        )
        _, _ = self.server.cpu_stats()
        self.server_docker_mock.assert_cpu_stats_called()
        self.server.stop()

    def test_cpu_stats_no_gpu(self):
        self._test_cpu_stats(gpus=[])

    def test_cpu_stats_select_gpu(self):
        self._test_cpu_stats(gpus=self._sys_gpus[:1])

    def tearDown(self):
        # In case test raises exception
        if self.server is not None:
            self.server.stop()

        # Stop mocking
        self.gpu_device_factory_mock.stop()
        self.server_docker_mock.stop()
        self.server_local_mock.stop()
        self.os_mock.stop()
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
