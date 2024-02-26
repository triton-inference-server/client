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
from unittest.mock import MagicMock, Mock, patch

from .mock_server import MockServerMethods


class MockServerLocalMethods(MockServerMethods):
    """
    Mocks the subprocess functions used in
    model_analyzer/triton/server/server_local.py.
    Provides functions to check operation.
    """

    def __init__(self):
        memory_full_attrs = {"uss": 0}
        virtual_memory_attrs = {"available": 0}
        process_attrs = {
            "memory_full_info": Mock(return_value=Mock(**memory_full_attrs))
        }
        psutil_attrs = {
            "Process": Mock(return_value=Mock(**process_attrs)),
            "virtual_memory": Mock(return_value=Mock(**virtual_memory_attrs)),
        }
        Popen_attrs = {
            "communicate.return_value": ("Triton Server Test Log", "Test Error")
        }
        self.patcher_popen = patch(
            "model_analyzer.triton.server.server_local.Popen",
            Mock(return_value=Mock(**Popen_attrs)),
        )
        self.patcher_stdout = patch(
            "model_analyzer.triton.server.server_local.STDOUT", MagicMock()
        )
        self.patcher_pipe = patch(
            "model_analyzer.triton.server.server_local.DEVNULL", MagicMock()
        )
        self.patcher_psutil = patch(
            "model_analyzer.triton.server.server_local.psutil", Mock(**psutil_attrs)
        )
        super().__init__()
        self._fill_patchers()

    def start(self):
        """
        Start the patchers
        """

        self.popen_mock = self.patcher_popen.start()
        self.stdout_mock = self.patcher_stdout.start()
        self.pipe_mock = self.patcher_pipe.start()
        self.psutil_mock = self.patcher_psutil.start()

    def _fill_patchers(self):
        """
        Fill patcher list
        """

        self._patchers.append(self.patcher_popen)
        self._patchers.append(self.patcher_stdout)
        self._patchers.append(self.patcher_pipe)
        self._patchers.append(self.patcher_psutil)

    def assert_server_process_start_called_with(self, cmd, gpus, stdout=MagicMock()):
        """
        Asserts that Popen was called
        with the cmd provided.
        """

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join([gpu.device_uuid() for gpu in gpus])

        self.popen_mock.assert_called_once_with(
            cmd,
            stdout=stdout,
            stderr=self.stdout_mock,
            start_new_session=True,
            universal_newlines=True,
            env=env,
        )

    def assert_server_process_terminate_called(self):
        """
        Asserts that terminate was called on
        the pipe (Popen object).
        """

        self.popen_mock.return_value.terminate.assert_called()
        self.popen_mock.return_value.communicate.assert_called()

    def set_cpu_memory_values(self, used_mem, free_mem):
        """
        Sets the value of psutil.Process.memory_full_info.uss
        and psutil.virtual_memory.available
        """

        self.psutil_mock.Process.return_value.memory_full_info.return_value.uss = (
            used_mem
        )
        self.psutil_mock.virtual_memory.return_value.available = free_mem

    def assert_cpu_stats_called(self):
        """
        Checks the call to psutil.Process.memory_full_info and psutil.virtual_memory
        """

        self.psutil_mock.Process.return_value.memory_full_info.assert_called()
        self.psutil_mock.virtual_memory.assert_called()

    def assert_cpu_stats_not_called(self):
        """
        Checks no calls to psutil.Process.memory_full_info and psutil.virtual_memory
        """

        self.psutil_mock.Process.return_value.memory_full_info.assert_not_called()
        self.psutil_mock.virtual_memory.assert_not_called()
