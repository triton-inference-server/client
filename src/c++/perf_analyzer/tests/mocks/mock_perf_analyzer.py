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

from unittest.mock import MagicMock, Mock, patch

from .mock_base import MockBase


class MockCalledProcessError(Exception):
    """
    A mock of subprocess.CalledProcessError
    """

    def __init__(self):
        self.returncode = 1
        self.cmd = ["dummy command"]
        self.output = "mock output"


class MockPerfAnalyzerMethods(MockBase):
    """
    Mocks the subprocess module functions used in
    model_analyzer/perf_analyzer/perf_analyzer.py
    Provides functions to check operation.
    """

    def __init__(self):
        self.mock_popen = MagicMock()
        self.mock_popen.pid = 10
        self.mock_popen.returncode = 0

        self.mock_popen_constructor = MagicMock()
        self.mock_popen_constructor.return_value = self.mock_popen

        self.patcher_popen_stdout_read = patch(
            "model_analyzer.perf_analyzer.perf_analyzer.Popen",
            self.mock_popen_constructor,
        )

        self.mock_file = MagicMock()
        self.mock_file.read.return_value = b""
        self.patcher_open = patch(
            "model_analyzer.perf_analyzer.perf_analyzer.tempfile.NamedTemporaryFile",
            Mock(return_value=self.mock_file),
        )
        super().__init__()
        self._fill_patchers()

    def start(self):
        """
        Start the patchers
        """

        self.popen_stdout_read = self.patcher_popen_stdout_read.start()
        self.open_mock = self.patcher_open.start()

    def _fill_patchers(self):
        """
        Fills patcher list
        """
        self._patchers.append(self.patcher_open)
        self._patchers.append(self.patcher_popen_stdout_read)

    def set_perf_analyzer_result_string(self, output_string):
        """
        Sets the return value of mock_file
        """

        self.mock_file.read.return_value = output_string.encode("utf-8")

    def get_perf_analyzer_popen_call_count(self):
        """
        Get PerfAnalyzer.Popen call count
        """

        return self.mock_popen_constructor.call_count

    def set_perf_analyzer_return_code(self, returncode):
        """
        Sets the returncode of Popen process
        """

        self.mock_popen.returncode = returncode

    def reset(self):
        """
        Resets the side effects
        and return values of the
        mocks in this module
        """
        self.mock_file.read.return_value = b""
