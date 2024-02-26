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

from unittest.mock import MagicMock, mock_open, patch

from .mock_base import MockBase


class MockIOMethods(MockBase):
    """
    A class that mocks filesystem
    operations open and write
    """

    def __init__(self, mock_paths, read_data=[]):
        self._mock_paths = mock_paths
        self._patchers_open = {}
        self._patchers_print = {}
        self._open_mocks = {}
        self._print_mocks = {}
        for i in range(len(mock_paths)):
            path = mock_paths[i]
            data = read_data[i] if i < len(read_data) else None
            self._patchers_open[path] = patch(f"{path}.open", mock_open(read_data=data))
            self._patchers_print[path] = patch(f"{path}.print", MagicMock())
        super().__init__()
        self._fill_patchers()

    def start(self):
        """
        start the patchers
        """

        for path in self._patchers_open:
            self._open_mocks[path] = self._patchers_open[path].start()
            self._print_mocks[path] = self._patchers_print[path].start()

    def _fill_patchers(self):
        """
        Loads patchers into list
        """

        for patcher in self._patchers_open.values():
            self._patchers.append(patcher)

        for patcher in self._patchers_print.values():
            self._patchers.append(patcher)

    def raise_exception_on_open(self):
        """
        Raises an OSError when open is called
        """

        for mock in self._open_mocks.values():
            mock.side_effect = OSError

    def raise_exception_on_write(self):
        """
        Raises an OSError when write is called
        """

        for mock in self._open_mocks.values():
            mock.return_value.write.side_effect = OSError

    def assert_open_called_with_args(self, path, filename):
        """
        Asserts that file open was called
        with given arguments
        """

        self._open_mocks[path].assert_called_with(filename)

    def assert_write_called_with_args(self, path, out):
        """
        Asserts that file write was called
        with given arguments
        """

        self._open_mocks[path].return_value.write.assert_called_with(out)

    def assert_print_called_with_args(self, path, out):
        """
        Asserts that print was called
        with given arguments
        """

        self._print_mocks[path].assert_called_with(out, end="")

    def reset(self):
        for mock in self._open_mocks.values():
            mock.side_effect = None
            mock.return_value.write.side_effect = None
