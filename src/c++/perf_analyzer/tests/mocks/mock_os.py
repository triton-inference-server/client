#!/usr/bin/env python3

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .mock_base import MockBase


class MockOSMethods(MockBase):
    """
    Mocks the methods for the os module
    """

    def __init__(self, mock_paths):
        path_attrs = {
            "join": MagicMock(return_value=""),
            "abspath": MagicMock(return_value=""),
            "isdir": MagicMock(return_value=True),
            "exists": MagicMock(return_value=True),
            "isfile": MagicMock(return_value=True),
            "split": os.path.split,
        }
        os_attrs = {
            "path": Mock(**path_attrs),
            "mkdir": MagicMock(),
            "makedirs": MagicMock(),
            "getenv": MagicMock(),
            "listdir": MagicMock(return_value=[]),
            "environ": {"VARIABLE": "value"},
        }
        self._mock_paths = mock_paths
        self._patchers_os = {}
        self._os_mocks = {}
        for path in mock_paths:
            self._patchers_os[path] = patch(f"{path}.os", Mock(**os_attrs))
        super().__init__()
        self._fill_patchers()

    def start(self):
        """
        start the patchers
        """

        for path in self._mock_paths:
            self._os_mocks[path] = self._patchers_os[path].start()

    def _fill_patchers(self):
        """
        Fills the patcher list for destruction
        """

        for patcher in self._patchers_os.values():
            self._patchers.append(patcher)

    def set_os_path_exists_return_value(self, value):
        """
        Sets the return value for os.path.exists
        """

        for mock in self._os_mocks.values():
            mock.path.exists.return_value = value

    def set_os_path_join_return_value(self, value):
        """
        Sets the return value for os.path.join
        """

        for mock in self._os_mocks.values():
            mock.path.join.return_value = value

    def set_os_path_isfile_return_value(self, value):
        """
        Sets the return value for os.path.isfile
        """

        for mock in self._os_mocks.values():
            mock.path.isfile.return_value = value
