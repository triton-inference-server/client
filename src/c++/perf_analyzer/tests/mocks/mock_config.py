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

from unittest.mock import MagicMock, mock_open, patch

from .mock_base import MockBase


class MockConfig(MockBase):
    def __init__(self, args, yaml_file_content):
        self.args = args
        self.yaml_file_content = yaml_file_content
        super().__init__()
        self._fill_patchers()

    def _fill_patchers(self):
        patchers = self._patchers

        patchers.append(
            patch("builtins.open", mock_open(read_data=self.yaml_file_content))
        )
        patchers.append(patch("sys.argv", self.args))
        patchers.append(patch("numba.cuda.is_available", MagicMock(True)))
