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

from unittest.mock import MagicMock, Mock, patch

from .mock_base import MockBase


class MockJSONMethods(MockBase):
    """
    Mocks the methods for the os module
    """

    def __init__(self):
        json_attrs = {"load": MagicMock(), "dump": MagicMock()}
        self.patcher_json = patch(
            "model_analyzer.state.analyzer_state_manager.json", Mock(**json_attrs)
        )
        super().__init__()
        self._fill_patchers()

    def start(self):
        """
        start the patchers
        """

        self.json_mock = self.patcher_json.start()

    def _fill_patchers(self):
        """
        Fills the patcher list for destruction
        """

        self._patchers.append(self.patcher_json)

    def set_json_load_return_value(self, value):
        """
        Sets the return value for json load
        """

        self.json_mock.load.return_value = value

    def set_json_load_side_effect(self, effect):
        """
        Sets a side effet
        """

        self.json_mock.load.side_effect = effect
