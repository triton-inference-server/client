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


class MockMatplotlibMethods(MockBase):
    """
    Mock class that mocks
    matplotlib
    """

    def __init__(self):
        plt_attrs = {"subplots": Mock(return_value=(MagicMock(), MagicMock()))}
        self.patcher_pyplot = patch(
            "model_analyzer.plots.simple_plot.plt", Mock(**plt_attrs)
        )
        super().__init__()
        self._fill_patchers()

    def start(self):
        """
        Start mock
        """

        self.pyplot_mock = self.patcher_pyplot.start()

    def _fill_patchers(self):
        """
        Add patchers to list
        """

        self._patchers.append(self.patcher_pyplot)

    def assert_called_subplots(self):
        """
        Checks for a call to subplots
        """

        self.pyplot_mock.subplots.assert_called()

    def assert_called_plot_with_args(self, x_data, y_data, marker, label):
        """
        Checks for call to axes.plot
        """

        self.pyplot_mock.subplots.return_value[1].plot.assert_called_with(
            x_data, y_data, marker=marker, label=label
        )

    def assert_called_save_with_args(self, filepath):
        """
        Checks for call to figure.savefig
        """

        self.pyplot_mock.subplots.return_value[0].savefig.assert_called_with(filepath)
