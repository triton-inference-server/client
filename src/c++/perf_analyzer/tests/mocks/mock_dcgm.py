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

from unittest.mock import MagicMock, patch

from .mock_base import MockBase
from .mock_dcgm_agent import MockDCGMAgent
from .mock_dcgm_field_group_watcher import MockDCGMFieldGroupWatcherHelper


class MockDCGM(MockBase):
    """
    Mocks dcgm_agent methods.
    """

    def __init__(self):
        super().__init__()
        self._fill_patchers()

    def _fill_patchers(self):
        patchers = self._patchers

        structs_imports_path = [
            "model_analyzer.monitor.dcgm.dcgm_monitor",
            "model_analyzer.device.gpu_device_factory",
        ]
        for import_path in structs_imports_path:
            patchers.append(patch(f"{import_path}.structs._dcgmInit", MagicMock()))

        dcgm_agent_imports_path = [
            "model_analyzer.monitor.dcgm.dcgm_monitor",
            "model_analyzer.device.gpu_device_factory",
        ]
        for import_path in dcgm_agent_imports_path:
            patchers.append(patch(f"{import_path}.dcgm_agent", MockDCGMAgent))

        patchers.append(
            patch(
                "model_analyzer.monitor.dcgm.dcgm_monitor.dcgm_field_helpers.DcgmFieldGroupWatcher",
                MockDCGMFieldGroupWatcherHelper,
            )
        )
