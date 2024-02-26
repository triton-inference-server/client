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

from unittest.mock import MagicMock, patch

from .mock_base import MockBase


class MockGPUDeviceFactory(MockBase):
    def __init__(self, gpus):
        self._cuda_visible_gpus = gpus
        super().__init__()
        self._fill_patchers()

    def _fill_patchers(self):
        self._patchers.append(
            patch(
                "model_analyzer.device.gpu_device_factory.GPUDeviceFactory.get_cuda_visible_gpus",
                MagicMock(return_value=dict.fromkeys(self._cuda_visible_gpus)),
            )
        )
