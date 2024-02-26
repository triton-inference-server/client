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

from abc import ABC, abstractmethod


class MockBase(ABC):
    """
    Base abstract class for all the mocks
    """

    def __init__(self):
        self._patchers = []

    @abstractmethod
    def _fill_patchers(self):
        pass

    def start(self):
        for patch in self._patchers:
            patch.start()

    def stop(self):
        for patch in self._patchers:
            patch.stop()
