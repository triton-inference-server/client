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

from abc import abstractmethod

from .mock_base import MockBase


class MockServerMethods(MockBase):
    """
    Interface for a mock server declaring
    the methods it must provide.
    """

    @abstractmethod
    def assert_server_process_start_called_with(self, *args, **kwargs):
        """
        Asserts that the tritonserver process was started with
        the supplied arguments
        """

    @abstractmethod
    def assert_server_process_terminate_called(self):
        """
        Assert that the server process was stopped
        """

    @abstractmethod
    def assert_cpu_stats_called(self):
        """
        Assert that correct cpu memory stats
        api functions wer called for this implementation
        """
