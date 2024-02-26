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

from unittest.mock import Mock, patch

from .mock_base import MockBase


class MockRequests(MockBase):
    def __init__(self, mock_paths):
        self._mock_paths = mock_paths
        response_attrs = {"content": Mock()}
        request_attrs = {"get": Mock(**response_attrs)}
        self._patchers_requests = {}
        self._request_mocks = {}
        for path in mock_paths:
            self._patchers_requests[path] = patch(
                f"{path}.requests", Mock(**request_attrs)
            )
        super().__init__()
        self._fill_patchers()

    def _fill_patchers(self):
        """
        Fills the patcher list for destruction
        """

        for patcher in self._patchers_requests.values():
            self._patchers.append(patcher)

    def start(self):
        """
        start the patchers
        """

        for path in self._mock_paths:
            self._request_mocks[path] = self._patchers_requests[path].start()

    def set_get_request_response(self, response):
        for mock in self._request_mocks.values():
            mock.get.return_value.content = response
