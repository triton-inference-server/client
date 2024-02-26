#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.generate.search_dimensions import SearchDimensions

from .common import test_result_collector as trc


class TestSearchDimensions(trc.TestResultCollector):
    def setUp(self):
        self._dims = SearchDimensions()
        self._dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_LINEAR),
            ],
        )

        self._dims.add_dimensions(
            1, [SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR)]
        )

    def test_list_behavior(self):
        """
        Confirm the following list-like behavior:
            - get length like a list
            - get from index like a list
            - iterate over values
            - raise IndexError on out of bounds index
        """
        dims = self._dims
        self.assertEqual(len(dims), 3)

        dim0 = dims[0]
        dim1 = dims[1]
        dim2 = dims[2]
        self.assertEqual(dim0.get_name(), "foo")
        self.assertEqual(dim1.get_name(), "bar")
        self.assertEqual(dim2.get_name(), "foo")

        count = 0
        for dim in dims:
            self.assertTrue(isinstance(dim, SearchDimension))
            count += 1
        self.assertEqual(count, 3)

        with self.assertRaises(IndexError):
            _ = dims[3]

    def test_get_values(self):
        """
        Test get_values_for_coordinate() functionality
        """
        dims = self._dims
        vals = dims.get_values_for_coordinate(Coordinate([0, 2, 4]))
        expected_vals = {
            0: {
                "foo": 1,
                "bar": 3,
            },
            1: {"foo": 5},
        }
        self.assertEqual(vals, expected_vals)
