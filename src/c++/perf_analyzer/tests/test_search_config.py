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

from model_analyzer.config.generate.search_config import SearchConfig
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.generate.search_dimensions import SearchDimensions

from .common import test_result_collector as trc


class TestSearchConfig(trc.TestResultCollector):
    def test_basic(self):
        sc = SearchConfig(SearchDimensions(), 0, 0)
        self.assertEqual(0, sc.get_num_dimensions())

    def test_config(self):
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        sc = SearchConfig(dimensions=dims, radius=4, min_initialized=2)

        self.assertEqual(2, sc.get_num_dimensions())
        self.assertEqual(4, sc.get_radius())
        self.assertEqual(2, sc.get_min_initialized())

        self.assertEqual("foo", sc.get_dimension(0).get_name())
        self.assertEqual("bar", sc.get_dimension(1).get_name())

        self.assertEqual(7, sc.get_dimension(0).get_value_at_idx(6))
        self.assertEqual(64, sc.get_dimension(1).get_value_at_idx(6))

    def test_get_min_indexes(self):
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR, 1, 10),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )
        sc = SearchConfig(dims, 0, 0)

        self.assertEqual([1, 0], sc.get_min_indexes())

    def test_get_neighborhood_config(self):
        """
        Test that we can get a NeighborhoodConfig from a SearchConfig and properly override the radius
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        sc = SearchConfig(dimensions=dims, radius=4, min_initialized=2)

        nc = sc.get_neighborhood_config(radius=5)
        self.assertEqual(2, nc.get_num_dimensions())
        self.assertEqual(5, nc.get_radius())
        self.assertEqual(2, nc.get_min_initialized())

        self.assertEqual("foo", nc.get_dimension(0).get_name())
        self.assertEqual("bar", nc.get_dimension(1).get_name())

        self.assertEqual(7, nc.get_dimension(0).get_value_at_idx(6))
        self.assertEqual(64, nc.get_dimension(1).get_value_at_idx(6))
