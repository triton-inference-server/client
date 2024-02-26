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

from model_analyzer.config.generate.search_dimension import SearchDimension

from .common import test_result_collector as trc


class TestSearchDimension(trc.TestResultCollector):
    def test_linear(self):
        sd = SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR, 0, 10)

        self.assertEqual("foo", sd.get_name())
        self.assertEqual(0, sd.get_min_idx())
        self.assertEqual(10, sd.get_max_idx())
        self.assertEqual(1, sd.get_value_at_idx(0))
        self.assertEqual(2, sd.get_value_at_idx(1))
        self.assertEqual(3, sd.get_value_at_idx(2))

    def test_exponential(self):
        sd = SearchDimension("foo", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        self.assertEqual(1, sd.get_value_at_idx(0))
        self.assertEqual(2, sd.get_value_at_idx(1))
        self.assertEqual(4, sd.get_value_at_idx(2))
        self.assertEqual(8, sd.get_value_at_idx(3))

    def test_out_of_bounds(self):
        sd = SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR, 2, 10)

        with self.assertRaises(IndexError):
            sd.get_value_at_idx(1)

        with self.assertRaises(IndexError):
            sd.get_value_at_idx(11)

    def test_no_max(self):
        sd = SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR)

        # Confirm no assert
        sd.get_value_at_idx(100000000)
