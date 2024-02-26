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

from copy import deepcopy

from model_analyzer.config.generate.coordinate import Coordinate

from .common import test_result_collector as trc


class TestCoordinate(trc.TestResultCollector):
    def test_basic(self):
        x = Coordinate([0, 0])

        self.assertEqual(x[0], 0)
        self.assertEqual(x[1], 0)

        x[0] = 2
        x[1] = 4

        self.assertEqual(x[0], 2)
        self.assertEqual(x[1], 4)

    def test_out_of_range(self):
        x = Coordinate([0, 0])
        with self.assertRaises(IndexError):
            _ = x[2]

    def test_assignment(self):
        c1 = Coordinate([2, 4])

        c2 = deepcopy(c1)

        self.assertEqual(c2[0], 2)
        self.assertEqual(c2[1], 4)

        # Confirm c1 unchanged by c2 update
        c2[0] = 5

        self.assertEqual(c1[0], 2)
        self.assertEqual(c1[1], 4)

    def test_coordinate_addition(self):
        c1 = Coordinate([2, 4])
        c2 = Coordinate([5, 1])

        c3 = c1 + c2
        self.assertEqual(c3[0], 7)
        self.assertEqual(c3[1], 5)

    def test_integer_addition(self):
        c1 = Coordinate([2, 4])
        c2 = c1 + 3

        self.assertEqual(c2[0], 5)
        self.assertEqual(c2[1], 7)

    def test_float_addition(self):
        c1 = Coordinate([2, 4])

        c2 = c1 + 3.5

        self.assertEqual(c2[0], 5.5)
        self.assertEqual(c2[1], 7.5)

    def test_division(self):
        c1 = Coordinate([7, 6])

        c1 /= 2

        self.assertEqual(c1[0], 3.5)
        self.assertEqual(c1[1], 3)

    def test_multiplication(self):
        c1 = Coordinate([7, 6])

        c1 *= 2

        self.assertEqual(c1[0], 14)
        self.assertEqual(c1[1], 12)

    def test_round(self):
        c1 = Coordinate([0.1, 4.6, 3.9])
        c1.round()
        self.assertEqual(c1, Coordinate([0, 5, 4]))

    def test_iterator(self):
        c1 = Coordinate([2, 4, 1, 7])

        values = []
        for x in c1:
            values.append(x)

        self.assertEqual(values, [2, 4, 1, 7])

    def test_enumerate(self):
        c1 = Coordinate([2, 4, 1, 7])

        indexes = []
        values = []
        for i, x in enumerate(c1):
            indexes.append(i)
            values.append(x)

        self.assertEqual(indexes, [0, 1, 2, 3])
        self.assertEqual(values, [2, 4, 1, 7])

    def test_stringification(self):
        c1 = Coordinate([2, 4, 1, 7])
        self.assertEqual("[2, 4, 1, 7]", c1.__str__())

    def test_compare(self):
        c1 = Coordinate([1, 3, 2])
        c2 = Coordinate([1, 3, 2])
        c3 = Coordinate([1, 3, 1])
        c4 = Coordinate([1, 3, 3])
        c5 = Coordinate([2, 1, 3])
        c6 = Coordinate([0, 1, 3])

        self.assertEqual(c1, c2)
        self.assertGreater(c1, c3)
        self.assertLess(c3, c1)
        self.assertLess(c1, c4)
        self.assertLess(c1, c5)
        self.assertGreater(c1, c6)
