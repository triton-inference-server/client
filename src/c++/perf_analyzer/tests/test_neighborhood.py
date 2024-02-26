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

from typing import List
from unittest.mock import MagicMock, patch

from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.config.generate.coordinate_data import CoordinateData
from model_analyzer.config.generate.neighborhood import Neighborhood
from model_analyzer.config.generate.search_config import NeighborhoodConfig
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.generate.search_dimensions import SearchDimensions

from .common import test_result_collector as trc
from .common.test_utils import (
    construct_constraint_manager,
    construct_run_config_measurement,
)


class TestNeighborhood(trc.TestResultCollector):
    def setUp(self):
        self.constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA
            """
        )

    # Coordinates can't be sorted by default, but if we convert each one
    # to a list those can be sorted
    def _sort_coordinates(self, coordinates: List[Coordinate]) -> List[Coordinate]:
        sorted_coordinates = coordinates
        sorted_coordinates.sort(key=lambda c: [x for x in c])
        return sorted_coordinates

    def tearDown(self):
        patch.stopall()

    def _construct_rcm(self, throughput: float, latency: float):
        model_config_name = ["modelA_config_0"]

        # yapf: disable
        non_gpu_metric_values = [{
            "perf_throughput": throughput,
            "perf_latency_p99": latency
        }]
        # yapf: enable

        metric_objectives = [{"perf_throughput": 1}]
        weights = [1]

        rcm = construct_run_config_measurement(
            model_name="modelA",
            model_config_names=model_config_name,
            model_specific_pa_params=MagicMock(),
            gpu_metric_values={},
            non_gpu_metric_values=non_gpu_metric_values,
            constraint_manager=self.constraint_manager,
            metric_objectives=metric_objectives,
            model_config_weights=weights,
        )
        return rcm

    def test_calc_distance(self):
        a = Coordinate([1, 4, 6, 3])
        b = Coordinate([4, 2, 6, 0])

        # Euclidean distance is the square root of the
        # sum of the distances of the coordinates
        #
        # Distance = sqrt( (1-4)^2 + (4-2)^2 + (6-6)^2 + (3-0)^2)
        # Distance = sqrt( 9 + 4 + 0 + 9 )
        # Distance = sqrt(22)
        # Distance = 4.69
        self.assertAlmostEqual(Neighborhood.calc_distance(a, b), 4.69, places=3)

    def test_create_neighborhood(self):
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(
            nc, home_coordinate=Coordinate([1, 1, 1]), coordinate_data=CoordinateData()
        )

        # These are all values within radius of 2 from [1,1,1]
        # but within the bounds (no negative values)
        #
        expected_neighborhood = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 3],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [1, 3, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
            [3, 1, 1],
        ]

        expected_coordinates = [Coordinate(x) for x in expected_neighborhood]

        self.assertEqual(
            self._sort_coordinates(n._neighborhood),
            self._sort_coordinates(expected_coordinates),
        )

    def test_large_neighborhood(self):
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("a1", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("b1", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("c1", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )
        dims.add_dimensions(
            1,
            [
                SearchDimension("a2", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("b2", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("c2", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )
        nc = NeighborhoodConfig(dims, radius=3, min_initialized=3)
        n = Neighborhood(
            nc,
            home_coordinate=Coordinate([1, 1, 1, 1, 1, 1]),
            coordinate_data=CoordinateData(),
        )

        self.assertEqual(2328, len(n._neighborhood))

    def test_num_initialized(self):
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]), coordinate_data=cd)

        rcm = self._construct_rcm(throughput=100, latency=80)

        # Start with 0 initialized
        self.assertEqual(0, len(n._get_coordinates_with_valid_measurements()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Home coordinate is ignored. No change to num initialized/visited.
        cd.set_measurement(Coordinate([1, 1, 1]), rcm)
        self.assertEqual(0, len(n._get_coordinates_with_valid_measurements()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Incremented to 1 initialized
        cd.set_measurement(Coordinate([0, 0, 0]), rcm)
        self.assertEqual(1, len(n._get_coordinates_with_valid_measurements()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set same point. No change to num initialized
        cd.set_measurement(Coordinate([0, 0, 0]), rcm)
        self.assertEqual(1, len(n._get_coordinates_with_valid_measurements()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set a point outside of neighborhood
        cd.set_measurement(Coordinate([0, 0, 4]), rcm)
        self.assertEqual(1, len(n._get_coordinates_with_valid_measurements()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set a third point inside of neighborhood
        cd.set_measurement(Coordinate([1, 0, 0]), rcm)
        self.assertEqual(2, len(n._get_coordinates_with_valid_measurements()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set the none measurement. No changed to num initialized.
        cd.set_measurement(Coordinate([0, 1, 1]), None)
        self.assertEqual(2, len(n._get_coordinates_with_valid_measurements()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set the last point inside of neighborhood
        cd.set_measurement(Coordinate([1, 1, 0]), rcm)
        self.assertEqual(3, len(n._get_coordinates_with_valid_measurements()))
        self.assertTrue(n.enough_coordinates_initialized())

    def test_get_all_adjacent_neighbors(self):
        """
        Test that _get_all_adjacent_neighbors() works, and understands dimension bounds

        For this test, home is [0,1,4].

        The possible adjacent neighbors are:
          [-1,1,4], [1,1,4], [0,0,4], [0,2,4], [0,1,3], [0,1,5]

        However, [-1,1,4] and [0,1,5] are outside of the dimension bounds and should not
        be part of the returned list
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("a", SearchDimension.DIMENSION_TYPE_LINEAR, min=0),
                SearchDimension("b", SearchDimension.DIMENSION_TYPE_LINEAR, min=0),
                SearchDimension(
                    "c", SearchDimension.DIMENSION_TYPE_LINEAR, min=1, max=4
                ),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(
            nc, home_coordinate=Coordinate([0, 1, 4]), coordinate_data=CoordinateData()
        )

        adjacent_neighbors = n._get_all_adjacent_neighbors()
        expected_list = [
            Coordinate([1, 1, 4]),
            Coordinate([0, 0, 4]),
            Coordinate([0, 2, 4]),
            Coordinate([0, 1, 3]),
        ]

        self.assertEqual(adjacent_neighbors, expected_list)

    def test_num_initialized_slow_mode(self):
        """
        Test that _enough_coordinates_initialized() works in slow mode

        Start with home=[0,0,1] in slow mode.
        Only once all of the adjacent neighbors are added should
        enough_coordinates_initialized() return true

        Those adjacent neighbors are [1,0,1], [0,1,1], [0,0,0], [0,0,2]
        """

        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([0, 0, 1]), coordinate_data=cd)
        n.force_slow_mode()

        # Start with None initialized
        self.assertFalse(n.enough_coordinates_initialized())

        # Home coordinate is ignored
        cd.set_measurement(Coordinate([0, 0, 1]), MagicMock())
        self.assertFalse(n.enough_coordinates_initialized())

        # Add 3 of the 4 neighbors
        cd.set_measurement(Coordinate([1, 0, 1]), MagicMock())
        self.assertFalse(n.enough_coordinates_initialized())
        cd.set_measurement(Coordinate([0, 1, 1]), MagicMock())
        self.assertFalse(n.enough_coordinates_initialized())
        cd.set_measurement(Coordinate([0, 0, 0]), MagicMock())
        self.assertFalse(n.enough_coordinates_initialized())

        # Add the final neighbor
        cd.set_measurement(Coordinate([0, 0, 2]), MagicMock())
        self.assertTrue(n.enough_coordinates_initialized())

    def test_is_slow_mode(self):
        """
        Test that _is_slow_mode() only returns true by these 3 conditions:
        - force_slow_mode() has been called
        - Home is passing and anyone in the neighborhood is failing
        - Home is failing and anyone in the neighborhood is passing

        Test this by running a number of sub-tests with different configurations
        of num_passing, num_failing, home_passing, and home_measured
        """

        # Home not measured -> False
        self._test_is_slow_mode_helper(home_measured=False, expected_result=False)

        # Home passing, all passing -> False
        self._test_is_slow_mode_helper(
            home_passing=True, num_passing=3, num_failing=0, expected_result=False
        )

        # Same as previous, but force_slow_mode is called -> True
        self._test_is_slow_mode_helper(
            force_slow=True,
            home_passing=True,
            num_passing=3,
            num_failing=0,
            expected_result=True,
        )

        # Home passing, some failing -> True
        self._test_is_slow_mode_helper(
            num_passing=3, num_failing=1, home_passing=True, expected_result=True
        )

        # Home passing, all failing -> True
        self._test_is_slow_mode_helper(
            num_passing=0, num_failing=3, home_passing=True, expected_result=True
        )

        # Home failing, all failing -> False
        self._test_is_slow_mode_helper(
            num_passing=0, num_failing=3, home_passing=False, expected_result=False
        )

        # Home failing, some passing -> True
        self._test_is_slow_mode_helper(
            num_passing=1, num_failing=3, home_passing=False, expected_result=True
        )

        # Home failing, all passing -> True
        self._test_is_slow_mode_helper(
            num_passing=3, num_failing=0, home_passing=False, expected_result=True
        )

    def _test_is_slow_mode_helper(
        self,
        home_measured=True,
        num_passing=1,
        num_failing=1,
        home_passing=True,
        force_slow=False,
        expected_result=False,
    ):
        total_measurements = num_failing + num_passing

        passing_ret_val = [[0] * num_passing, []]
        all_ret_val = [[0] * total_measurements, []]

        patches = []
        patches.append(
            patch.object(
                Neighborhood, "_is_home_measured", MagicMock(return_value=home_measured)
            )
        )
        patches.append(
            patch.object(
                Neighborhood,
                "_get_measurements_passing_constraints",
                MagicMock(return_value=passing_ret_val),
            )
        )
        patches.append(
            patch.object(
                Neighborhood,
                "_get_all_measurements",
                MagicMock(return_value=all_ret_val),
            )
        )
        patches.append(
            patch.object(
                Neighborhood,
                "_is_home_passing_constraints",
                MagicMock(return_value=home_passing),
            )
        )

        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(
            nc, home_coordinate=Coordinate([1, 1]), coordinate_data=CoordinateData()
        )

        for p in patches:
            p.start()

        if force_slow:
            n.force_slow_mode()

        self.assertEqual(n._is_slow_mode(), expected_result)

        for p in patches:
            p.stop()

    def test_get_all_measurements(self):
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]), coordinate_data=cd)

        rcm0 = self._construct_rcm(throughput=100, latency=50)
        rcm1 = self._construct_rcm(throughput=700, latency=350)
        rcm2 = self._construct_rcm(throughput=300, latency=130)

        cd.set_measurement(Coordinate([1, 1, 1]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([2, 1, 1]), rcm1)
        cd.set_measurement(Coordinate([1, 2, 1]), rcm2)
        cd.set_measurement(Coordinate([1, 2, 2]), None)

        expected_vectors = [
            Coordinate([1, 2, 1]) - Coordinate([1, 1, 1]),
            Coordinate([2, 1, 1]) - Coordinate([1, 1, 1]),
        ]
        expected_measurements = [rcm2, rcm1]

        vectors, measurements = n._get_all_measurements()
        for ev, em in zip(expected_vectors, expected_measurements):
            self.assertIn(ev, vectors)
            self.assertIn(em, measurements)

    def test_get_constraints_passing_measurements(self):
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]), coordinate_data=cd)

        # Constraints:
        #   - Minimum throughput of 100 infer/sec
        #   - Maximum latency of 300 ms
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 100
                  perf_latency_p99:
                    max: 300
            """
        )

        rcm0 = self._construct_rcm(throughput=100, latency=50)  # pass
        rcm0.set_constraint_manager(constraint_manager)

        rcm1 = self._construct_rcm(throughput=700, latency=350)  # fail
        rcm1.set_constraint_manager(constraint_manager)

        rcm2 = self._construct_rcm(throughput=300, latency=130)  # pass
        rcm2.set_constraint_manager(constraint_manager)

        rcm3 = self._construct_rcm(throughput=400, latency=290)  # pass
        rcm3.set_constraint_manager(constraint_manager)

        rcm4 = self._construct_rcm(throughput=850, latency=400)  # fail
        rcm4.set_constraint_manager(constraint_manager)

        cd.set_measurement(Coordinate([1, 1, 1]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([2, 1, 1]), rcm1)
        cd.set_measurement(Coordinate([1, 2, 1]), rcm2)
        cd.set_measurement(Coordinate([1, 1, 2]), rcm3)
        cd.set_measurement(Coordinate([1, 2, 2]), rcm4)

        expected_vectors = [
            Coordinate([1, 1, 2]) - Coordinate([1, 1, 1]),
            Coordinate([1, 2, 1]) - Coordinate([1, 1, 1]),
        ]
        expected_measurements = [rcm3, rcm2]

        vectors, measurements = n._get_measurements_passing_constraints()
        for ev, em in zip(expected_vectors, expected_measurements):
            self.assertIn(ev, vectors)
            self.assertIn(em, measurements)

    def test_step_vector_both_home_and_neighbors_passing_constraints(self):
        """
        Test _get_step_vector method where home and neighbors are passing

          1. Get all vectors and measurements
                [2, 3, 1] - [1, 1, 1] = [1, 2, 0] (throughput=300, latency=130)
                [1, 1, 2] - [1, 1, 1] = [0, 0, 1] (throughput=400, latency=290)

          2. Apply the objective-comparison weights to any non-zero dimensions
                [1,2,0] with weight 1.0 -> [1.0, 1.0, 0.0]
                [0,0,1] with weight 1.2 -> [0.0, 0.0, 1.2]

          3. Sum the results
                [1.0, 1.0, 0.0] + [0.0, 0.0, 1.2] = [1.0, 1.0, 1.2]

          4. Divide by the sum of the vectors:
                [1.0, 1.0, 1.2] / [1,2,1] = [1.0, 0.5, 1.2]
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=3, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]), coordinate_data=cd)

        # Constraints:
        #   - Minimum throughput of 100 infer/sec
        #   - Maximum latency of 300 ms
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 100
                  perf_latency_p99:
                    max: 300
            """
        )

        rcm0 = self._construct_rcm(throughput=100, latency=50)  # pass
        rcm0.set_constraint_manager(constraint_manager)

        rcm1 = self._construct_rcm(throughput=300, latency=130)  # pass
        rcm1.set_constraint_manager(constraint_manager)

        rcm2 = self._construct_rcm(throughput=400, latency=290)  # pass
        rcm2.set_constraint_manager(constraint_manager)

        cd.set_measurement(Coordinate([1, 1, 1]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([2, 3, 1]), rcm1)
        cd.set_measurement(Coordinate([1, 1, 2]), rcm2)

        hm = cd.get_measurement(Coordinate([1, 1, 1]))
        self.assertTrue(hm.is_passing_constraints())

        step_vector = n._get_step_vector()
        expected_step_vector = Coordinate([1.0, 0.5, 1.2])

        self.assertEqual(step_vector, expected_step_vector)

    def test_step_vector_both_home_and_neighbors_failing_constraints(self):
        """
        Test _get_step_vector method where home and neighbors are failing

          1. Get all vectors and measurements
                Home latency is 450
                [3, 2, 1] - [1, 1, 1] = [2, 1, 0] (latency=360)
                [1, 2, 0] - [1, 1, 1] = [0, 1, -1] (latency=540)

          2. Apply the constraint-comparison weights to any non-zero dimensions,
             inverting the weight if coordinate is negative
                [2,1,0] with weight 0.3 -> [0.3, 0.3, 0.0]
                [0,1,-1] with weight -0.3 -> [0.0, -0.3, 0.3]

          3. Sum the results
                [0.3, 0.3, 0.0] + [0.0, -0.3, 0.3] = [0.3, 0.0, 0.3]

          4. Divide by the sum of the absolute value of the vectors:
                [0.3, 0.0, 0.3] / [2,2,1] = [0.15, 0.0, 0.3]

        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=3, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]), coordinate_data=cd)

        # Constraints:
        #   - Minimum throughput of 100 infer/sec
        #   - Maximum latency of 300 ms
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 100
                  perf_latency_p99:
                    max: 300
            """
        )

        rcm0 = self._construct_rcm(throughput=500, latency=450)  # fail
        rcm0.set_constraint_manager(constraint_manager)

        rcm1 = self._construct_rcm(throughput=700, latency=360)  # fail
        rcm1.set_constraint_manager(constraint_manager)

        rcm2 = self._construct_rcm(throughput=850, latency=540)  # fail
        rcm2.set_constraint_manager(constraint_manager)

        cd.set_measurement(Coordinate([1, 1, 1]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([3, 2, 1]), rcm1)
        cd.set_measurement(Coordinate([1, 2, 0]), rcm2)

        hm = cd.get_measurement(Coordinate([1, 1, 1]))
        self.assertFalse(hm.is_passing_constraints())

        step_vector = n._get_step_vector()
        expected_step_vector = Coordinate([0.15, 0.0, 0.3])
        self.assertEqual(step_vector, expected_step_vector)

    def test_determine_new_home_one_dimension(self):
        """
        Test determine_new_home for a case where only
        one of the dimensions increases the measurement.
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([0, 0]), coordinate_data=cd)

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=1, latency=5)

        cd.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([1, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1]), rcm2)

        new_coord = n.determine_new_home()
        self.assertEqual(new_coord, Coordinate([2, 0]))

    def test_determine_new_home_two_dimensions(self):
        """
        Test determine_new_home for a case where both of the
        dimensions increases the measurement equally.
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([0, 0]), coordinate_data=cd)

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=3, latency=5)

        cd.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([1, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1]), rcm2)

        new_coord = n.determine_new_home()
        self.assertEqual(new_coord, Coordinate([2, 2]))

    def test_determine_new_home(self):
        """
        Test determine_new_home for a case where both of the
        dimensions increases the measurement equally
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([0, 0]), coordinate_data=cd)

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=7, latency=5)
        rcm2 = self._construct_rcm(throughput=7, latency=5)

        cd.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([1, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1]), rcm2)

        # 7x increase in throughputs will result in the maximum step of [3,3]
        new_coord = n.determine_new_home()
        self.assertEqual(new_coord, Coordinate([3, 3]))

        new_coord = n.determine_new_home()
        self.assertEqual(new_coord, Coordinate([3, 3]))

    def test_determine_new_home_out_of_bounds(self):
        """
        Test that determine_new_home will clamp the result to
        the search dimension bounds.

        Both dimensions are defined to only be from 2-7. The test sets up
        the case where the next step WOULD be from [3,6] -> [0,9] if not
        for bounding into the defined range.
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension(
                    "foo", SearchDimension.DIMENSION_TYPE_LINEAR, min=2, max=7
                ),
                SearchDimension(
                    "bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL, min=2, max=7
                ),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=8, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([3, 6]), coordinate_data=cd)

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=7, latency=5)

        cd.set_measurement(Coordinate([3, 6]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([2, 7]), rcm1)

        new_coord = n.determine_new_home()
        self.assertEqual(new_coord, Coordinate([2, 7]))

    def test_all_same_throughputs(self):
        """
        Test that when all the coordinates in the neighborhood has the
        same throughputs, the step vector is zero and new coordinate is
        same as the home coordinate.
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )

        nc = NeighborhoodConfig(dims, radius=3, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]), coordinate_data=cd)

        rcm0 = self._construct_rcm(throughput=10, latency=5)
        rcm1 = self._construct_rcm(throughput=10, latency=5)
        rcm2 = self._construct_rcm(throughput=10, latency=5)
        rcm3 = self._construct_rcm(throughput=10, latency=5)

        cd.set_measurement(Coordinate([1, 1, 1]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([1, 0, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1, 0]), rcm2)
        cd.set_measurement(Coordinate([0, 0, 1]), rcm3)

        step_vector = n._get_step_vector()
        self.assertEqual(step_vector, Coordinate([0, 0, 0]))

        n.force_slow_mode()
        new_coord = n.determine_new_home()
        self.assertEqual(new_coord, Coordinate([1, 1, 1]))

    def test_translate_step_vector(self):
        """
        Test the functionality of translate_step_vector()

        Any dimension value that is less than translation_list[0] will get 0

        Any dimension value that is between translation_list[i] and translation_list[i+1]
        will get i+1
        """

        # Test generic
        self._test_translate_step_vector_helper(
            input_vector=[0.35, 0.25, 0.15, 0.05],
            translation_list=[0.1, 0.2, 0.3],
            expected_vector=[3, 2, 1, 0],
        )

        # Test negative
        self._test_translate_step_vector_helper(
            input_vector=[-0.35, -0.25, -0.15, -0.05],
            translation_list=[0.1, 0.2, 0.3],
            expected_vector=[-3, -2, -1, 0],
        )

        # Test boundary case. Exact match rounds down
        self._test_translate_step_vector_helper(
            input_vector=[0.099, 0.100, 0.101],
            translation_list=[0.1],
            expected_vector=[0, 0, 1],
        )

    def _test_translate_step_vector_helper(
        self, input_vector, translation_list, expected_vector
    ):
        dims = SearchDimensions()
        dims.add_dimensions(
            0, [SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR)]
        )
        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([0]), coordinate_data=cd)

        input_coord = Coordinate(input_vector)
        expected_coord = Coordinate(expected_vector)
        output_coord = n._translate_step_vector(input_coord, translation_list)

        self.assertEqual(expected_coord, output_coord)

    def _test_calculate_step_vector_from_vectors_and_weights(self):
        """
        Test the functionality of __test_calculate_step_vector_from_vectors_and_weights()
        """
        dims = SearchDimensions()
        dims.add_dimensions(
            0,
            [
                SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            ],
        )
        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()
        n = Neighborhood(nc, home_coordinate=Coordinate([0, 0, 0]), coordinate_data=cd)

        vectors = [[0, 0, 1], [1, -3, 2], [-4, 2, 0], [3, 2, 6]]
        weights = [0.5, -3.0, -1, 0]

        # Sum up weights into vectors:
        # [0,0,0.5] + [-3,3,-3] + [1, -1, 0] + [0,0,0] = [-2, 2, -2.5]
        #
        # Sum up absolute value of vectors:
        # [0,0,1] + [1,3,2] + [4,2,0] + [3,2,6] = [8, 7, 9]
        #
        # Divide:
        # [-2, 2, -2.5] / [8, 7, 9] = [-2/8, 2/7, -2.5/9]

        step_vector = n._calculate_step_vector_from_vectors_and_weights(
            vectors, weights
        )
        expected_result = [-2 / 8, 2 / 7, -2.5 / 9]
        self.assertEqual(step_vector, expected_result)
