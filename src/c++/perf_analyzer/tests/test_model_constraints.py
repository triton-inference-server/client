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

import unittest
from unittest.mock import patch

from model_analyzer.result.model_constraints import ModelConstraints

from .common import test_result_collector as trc


class TestModelConstraints(trc.TestResultCollector):
    def tearDown(self):
        patch.stopall()

    def test_to_dict(self):
        """
        Test that constraints object to dict
        """
        constraints_dict_1 = {
            "perf_latency_p99": {"max": 100},
            "gpu_used_memory": {"max": 20000},
        }
        constraints_dict_2 = {}

        constraints_obj_1 = ModelConstraints(constraints=constraints_dict_1)
        constraints_obj_2 = ModelConstraints(constraints=constraints_dict_2)

        self.assertEqual(constraints_obj_1.to_dict(), constraints_dict_1)
        self.assertEqual(constraints_obj_2.to_dict(), constraints_dict_2)

    def test_has_metric(self):
        """
        Test that constraints object has a given key
        """
        constraints_dict_1 = {
            "perf_latency_p99": {"max": 100},
            "gpu_used_memory": {"max": 20000},
        }
        constraints_dict_2 = {}

        constraints_obj_1 = ModelConstraints(constraints=constraints_dict_1)
        constraints_obj_2 = ModelConstraints(constraints=constraints_dict_2)

        self.assertTrue(constraints_obj_1.has_metric("perf_latency_p99"))
        self.assertTrue(constraints_obj_1.has_metric("gpu_used_memory"))
        self.assertFalse(constraints_obj_1.has_metric("perf_throughput"))
        self.assertFalse(constraints_obj_1.has_metric(""))
        self.assertFalse(constraints_obj_2.has_metric("perf_latency_p99"))
        self.assertFalse(constraints_obj_2.has_metric("gpu_used_memory"))

    def test_getitem(self):
        """
        Test that metrics are subscriptable from constraints object
        """
        constraints_dict = {
            "perf_latency_p99": {"max": 100},
            "gpu_used_memory": {"max": 20000},
        }

        constraints_obj = ModelConstraints(constraints=constraints_dict)

        self.assertEqual(constraints_obj["perf_latency_p99"], {"max": 100})
        self.assertEqual(constraints_obj["gpu_used_memory"], {"max": 20000})

    def test_bool(self):
        """
        Test that constraints are empty
        """
        constraints_dict_1 = {
            "perf_latency_p99": {"max": 100},
            "gpu_used_memory": {"max": 20000},
        }
        constraints_dict_2 = {}

        constraints_obj_1 = ModelConstraints(constraints=constraints_dict_1)
        constraints_obj_2 = ModelConstraints(constraints=constraints_dict_2)

        self.assertTrue(constraints_obj_1)
        self.assertFalse(constraints_obj_2)

    def test_eq(self):
        """
        Test that constraint objects are equal
        """
        constraints_dict_1 = {
            "perf_latency_p99": {"max": 100},
            "gpu_used_memory": {"max": 20000},
        }
        constraints_dict_2 = {
            "perf_latency_p99": {"max": 100},
            "gpu_used_memory": {"max": 20000},
        }
        constraints_dict_3 = {}

        constraints_obj_1 = ModelConstraints(constraints=constraints_dict_1)
        constraints_obj_2 = ModelConstraints(constraints=constraints_dict_2)
        constraints_obj_3 = ModelConstraints(constraints=constraints_dict_3)

        self.assertEqual(constraints_obj_1, constraints_obj_2)
        self.assertNotEqual(constraints_obj_3, constraints_obj_2)
        self.assertNotEqual(constraints_obj_1, constraints_obj_3)


if __name__ == "__main__":
    unittest.main()
