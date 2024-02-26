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

from model_analyzer.result.results import Results

from .common import test_result_collector as trc
from .common.test_utils import construct_run_config


class TestResults(trc.TestResultCollector):
    def setUp(self):
        self._construct_results()

    def tearDown(self):
        patch.stopall()

    def test_contains_model(self):
        """
        Test for the existence of expected models
        """
        self.assertTrue(self._result.contains_model("modelA"))
        self.assertTrue(self._result.contains_model("modelB"))
        self.assertFalse(self._result.contains_model("modelC"))

    def test_contains_model_variant(self):
        """
        Test for the existence of expected model variants in modelA
        """
        self.assertTrue(self._result.contains_model_variant("modelA", "model_config_0"))
        self.assertTrue(self._result.contains_model_variant("modelA", "model_config_1"))
        self.assertTrue(self._result.contains_model_variant("modelA", "model_config_2"))
        self.assertFalse(
            self._result.contains_model_variant("modelA", "model_config_3")
        )
        self.assertFalse(
            self._result.contains_model_variant("modelC", "model_config_0")
        )

    def test_get_list_of_models(self):
        """
        Test that the list of models is returned correctly
        """
        model_list = self._result.get_list_of_models()

        self.assertEqual(model_list, ["modelA", "modelB"])

    def test_get_list_of_model_config_measurement_tuples(self):
        """
        Test that the correct number of measurements is returned per model config
        """
        model_config_measurements_list = (
            self._result.get_list_of_model_config_measurement_tuples()
        )
        num_model_config_measurements = [
            len(mcm) for mcm in model_config_measurements_list
        ]

        self.assertEqual(num_model_config_measurements[0], 3)
        self.assertEqual(num_model_config_measurements[1], 2)

    def test_get_list_of_run_config_measurements(self):
        """
        Test that the measurements were correctly added to the dictionary
        """
        measurements = self._result.get_list_of_run_config_measurements()

        self.assertEqual(measurements, self._measurements_added)

    def test_get_model_measurements_dict(self):
        """
        Test that the measurements were correctly added to the model dictionaries
        """
        model_measurements = self._result.get_model_measurements_dict("modelA")

        for index, (run_config, measurements) in enumerate(model_measurements.values()):
            # There were 3 runs per model config. Make sure their values all match
            base_index = index * 3
            self.assertEqual(
                run_config.model_run_configs()[0]
                .model_config_variant()
                .model_config.get_config(),
                self._run_configs[base_index]
                .model_run_configs()[0]
                .model_config_variant()
                .model_config.get_config(),
            )
            self.assertEqual(
                run_config.model_run_configs()[0]
                .model_config_variant()
                .model_config.get_config(),
                self._run_configs[base_index + 1]
                .model_run_configs()[0]
                .model_config_variant()
                .model_config.get_config(),
            )
            self.assertEqual(
                run_config.model_run_configs()[0]
                .model_config_variant()
                .model_config.get_config(),
                self._run_configs[base_index + 2]
                .model_run_configs()[0]
                .model_config_variant()
                .model_config.get_config(),
            )
            self.assertEqual(measurements, self._measurements[index])

    def test_bad_get_model_measurements_dict(self):
        """
        Test that we return an empty dict and don't assert if the model
        doesn't exist in the measurements
        """
        model_measurements = self._result.get_model_measurements_dict("bad_model_name")
        self.assertEqual(model_measurements, {})

    def test_get_model_config_measurements_dict(self):
        """
        Test that the measurements were correctly added to the model config dictionaries
        """
        model_config_measurements = self._result.get_model_variants_measurements_dict(
            "modelA", "model_config_1"
        )

        self.assertEqual(model_config_measurements, self._measurements[1])

    def test_get_all_model_variant_measurements(self):
        """
        Test that the list of measurements were correctly generated for a
        given model + model_config combination
        """
        run_config, measurements = self._result.get_all_model_variant_measurements(
            "modelA", "model_config_1"
        )

        self.assertEqual(
            run_config.model_run_configs()[0].model_config(),
            self._run_configs[3]
            .model_run_configs()[0]
            .model_config_variant()
            .model_config,
        )
        self.assertEqual(measurements, list(self._measurements[1].values()))

    def _construct_results(self):
        """
        Creates an instance of the results class with two models, and
        various model configs. Treating the measurements as strings,
        which makes debugging easier
        """
        self._result = Results()

        self._run_configs = [
            construct_run_config("modelA", "model_config_0", "key_A"),
            construct_run_config("modelA", "model_config_0", "key_B"),
            construct_run_config("modelA", "model_config_0", "key_C"),
            construct_run_config("modelA", "model_config_1", "key_D"),
            construct_run_config("modelA", "model_config_1", "key_E"),
            construct_run_config("modelA", "model_config_1", "key_F"),
            construct_run_config("modelA", "model_config_2", "key_G"),
            construct_run_config("modelA", "model_config_2", "key_H"),
            construct_run_config("modelA", "model_config_2", "key_I"),
        ]

        self._measurements = []
        self._measurements.append(
            {
                "model_config_0 -m key_A": "1",
                "model_config_0 -m key_B": "2",
                "model_config_0 -m key_C": "3",
            }
        )
        self._measurements.append(
            {
                "model_config_1 -m key_D": "4",
                "model_config_1 -m key_E": "5",
                "model_config_1 -m key_F": "6",
            }
        )
        self._measurements.append(
            {
                "model_config_2 -m key_G": "7",
                "model_config_2 -m key_H": "8",
                "model_config_2 -m key_I": "9",
            }
        )

        self._result.add_run_config_measurement(self._run_configs[0], "1")
        self._result.add_run_config_measurement(self._run_configs[1], "2")
        self._result.add_run_config_measurement(self._run_configs[2], "3")

        self._result.add_run_config_measurement(self._run_configs[3], "4")
        self._result.add_run_config_measurement(self._run_configs[4], "5")
        self._result.add_run_config_measurement(self._run_configs[5], "6")

        self._result.add_run_config_measurement(self._run_configs[6], "7")
        self._result.add_run_config_measurement(self._run_configs[7], "8")
        self._result.add_run_config_measurement(self._run_configs[8], "9")

        model_run_config_0f = construct_run_config("modelB", "model_config_0", "key_F")
        model_run_config_0e = construct_run_config("modelB", "model_config_0", "key_E")
        model_run_config_0d = construct_run_config("modelB", "model_config_0", "key_D")
        self._result.add_run_config_measurement(model_run_config_0f, "6")
        self._result.add_run_config_measurement(model_run_config_0e, "5")
        self._result.add_run_config_measurement(model_run_config_0d, "4")

        model_run_config_1c = construct_run_config("modelB", "model_config_1", "key_C")
        model_run_config_1b = construct_run_config("modelB", "model_config_1", "key_B")
        model_run_config_1a = construct_run_config("modelB", "model_config_1", "key_A")
        self._result.add_run_config_measurement(model_run_config_1c, "3")
        self._result.add_run_config_measurement(model_run_config_1b, "2")
        self._result.add_run_config_measurement(model_run_config_1a, "1")

        self._measurements_added = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "6",
            "5",
            "4",
            "3",
            "2",
            "1",
        ]


if __name__ == "__main__":
    unittest.main()
