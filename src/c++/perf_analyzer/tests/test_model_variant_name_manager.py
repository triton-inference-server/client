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

from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.constants import DEFAULT_CONFIG_PARAMS
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.model.model_config_variant import ModelConfigVariant
from tests.common.test_utils import default_encode

from .common import test_result_collector as trc


class TestModelVariantNameManager(trc.TestResultCollector):
    def setUp(self):
        self._mvnm = ModelVariantNameManager()
        self._non_default_param_combo = {"foo": 1}

    def tearDown(self):
        patch.stopall()

    def test_default(self):
        """
        Check that default config is returned
        """
        name = self._mvnm.get_model_variant_name(
            "modelA", {"A": 1}, DEFAULT_CONFIG_PARAMS
        )

        self.assertEqual(name, (False, "modelA_config_default"))

    def test_basic(self):
        """
        If multiple unique model configs are passed in, the name will keep
        incrementing
        """
        a0 = self._mvnm.get_model_variant_name(
            "modelA", {"A": 1}, self._non_default_param_combo
        )
        a1 = self._mvnm.get_model_variant_name(
            "modelA", {"A": 2}, self._non_default_param_combo
        )
        a2 = self._mvnm.get_model_variant_name(
            "modelA", {"A": 4}, self._non_default_param_combo
        )

        self.assertEqual(a0, (False, "modelA_config_0"))
        self.assertEqual(a1, (False, "modelA_config_1"))
        self.assertEqual(a2, (False, "modelA_config_2"))

    def test_multiple_models(self):
        """
        The two models should have no impact on each other's naming or counts
        """

        a0 = self._mvnm.get_model_variant_name(
            "modelA", {"A": 1}, self._non_default_param_combo
        )
        b0 = self._mvnm.get_model_variant_name(
            "modelB", {"A": 1}, self._non_default_param_combo
        )
        a1 = self._mvnm.get_model_variant_name(
            "modelA", {"A": 2}, self._non_default_param_combo
        )
        b1 = self._mvnm.get_model_variant_name(
            "modelB", {"A": 2}, self._non_default_param_combo
        )

        self.assertEqual(a0, (False, "modelA_config_0"))
        self.assertEqual(a1, (False, "modelA_config_1"))
        self.assertEqual(b0, (False, "modelB_config_0"))
        self.assertEqual(b1, (False, "modelB_config_1"))

    def test_repeat(self):
        """
        Calling with the same model name/config/combo multiple times
        should result in the same config name being returned
        """

        a0 = self._mvnm.get_model_variant_name(
            "modelA", {"A": 1}, self._non_default_param_combo
        )
        a1 = self._mvnm.get_model_variant_name(
            "modelA", {"A": 1}, self._non_default_param_combo
        )

        self.assertEqual(a0, (False, "modelA_config_0"))
        self.assertEqual(a1, (True, "modelA_config_0"))

    def test_nested_dicts_matching(self):
        """
        Test matching with a model config consisting of matching nested dicts
        """
        model_config_A_0 = {"A": {"B": {"C": 1, "D": 2}}}
        model_config_A_1 = {"A": {"B": {"D": 2, "C": 1}}}

        a0 = self._mvnm.get_model_variant_name(
            "modelA", model_config_A_0, self._non_default_param_combo
        )

        a1 = self._mvnm.get_model_variant_name(
            "modelA", model_config_A_1, self._non_default_param_combo
        )

        self.assertEqual(a0, (False, "modelA_config_0"))
        self.assertEqual(a1, (True, "modelA_config_0"))

    def test_nested_dicts_different(self):
        """
        Test matching with a model config consisting of different nested dicts
        """
        model_config_A_0 = {"A": {"B": {"C": 1, "D": 2}}}
        model_config_A_1 = {"A": {"B": {"D": 1, "C": 2}}}

        a0 = self._mvnm.get_model_variant_name(
            "modelA", model_config_A_0, self._non_default_param_combo
        )

        a1 = self._mvnm.get_model_variant_name(
            "modelA", model_config_A_1, self._non_default_param_combo
        )

        self.assertEqual(a0, (False, "modelA_config_0"))
        self.assertEqual(a1, (False, "modelA_config_1"))

    def test_ensemble_default(self):
        """
        Test that a default ensemble config is returned
        """
        sub_configA = ModelConfigVariant(ModelConfig({}), "modelA_config_default")
        sub_configB = ModelConfigVariant(ModelConfig({}), "modelB_config_default")

        ensemble_key = ModelVariantNameManager.make_ensemble_composing_model_key(
            [sub_configA, sub_configB]
        )

        name = self._mvnm.get_ensemble_model_variant_name(
            "ensemble_model", ensemble_key
        )

        self.assertEqual(name, (False, "ensemble_model_config_default"))

    def test_ensemble_basic(self):
        """
        Test that we can increment the ensemble config numbers
        """
        sub_configA = ModelConfigVariant(ModelConfig({}), "modelA_config_0")
        sub_configB = ModelConfigVariant(ModelConfig({}), "modelB_config_0")

        ensemble_key = ModelVariantNameManager.make_ensemble_composing_model_key(
            [sub_configA, sub_configB]
        )

        name = self._mvnm.get_ensemble_model_variant_name(
            "ensemble_model", ensemble_key
        )

        self.assertEqual(name, (False, "ensemble_model_config_0"))

        sub_configB = ModelConfigVariant(ModelConfig({}), "modelB_config_1")

        ensemble_key = ModelVariantNameManager.make_ensemble_composing_model_key(
            [sub_configA, sub_configB]
        )

        name = self._mvnm.get_ensemble_model_variant_name(
            "ensemble_model", ensemble_key
        )

        self.assertEqual(name, (False, "ensemble_model_config_1"))

        sub_configA = ModelConfigVariant(ModelConfig({}), "modelA_config_1")

        ensemble_key = ModelVariantNameManager.make_ensemble_composing_model_key(
            [sub_configA, sub_configB]
        )

        name = self._mvnm.get_ensemble_model_variant_name(
            "ensemble_model", ensemble_key
        )

        self.assertEqual(name, (False, "ensemble_model_config_2"))

    def test_ensemble_repeat(self):
        """
        Calling with the same model name/ensemble key multiple times
        should result in the same config name being returned
        """
        sub_configA = ModelConfigVariant(ModelConfig({}), "modelA_config_0")
        sub_configB = ModelConfigVariant(ModelConfig({}), "modelB_config_0")

        ensemble_key = ModelVariantNameManager.make_ensemble_composing_model_key(
            [sub_configA, sub_configB]
        )

        name = self._mvnm.get_ensemble_model_variant_name(
            "ensemble_model", ensemble_key
        )

        self.assertEqual(name, (False, "ensemble_model_config_0"))

        name = self._mvnm.get_ensemble_model_variant_name(
            "ensemble_model", ensemble_key
        )

        self.assertEqual(name, (True, "ensemble_model_config_0"))

    def test_from_dict(self):
        """
        Restoring from a dict should see existing configs
        """
        _ = self._mvnm.get_model_variant_name(
            "modelA", {"A": 1}, self._non_default_param_combo
        )
        _ = self._mvnm.get_model_variant_name(
            "modelB", {"A": 1}, self._non_default_param_combo
        )
        _ = self._mvnm.get_model_variant_name(
            "modelA", {"A": 2}, self._non_default_param_combo
        )

        mvnm_dict = default_encode(self._mvnm)

        mvnm = ModelVariantNameManager.from_dict(mvnm_dict)

        self.assertEqual(mvnm._model_config_dicts, self._mvnm._model_config_dicts)
        self.assertEqual(mvnm._model_name_index, self._mvnm._model_name_index)

        a0 = mvnm.get_model_variant_name(
            "modelA", {"A": 1}, self._non_default_param_combo
        )
        b0 = mvnm.get_model_variant_name(
            "modelB", {"A": 1}, self._non_default_param_combo
        )
        a1 = mvnm.get_model_variant_name(
            "modelA", {"A": 2}, self._non_default_param_combo
        )
        b1 = mvnm.get_model_variant_name(
            "modelB", {"A": 2}, self._non_default_param_combo
        )

        self.assertEqual(a0, (True, "modelA_config_0"))
        self.assertEqual(a1, (True, "modelA_config_1"))
        self.assertEqual(b0, (True, "modelB_config_0"))
        self.assertEqual(b1, (False, "modelB_config_1"))


if __name__ == "__main__":
    unittest.main()
