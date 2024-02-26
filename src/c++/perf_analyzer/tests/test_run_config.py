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
from unittest.mock import MagicMock, patch

from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.model.model_config_variant import ModelConfigVariant

from .common import test_result_collector as trc


class TestRunConfig(trc.TestResultCollector):
    def tearDown(self):
        patch.stopall()

    def test_triton_env(self):
        """
        Test triton env initialized correctly and returns correctly from function
        """
        fake_env = {"a": 5, "b": {"c": 7}}
        rc = RunConfig(fake_env)
        self.assertEqual(rc.triton_environment(), fake_env)

    def test_model_run_configs(self):
        """
        Test adding and getting ModelRunConfigs
        """
        mrc1 = ModelRunConfig("model1", MagicMock(), MagicMock())
        mrc2 = ModelRunConfig("model2", MagicMock(), MagicMock())
        rc = RunConfig({})
        rc.add_model_run_config(mrc1)
        rc.add_model_run_config(mrc2)

        mrc_out = rc.model_run_configs()

        self.assertEqual(mrc_out[0], mrc1)
        self.assertEqual(mrc_out[1], mrc2)

    def test_representation(self):
        """
        Test that representation() is just a string join of member ModelRunConfig's representations
        """
        pc1 = PerfAnalyzerConfig()
        pc1.update_config({"model-name": "TestModel1"})
        pc2 = PerfAnalyzerConfig()
        pc2.update_config({"model-name": "TestModel2"})
        mrc1 = ModelRunConfig(
            "model1", ModelConfigVariant(MagicMock(), "model1_config_0"), pc1
        )
        mrc2 = ModelRunConfig(
            "model2", ModelConfigVariant(MagicMock(), "model2_config_0"), pc2
        )
        rc = RunConfig({})
        rc.add_model_run_config(mrc1)
        rc.add_model_run_config(mrc2)

        expected_representation = (
            "model1_config_0 "
            + pc1.representation()
            + "model2_config_0 "
            + pc2.representation()
        )
        self.assertEqual(rc.representation(), expected_representation)

    def test_representation_mrc_removal(self):
        """
        Test that representation removes measurement request count
        """
        pc = PerfAnalyzerConfig()
        pc.update_config({"model-name": "TestModel1"})
        pc.update_config({"measurement-request-count": "500"})

        mrc = ModelRunConfig(
            "model1", ModelConfigVariant(MagicMock(), "model1_config_0"), pc
        )

        expected_representation = "model1_config_0 -m TestModel1"
        self.assertEqual(mrc.representation(), expected_representation)

    def test_cpu_only(self):
        """
        Test that cpu_only() is only true if all ModelConfigs are cpu_only()
        """
        cpu_only_true_mc = ModelConfig({})
        cpu_only_true_mcv = ModelConfigVariant(
            cpu_only_true_mc, "cpu_only_true_config_default", True
        )

        cpu_only_false_mc = ModelConfig({})
        cpu_only_false_mcv = ModelConfigVariant(
            cpu_only_false_mc, "cpu_only_false_config_default", False
        )

        mrc1 = ModelRunConfig("model1", cpu_only_true_mcv, MagicMock())
        mrc2 = ModelRunConfig("model2", cpu_only_false_mcv, MagicMock())

        rc = RunConfig({})
        rc.add_model_run_config(mrc1)
        self.assertTrue(rc.cpu_only())

        rc.add_model_run_config(mrc2)
        self.assertFalse(rc.cpu_only())

    def test_mrc_with_illegal_combinations(self):
        """
        Test ModelRunConfig with illegal combinations
        """
        mc = ModelConfig({})
        pc = PerfAnalyzerConfig()

        # Valid client batch-size and invalid model preferred_batch_size
        pc["batch-size"] = 2
        mc.set_config(
            {
                "name": "test_model",
                "max_batch_size": 4,
                "dynamic_batching": {"preferred_batch_size": [4, 8]},
            }
        )
        mcv = ModelConfigVariant(mc, "test_model_config_default")
        mrc = ModelRunConfig("modelA", mcv, pc)
        self.assertFalse(mrc.is_legal_combination())

        # Invalid client batch-size and invalid model preferred_batch_size
        pc["batch-size"] = 8
        mrc = ModelRunConfig("modelB", mcv, pc)
        self.assertFalse(mrc.is_legal_combination())

        # Invalid client batch-size and valid model preferred_batch_size
        mc.set_config(
            {"max_batch_size": 4, "dynamic_batching": {"preferred_batch_size": [2]}}
        )
        mrc = ModelRunConfig("modelC", mcv, pc)
        self.assertFalse(mrc.is_legal_combination())

    def test_mrc_with_legal_combinations(self):
        """
        Test ModelRunConfig with legal combinations
        """
        mc = ModelConfig({})
        pc = PerfAnalyzerConfig()

        # Valid client batch-size and no model preferred_batch_size
        pc["batch-size"] = 2
        mc.set_config({"max_batch_size": 8})
        mcv = ModelConfigVariant(mc, "test_model_config_default")
        mrc = ModelRunConfig("modelA", mcv, pc)
        self.assertTrue(mrc.is_legal_combination())

        # Valid client batch-size and valid model preferred_batch_size
        mc.set_config(
            {"max_batch_size": 8, "dynamic_batching": {"preferred_batch_size": [4]}}
        )
        mrc = ModelRunConfig("modelB", mcv, pc)
        self.assertTrue(mrc.is_legal_combination())

    def test_composing_mrc_with_illegal_combinations(self):
        """
        Test ModelRunConfig with illegal combinations in composing config
        """
        mc = ModelConfig({})
        pc = PerfAnalyzerConfig()
        composing_model_config_variants = [
            ModelConfigVariant(ModelConfig({}), "composing_model_A_config_default"),
            ModelConfigVariant(ModelConfig({}), "composing_model_B_config_default"),
        ]

        # Invalid client batch-size and valid model preferred_batch_size for composing_config[1]
        pc["batch-size"] = 2
        mc.set_config({"max_batch_size": 8, "name": "test_model"})
        mcv = ModelConfigVariant(mc, "test_model_config_default")
        mrc = ModelRunConfig("modelC", mcv, pc)

        composing_model_config_variants[0].model_config.set_config(
            {
                "name": "composing_model_A",
                "max_batch_size": 4,
                "dynamic_batching": {"preferred_batch_size": [2]},
            }
        )
        composing_model_config_variants[1].model_config.set_config(
            {
                "name": "composing_model_B",
                "max_batch_size": 4,
                "dynamic_batching": {"preferred_batch_size": [4, 8]},
            }
        )

        mrc.add_composing_model_config_variants(composing_model_config_variants)

        self.assertFalse(mrc.is_legal_combination())


if __name__ == "__main__":
    unittest.main()
