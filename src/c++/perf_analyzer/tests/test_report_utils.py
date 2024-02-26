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

from model_analyzer.reports.report_utils import truncate_model_config_name

from .common import test_result_collector as trc


class TestReportUtils(trc.TestResultCollector):
    """
    Tests the report_utils functions
    """

    def tearDown(self):
        patch.stopall()

    def test_truncate_model_config_name(self):
        """
        Test the behavior of the truncate_longer_model_config_name function.
        """

        # Test: Shorter model config name (below 35 characters threshold)
        model_config_name = "ensemble_model_23"
        result = truncate_model_config_name(model_config_name)
        self.assertEqual(model_config_name, result)

        # Test: Model config name ends with 'config_#'
        model_config_name = "long_pytorch_platform_handler_config_10"
        result = truncate_model_config_name(model_config_name)
        self.assertEqual(result, "long_pytorch_platform_h...config_10")

        # Test: Model config name ends with 'config_default'.
        model_config_name = "long_pytorch_platform_handler_config_default"
        result = truncate_model_config_name(model_config_name)
        self.assertEqual(result, "long_pytorch_platf...config_default")

        # Test: Model config name includes the "config" keyword in the model name
        model_config_name = "long_config_pytorch_platform_handler_config_128"
        result = truncate_model_config_name(model_config_name)
        self.assertEqual(result, "long_config_pytorch_pl...config_128")


if __name__ == "__main__":
    unittest.main()
