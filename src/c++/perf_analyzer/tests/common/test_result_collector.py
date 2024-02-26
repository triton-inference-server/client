#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import unittest


class TestResultCollector(unittest.TestCase):
    """
    TestResultCollector stores test result and prints it to stdout. In order
    to use this class, unit tests must inherit this class. Use
    `check_test_results` bash function from `common/util.sh` to verify the
    expected number of tests produced by this class
    """

    @classmethod
    def setResult(cls, total, errors, failures):
        cls.total, cls.errors, cls.failures = total, errors, failures

    @classmethod
    def tearDownClass(cls):
        # this method is called when all the unit tests in a class are
        # finished.
        json_res = {"total": cls.total, "errors": cls.errors, "failures": cls.failures}
        print(json.dumps(json_res))

    def run(self, result=None):
        # result argument stores the accumulative test results
        test_result = super().run(result)
        if test_result:
            total = test_result.testsRun
            errors = len(test_result.errors)
            failures = len(test_result.failures)
            self.setResult(total, errors, failures)
