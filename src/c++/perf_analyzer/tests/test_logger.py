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

# This test exercises all of the cli options.
# It does a basic test of the cli and help message.
# Then the rest of the testing uses an OptionStruct, which holds all
# of the data necessary to test a command, and feeds that to the
# CLI parser. The result of the CLI parsing is compared against the
# expected value for the CLI. Default values are also verified as well as
# values that are expected to cause failures.

import logging
import re
import sys
import unittest
from contextlib import contextmanager
from io import StringIO
from unittest.mock import patch

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.log_formatter import setup_logging

from .common import test_result_collector as trc

logger = logging.getLogger(LOGGER_NAME)


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestLogger(trc.TestResultCollector):
    def tearDown(self):
        patch.stopall()

    def test_info_normal(self):
        """Test expected format of logger.info"""
        with captured_output() as (out, err):
            setup_logging(verbose=False, quiet=False)
            logger.info("ABC")
            expected_output = "[Model Analyzer] ABC"
            output = out.getvalue().strip()
            self.assertEqual(output, expected_output)

    def test_info_quiet(self):
        """
        Test expected format of logger.info in quiet mode
        (Nothing should be printed)
        """
        with captured_output() as (out, err):
            setup_logging(verbose=False, quiet=True)
            logger.info("ABC")
            expected_output = ""
            output = out.getvalue().strip()
            self.assertEqual(output, expected_output)

    def test_info_verbose(self):
        """
        Test expected format of logger.info in verbose mode
        (Extra date info should be added to the start of the line)
        """
        with captured_output() as (out, err):
            setup_logging(verbose=True, quiet=False)
            logger.info("ABC")
            output = out.getvalue().strip()
            self.assertTrue(re.match("^\d\d:\d\d:\d\d \[Model Analyzer\] ABC$", output))

    def test_debug_normal(self):
        """
        Test expected format of logger.debug
        (Nothing should be printed since we aren't in verbose mode)
        """
        with captured_output() as (out, err):
            setup_logging(verbose=False, quiet=False)
            logger.debug("ABC")
            expected_output = ""
            output = out.getvalue().strip()
            self.assertEqual(output, expected_output)

    def test_debug_quiet(self):
        """
        Test expected format of logger.debug in quiet mode
        (Nothing should be printed since we aren't in verbose mode)
        """
        with captured_output() as (out, err):
            setup_logging(verbose=False, quiet=True)
            logger.debug("ABC")
            expected_output = ""
            output = out.getvalue().strip()
            self.assertEqual(output, expected_output)

    def test_debug_verbose(self):
        """
        Test expected format of logger.debug in verbose mode
        (Extra date info should be added to the start of the line)
        """
        with captured_output() as (out, err):
            setup_logging(verbose=True, quiet=False)
            logger.debug("ABC")
            output = out.getvalue().strip()
            self.assertTrue(
                re.match("^\d\d:\d\d:\d\d \[Model Analyzer\] DEBUG: ABC$", output)
            )

    def test_error_normal(self):
        """
        Test expected format of logger.error
        """
        with captured_output() as (out, err):
            setup_logging(verbose=False, quiet=False)
            logger.error("ABC")
            expected_output = "[Model Analyzer] ERROR: ABC"
            output = out.getvalue().strip()
            self.assertEqual(output, expected_output)

    def test_error_quiet(self):
        """
        Test expected format of logger.error in quiet mode
        (It will be printed despite verbose mode)
        """
        with captured_output() as (out, err):
            setup_logging(verbose=False, quiet=True)
            logger.error("ABC")
            expected_output = "[Model Analyzer] ERROR: ABC"
            output = out.getvalue().strip()
            self.assertEqual(output, expected_output)

    def test_error_verbose(self):
        """
        Test expected format of logger.error in verbose mode
        (Extra date info should be added to the start of the line)
        """
        with captured_output() as (out, err):
            setup_logging(verbose=True, quiet=False)
            logger.error("ABC")
            output = out.getvalue().strip()
            self.assertTrue(
                re.match("^\d\d:\d\d:\d\d \[Model Analyzer\] ERROR: ABC$", output)
            )

    def test_quiet_trumps_verbose(self):
        """
        Test that calling setup_logging with both verbose and quiet true
        will result in quiet mode
        """
        with captured_output() as (out, err):
            setup_logging(verbose=True, quiet=True)
            logger.info("ABC")
            expected_output = ""
            output = out.getvalue().strip()
            self.assertEqual(output, expected_output)


if __name__ == "__main__":
    unittest.main()
