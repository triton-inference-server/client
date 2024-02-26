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

import time
import unittest
from unittest.mock import patch

from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.monitor.dcgm.dcgm_monitor import DCGMMonitor
from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_utilization import GPUUtilization

from .common import test_result_collector as trc
from .mocks.mock_dcgm import MockDCGM
from .mocks.mock_dcgm_agent import TEST_PCI_BUS_ID, TEST_UUID
from .mocks.mock_dcgm_field_group_watcher import TEST_RECORD_VALUE
from .mocks.mock_numba import MockNumba

TEST_DEVICE_NAME = "TEST_DEVICE_NAME"
TEST_DEVICE_ID = 0


class TestDCGMMonitor(trc.TestResultCollector):
    def setUp(self):
        self.mock_dcgm = MockDCGM()
        self.mock_numba = MockNumba(
            mock_paths=[
                "model_analyzer.device.gpu_device_factory",
            ]
        )
        self.mock_dcgm.start()
        self.mock_numba.start()

        self._gpus = [
            GPUDevice(TEST_DEVICE_NAME, TEST_DEVICE_ID, TEST_PCI_BUS_ID, TEST_UUID)
        ]

    def test_record_memory(self):
        # One measurement every 0.01 seconds
        frequency = 1
        monitoring_time = 0.1
        metrics = [GPUUsedMemory, GPUFreeMemory]
        dcgm_monitor = DCGMMonitor(self._gpus, frequency, metrics)
        dcgm_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcgm_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            self.assertEqual(record.value(), TEST_RECORD_VALUE)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreater(len(records), 0)
        self.assertGreaterEqual(
            records[-1].timestamp() - records[0].timestamp(), monitoring_time
        )

        with self.assertRaises(TritonModelAnalyzerException):
            dcgm_monitor.stop_recording_metrics()

        dcgm_monitor.destroy()

        metrics = ["UndefinedTag"]
        with self.assertRaises(TritonModelAnalyzerException):
            DCGMMonitor(self._gpus, frequency, metrics)

    def test_record_power(self):
        # One measurement every 0.01 seconds
        frequency = 1
        monitoring_time = 1.1
        metrics = [GPUPowerUsage]
        dcgm_monitor = DCGMMonitor(self._gpus, frequency, metrics)
        dcgm_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcgm_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            self.assertEqual(record.value(), TEST_RECORD_VALUE)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreater(len(records), 0)
        self.assertGreaterEqual(
            records[-1].timestamp() - records[0].timestamp(), monitoring_time
        )

        dcgm_monitor.destroy()

    def test_record_utilization(self):
        # One measurement every 0.01 seconds
        frequency = 1
        monitoring_time = 1.1
        metrics = [GPUUtilization]
        dcgm_monitor = DCGMMonitor(self._gpus, frequency, metrics)
        dcgm_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcgm_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            self.assertLessEqual(record.value(), 100)
            self.assertEqual(record.value(), TEST_RECORD_VALUE)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreater(len(records), 0)
        self.assertGreaterEqual(
            records[-1].timestamp() - records[0].timestamp(), monitoring_time
        )

        dcgm_monitor.destroy()

    def test_immediate_start_stop(self):
        frequency = 1
        metrics = [GPUUsedMemory, GPUFreeMemory]
        dcgm_monitor = DCGMMonitor(self._gpus, frequency, metrics)
        dcgm_monitor.start_recording_metrics()
        dcgm_monitor.stop_recording_metrics()
        dcgm_monitor.destroy()

    def tearDown(self):
        patch.stopall()
        self.mock_dcgm.stop()
        self.mock_numba.stop()


if __name__ == "__main__":
    unittest.main()
