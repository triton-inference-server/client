#!/usr/bin/env python3

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.monitor.remote_monitor import RemoteMonitor
from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_utilization import GPUUtilization
from tests.mocks.mock_requests import MockRequests

from .common import test_result_collector as trc

TEST_DEVICE_NAME = "TEST_DEVICE_NAME"
TEST_DEVICE_ID = 0
TEST_METRICS_URL = "localhost:8002"

TEST_POWER_USAGE = 20
TEST_GPU_MEMORY_USAGE = 400000000
TEST_GPU_UTILIZATION = 0.5
TEST_TOTAL_BYTES = 1000000000

TEST_METRICS_RESPONSE = bytes(
    "# HELP nv_inference_request_success Number of successful"
    "inference requests, all batch sizes\n# TYPE nv_inference_request_success counter\n# "
    "HELP nv_inference_request_failure Number of failed inference requests, all batch sizes\n# "
    "TYPE nv_inference_request_failure counter\n# HELP nv_inference_count Number of inferences "
    "performed\n# TYPE nv_inference_count counter\n# HELP nv_inference_exec_count Number of model"
    "executions performed\n# TYPE nv_inference_exec_count counter\n# "
    "HELP nv_inference_request_duration_us Cumulative inference request duration in microseconds\n# "
    "TYPE nv_inference_request_duration_us counter\n# HELP nv_inference_queue_duration_us Cumulative "
    "inference queuing duration in microseconds\n# TYPE nv_inference_queue_duration_us counter\n# HELP "
    "nv_inference_compute_input_duration_us Cumulative compute input duration in microseconds\n# TYPE "
    "nv_inference_compute_input_duration_us counter\n# HELP nv_inference_compute_infer_duration_us Cumulative "
    "compute inference duration in microseconds\n# TYPE nv_inference_compute_infer_duration_us counter\n# HELP "
    "nv_inference_compute_output_duration_us Cumulative inference compute output duration in microseconds\n# TYPE "
    "nv_inference_compute_output_duration_us counter\n# HELP nv_gpu_utilization GPU utilization rate [0.0 - 1.0)\n# "
    'TYPE nv_gpu_utilization gauge\nnv_gpu_utilization{gpu_uuid="GPU-e35ba3d2-6eef-2bb9-e35c-6ef6eada4f11"} '
    f"{TEST_GPU_UTILIZATION}\n# HELP nv_gpu_memory_total_bytes GPU total memory, in bytes\n# TYPE nv_gpu_memory_total_bytes "
    'gauge\nnv_gpu_memory_total_bytes{gpu_uuid="GPU-e35ba3d2-6eef-2bb9-e35c-6ef6eada4f11"} '
    f"{TEST_TOTAL_BYTES}\n# HELP nv_gpu_memory_used_bytes GPU used memory, in bytes\n# TYPE "
    'nv_gpu_memory_used_bytes gauge\nnv_gpu_memory_used_bytes{gpu_uuid="GPU-e35ba3d2-6eef-2bb9-e35c-6ef6eada4f11"} '
    f"{TEST_GPU_MEMORY_USAGE}\n# HELP nv_gpu_power_usage GPU power usage in watts\n# TYPE nv_gpu_power_usage "
    'gauge\nnv_gpu_power_usage{gpu_uuid="GPU-e35ba3d2-6eef-2bb9-e35c-6ef6eada4f11"} '
    f"{TEST_POWER_USAGE}\n# HELP nv_gpu_power_limit GPU power management limit in watts\n# TYPE "
    'nv_gpu_power_limit gauge\nnv_gpu_power_limit{gpu_uuid="GPU-e35ba3d2-6eef-2bb9-e35c-6ef6eada4f11"} 280.000000\n# '
    "HELP nv_energy_consumption GPU energy consumption in joules since the Triton Server started\n# TYPE nv_energy_consumption "
    'counter\nnv_energy_consumption{gpu_uuid="GPU-e35ba3d2-6eef-2bb9-e35c-6ef6eada4f11"} 1474.042000\n',
    encoding="ascii",
)


class TestRemoteMonitor(trc.TestResultCollector):
    def setUp(self):
        self.mock_requests = MockRequests(
            mock_paths=["model_analyzer.monitor.remote_monitor"]
        )
        self.mock_requests.start()
        self.mock_requests.set_get_request_response(TEST_METRICS_RESPONSE)

    def test_record_memory(self):
        # One measurement every 0.1 seconds
        frequency = 0.1
        monitoring_time = 0.1
        metrics = [GPUUsedMemory, GPUFreeMemory]
        gpu_monitor = RemoteMonitor(TEST_METRICS_URL, frequency, metrics)
        gpu_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = gpu_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            if isinstance(record, GPUFreeMemory):
                self.assertEqual(
                    record.value(), (TEST_TOTAL_BYTES - TEST_GPU_MEMORY_USAGE) // 1e6
                )
            else:
                self.assertEqual(record.value(), TEST_GPU_MEMORY_USAGE // 1e6)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreater(len(records), 0)

        with self.assertRaises(TritonModelAnalyzerException):
            gpu_monitor.stop_recording_metrics()

        gpu_monitor.destroy()

        metrics = ["UndefinedTag"]
        with self.assertRaises(TritonModelAnalyzerException):
            RemoteMonitor(TEST_METRICS_URL, frequency, metrics)

    def test_record_power(self):
        # One measurement every 0.01 seconds
        frequency = 0.1
        monitoring_time = 0.1
        metrics = [GPUPowerUsage]
        gpu_monitor = RemoteMonitor(TEST_METRICS_URL, frequency, metrics)
        gpu_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = gpu_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            self.assertEqual(record.value(), TEST_POWER_USAGE)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreater(len(records), 0)

        gpu_monitor.destroy()

    def test_record_utilization(self):
        # One measurement every 0.01 seconds
        frequency = 0.1
        monitoring_time = 0.1
        metrics = [GPUUtilization]
        gpu_monitor = RemoteMonitor(TEST_METRICS_URL, frequency, metrics)
        gpu_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = gpu_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            self.assertLessEqual(record.value(), 100)
            self.assertEqual(record.value(), TEST_GPU_UTILIZATION * 100)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreater(len(records), 0)

        gpu_monitor.destroy()

    def test_immediate_start_stop(self):
        frequency = 1
        metrics = [GPUUsedMemory, GPUFreeMemory]
        gpu_monitor = RemoteMonitor(TEST_METRICS_URL, frequency, metrics)
        gpu_monitor.start_recording_metrics()
        gpu_monitor.stop_recording_metrics()
        gpu_monitor.destroy()

    def tearDown(self):
        patch.stopall()
        self.mock_requests.stop()


if __name__ == "__main__":
    unittest.main()
