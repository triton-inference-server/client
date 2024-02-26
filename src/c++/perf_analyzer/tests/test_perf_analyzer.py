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

import unittest
from unittest.mock import MagicMock, mock_open, patch

from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import (
    MEASUREMENT_REQUEST_COUNT_STEP,
    MEASUREMENT_WINDOW_STEP,
    PERF_ANALYZER_MEASUREMENT_WINDOW,
    PERF_ANALYZER_MINIMUM_REQUEST_COUNT,
)
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_utilization import GPUUtilization
from model_analyzer.record.types.perf_client_response_wait import PerfClientResponseWait
from model_analyzer.record.types.perf_client_send_recv import PerfClientSendRecv
from model_analyzer.record.types.perf_latency_avg import PerfLatencyAvg
from model_analyzer.record.types.perf_latency_p90 import PerfLatencyP90
from model_analyzer.record.types.perf_latency_p95 import PerfLatencyP95
from model_analyzer.record.types.perf_latency_p99 import PerfLatencyP99
from model_analyzer.record.types.perf_server_compute_infer import PerfServerComputeInfer
from model_analyzer.record.types.perf_server_compute_input import PerfServerComputeInput
from model_analyzer.record.types.perf_server_compute_output import (
    PerfServerComputeOutput,
)
from model_analyzer.record.types.perf_server_queue import PerfServerQueue
from model_analyzer.record.types.perf_throughput import PerfThroughput
from model_analyzer.triton.client.client_factory import TritonClientFactory
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory

from .common import test_result_collector as trc
from .mocks.mock_client import MockTritonClientMethods
from .mocks.mock_perf_analyzer import MockPerfAnalyzerMethods
from .mocks.mock_psutil import MockPSUtil
from .mocks.mock_server_local import MockServerLocalMethods

# Test Parameters
MODEL_LOCAL_PATH = "/model_analyzer/models"
MODEL_REPOSITORY_PATH = "/model_analyzer/models"
PERF_BIN_PATH = "perf_analyzer"
TRITON_LOCAL_BIN_PATH = "test_path"
TEST_MODEL_NAME = "test_model"
TEST_CONCURRENCY_RANGE = "1:16:2"
CONFIG_TEST_ARG = "sync"
TEST_GRPC_URL = "test_hostname:test_port"


class TestPerfAnalyzerMethods(trc.TestResultCollector):
    def setUp(self):
        # Mocks
        self.server_local_mock = MockServerLocalMethods()
        self.perf_mock = MockPerfAnalyzerMethods()
        self.client_mock = MockTritonClientMethods()
        self.mock_psutil = MockPSUtil()
        self.mock_psutil.start()
        self.server_local_mock.start()
        self.perf_mock.start()
        self.client_mock.start()

        # PerfAnalyzer config for all tests
        self.config = PerfAnalyzerConfig()
        self.config["model-name"] = TEST_MODEL_NAME
        self.config["measurement-interval"] = 1000
        self.config["measurement-request-count"] = 50

        self.run_config = RunConfig({})
        self.run_config.add_model_run_config(
            ModelRunConfig("fake_name", MagicMock(), self.config)
        )

        self.gpus = [GPUDevice("TEST_DEVICE_NAME", 0, "TEST_PCI_BUS_ID", "TEST_UUID")]

        # Triton Server
        self.server = None
        self.client = None

    def test_perf_analyzer_config(self):
        # Check config initializations
        self.assertIsNone(
            self.config[CONFIG_TEST_ARG],
            msg="Server config had unexpected initial" f" value for {CONFIG_TEST_ARG}",
        )

        # Set value
        self.config[CONFIG_TEST_ARG] = True

        # Test get again
        self.assertTrue(
            self.config[CONFIG_TEST_ARG], msg=f"{CONFIG_TEST_ARG} was not set"
        )

        # Try to set an unsupported config argument, expect failure
        with self.assertRaises(
            TritonModelAnalyzerException,
            msg="Expected exception on trying to set"
            "unsupported argument in perf_analyzer"
            "config",
        ):
            self.config["dummy"] = 1

        # set and get value for each subtype of arguments
        self.config["model-name"] = TEST_MODEL_NAME
        self.assertEqual(self.config["model-name"], TEST_MODEL_NAME)

        self.config["concurrency-range"] = TEST_CONCURRENCY_RANGE
        self.assertEqual(self.config["concurrency-range"], TEST_CONCURRENCY_RANGE)

        self.config["extra-verbose"] = True
        self.assertTrue(self.config["extra-verbose"])

    def test_perf_analyzer_boolean_args(self):
        """Test that only positive boolean args get added"""
        expected_cli_str = "-m test_model --measurement-interval=1000 --binary-search --measurement-request-count=50"

        self.config["async"] = "False"
        self.config["binary-search"] = "True"

        self.assertEqual(self.config.to_cli_string(), expected_cli_str)

    def test_perf_analyzer_additive_args(self):
        shape = ["name1:1,2,3", "name2:4,5,6"]
        expected_cli_str = "-m test_model --measurement-interval=1000 --shape=name1:1,2,3 --shape=name2:4,5,6 --measurement-request-count=50"

        self.config["shape"] = shape[:]

        self.assertEqual(self.config["shape"], shape)
        self.assertEqual(self.config.to_cli_string(), expected_cli_str)

        shape = "name1:1,2,3"
        expected_cli_str = "-m test_model --measurement-interval=1000 --shape=name1:1,2,3 --measurement-request-count=50"
        self.config["shape"] = shape

        self.assertEqual(self.config.to_cli_string(), expected_cli_str)

        shape = 5
        self.config["shape"] = shape

        with self.assertRaises(TritonModelAnalyzerException):
            self.config.to_cli_string()

    def test_perf_analyzer_ssl_args(self):
        """
        Verify that the generated cli string passed to PA matches our expected output.
        """
        ssl_grpc_use_ssl = "True"
        ssl_grpc_root_certifications_file = "a"
        ssl_grpc_private_key_file = "b"
        ssl_grpc_certificate_chain_file = "c"
        ssl_https_verify_peer = 1
        ssl_https_verify_host = 2
        ssl_https_ca_certificates_file = "d"
        ssl_https_client_certificate_type = "e"
        ssl_https_client_certificate_file = "f"
        ssl_https_private_key_type = "g"
        ssl_https_private_key_file = "h"

        expected_cli_str = (
            f"-m test_model --measurement-interval=1000 --measurement-request-count=50 --ssl-grpc-use-ssl "
            f"--ssl-grpc-root-certifications-file=a --ssl-grpc-private-key-file=b --ssl-grpc-certificate-chain-file=c "
            f"--ssl-https-verify-peer=1 --ssl-https-verify-host=2 --ssl-https-ca-certificates-file=d --ssl-https-client-certificate-type=e "
            f"--ssl-https-client-certificate-file=f --ssl-https-private-key-type=g --ssl-https-private-key-file=h"
        )

        self.config["ssl-grpc-use-ssl"] = ssl_grpc_use_ssl
        self.config[
            "ssl-grpc-root-certifications-file"
        ] = ssl_grpc_root_certifications_file
        self.config["ssl-grpc-private-key-file"] = ssl_grpc_private_key_file
        self.config["ssl-grpc-certificate-chain-file"] = ssl_grpc_certificate_chain_file
        self.config["ssl-https-verify-peer"] = ssl_https_verify_peer
        self.config["ssl-https-verify-host"] = ssl_https_verify_host
        self.config["ssl-https-ca-certificates-file"] = ssl_https_ca_certificates_file
        self.config[
            "ssl-https-client-certificate-type"
        ] = ssl_https_client_certificate_type
        self.config[
            "ssl-https-client-certificate-file"
        ] = ssl_https_client_certificate_file
        self.config["ssl-https-private-key-type"] = ssl_https_private_key_type
        self.config["ssl-https-private-key-file"] = ssl_https_private_key_file

        self.assertEqual(self.config["ssl-grpc-use-ssl"], ssl_grpc_use_ssl)
        self.assertEqual(
            self.config["ssl-grpc-root-certifications-file"],
            ssl_grpc_root_certifications_file,
        )
        self.assertEqual(
            self.config["ssl-grpc-private-key-file"], ssl_grpc_private_key_file
        )
        self.assertEqual(
            self.config["ssl-grpc-certificate-chain-file"],
            ssl_grpc_certificate_chain_file,
        )
        self.assertEqual(self.config["ssl-https-verify-peer"], ssl_https_verify_peer)
        self.assertEqual(self.config["ssl-https-verify-host"], ssl_https_verify_host)
        self.assertEqual(
            self.config["ssl-https-ca-certificates-file"],
            ssl_https_ca_certificates_file,
        )
        self.assertEqual(
            self.config["ssl-https-client-certificate-type"],
            ssl_https_client_certificate_type,
        )
        self.assertEqual(
            self.config["ssl-https-client-certificate-file"],
            ssl_https_client_certificate_file,
        )
        self.assertEqual(
            self.config["ssl-https-private-key-type"], ssl_https_private_key_type
        )
        self.assertEqual(
            self.config["ssl-https-private-key-file"], ssl_https_private_key_file
        )

        self.assertEqual(self.config.to_cli_string(), expected_cli_str)

        # Set ssl-grpc-use-ssl to False should remove it from the cli string
        ssl_grpc_use_ssl = "False"
        self.config["ssl-grpc-use-ssl"] = ssl_grpc_use_ssl
        self.assertEqual(self.config["ssl-grpc-use-ssl"], ssl_grpc_use_ssl)
        expected_cli_str = (
            f"-m test_model --measurement-interval=1000 --measurement-request-count=50 "
            f"--ssl-grpc-root-certifications-file=a --ssl-grpc-private-key-file=b --ssl-grpc-certificate-chain-file=c "
            f"--ssl-https-verify-peer=1 --ssl-https-verify-host=2 --ssl-https-ca-certificates-file=d --ssl-https-client-certificate-type=e "
            f"--ssl-https-client-certificate-file=f --ssl-https-private-key-type=g --ssl-https-private-key-file=h"
        )
        self.assertEqual(self.config.to_cli_string(), expected_cli_str)

    def test_run(self):
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Create server, client, PerfAnalyzer, and wait for server ready
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=self.gpus
        )

        perf_analyzer = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=self.run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )
        self.client = TritonClientFactory.create_grpc_client(server_url=TEST_GRPC_URL)
        self.server.start()
        self.client.wait_for_server_ready(num_retries=1)

        pa_csv_mock = """Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,"""
        pa_csv_mock += """Client Recv,p50 latency,p90 latency,p95 latency,p99 latency,Avg latency,request/response,response wait,"""
        pa_csv_mock += """Avg GPU Utilization,Avg GPU Power Usage,Max GPU Memory Usage,Total GPU Memory\n"""
        pa_csv_mock += """1,46.8,2,187,18,34,65,16,1,4600,4700,4800,4900,5000,3,314,"""
        pa_csv_mock += """GPU-aaf4fea0:0.809;GPU-aaf4fea1:0.901;GPU-aaf4fea2:0.745;,GPU-aaf4fea0:91.2;GPU-aaf4fea1:100;,GPU-aaf4fea0:1000000000;GPU-aaf4fea1:2000000000,GPU-aaf4fea0:1500000000;GPU-aaf4fea2:3000000000"""

        # Test avg latency parsing. GPU metric is ignored for get_perf_records()
        perf_metrics = [PerfLatencyAvg, GPUUtilization]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 5)

        # Test p90 latency parsing
        perf_metrics = [PerfLatencyP90]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 4.7)

        # Test p95 latency parsing
        perf_metrics = [PerfLatencyP95]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 4.8)

        # Test p99 latency parsing
        perf_metrics = [PerfLatencyP99]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 4.9)

        # Test throughput parsing
        perf_metrics = [PerfThroughput]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 46.8)

        # Test client response wait
        perf_metrics = [PerfClientResponseWait]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 0.314)

        # Test server queue
        perf_metrics = [PerfServerQueue]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 0.018)

        # Test server compute infer
        perf_metrics = [PerfServerComputeInfer]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 0.065)

        # Test server compute input
        perf_metrics = [PerfServerComputeInput]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 0.034)

        # Test server compute infer
        perf_metrics = [PerfServerComputeOutput]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        records = perf_analyzer.get_perf_records()
        self.assertEqual(len(records[TEST_MODEL_NAME]), 1)
        self.assertEqual(records[TEST_MODEL_NAME][0].value(), 0.016)

        # Test Avg GPU Utilizations. Perf metric is ignored for get_gpu_records()
        gpu_metrics = [GPUUtilization, PerfLatencyAvg]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(gpu_metrics)

        records = perf_analyzer.get_gpu_records()
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].device_uuid(), "GPU-aaf4fea0")
        self.assertEqual(records[0].value(), 80.9)
        self.assertEqual(records[1].device_uuid(), "GPU-aaf4fea1")
        self.assertEqual(records[1].value(), 90.1)
        self.assertEqual(records[2].device_uuid(), "GPU-aaf4fea2")
        self.assertEqual(records[2].value(), 74.5)

        # Test GPU Power Usage
        gpu_metrics = [GPUPowerUsage]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(gpu_metrics)

        records = perf_analyzer.get_gpu_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].device_uuid(), "GPU-aaf4fea0")
        self.assertEqual(records[0].value(), 91.2)
        self.assertEqual(records[1].device_uuid(), "GPU-aaf4fea1")
        self.assertEqual(records[1].value(), 100)

        # Test GPU Memory Usage
        gpu_metrics = [GPUUsedMemory]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(gpu_metrics)

        records = perf_analyzer.get_gpu_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].device_uuid(), "GPU-aaf4fea0")
        self.assertEqual(records[0].value(), 1000)
        self.assertEqual(records[1].device_uuid(), "GPU-aaf4fea1")
        self.assertEqual(records[1].value(), 2000)

        # Test Free GPU Memory (Must be measured with GPUUsedMemory)
        # GPU a0 has 1500 total memory and 1000 used memory, so free == 500
        # GPU a1 has no value reported for total, so it is ignored
        # GPU a2 has no value reported for used, so it is ignored
        gpu_metrics = [GPUFreeMemory, GPUUsedMemory]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(gpu_metrics)

        records = perf_analyzer.get_gpu_records()
        self.assertEqual(len(records), 3)
        self.assertEqual(type(records[0]), GPUUsedMemory)
        self.assertEqual(type(records[1]), GPUUsedMemory)
        self.assertEqual(type(records[2]), GPUFreeMemory)
        self.assertEqual(records[2].device_uuid(), "GPU-aaf4fea0")
        self.assertEqual(records[2].value(), 1500 - 1000)

        # # Test parsing for subset
        perf_metrics = [
            PerfThroughput,
            PerfLatencyAvg,
            PerfLatencyP90,
            PerfLatencyP95,
            PerfLatencyP99,
            GPUUtilization,
            GPUPowerUsage,
        ]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(perf_metrics)

        perf_records = perf_analyzer.get_perf_records()
        gpu_records = perf_analyzer.get_gpu_records()

        self.assertEqual(len(perf_records[TEST_MODEL_NAME]), 5)
        # GPUPowerUsage has 2 devices and GPUUtilization has 3
        self.assertEqual(len(gpu_records), 5)

        # Test no exceptions are raised when nothing can be parsed
        pa_csv_empty = ""
        perf_metrics = [
            PerfThroughput,
            PerfClientSendRecv,
            PerfClientResponseWait,
            PerfServerQueue,
            PerfServerComputeInfer,
            PerfServerComputeInput,
            PerfServerComputeOutput,
            GPUFreeMemory,
        ]
        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_empty),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            self.assertFalse(perf_analyzer.run(perf_metrics))

        # Test case where PA returns blank values for some GPU metrics
        pa_csv_mock = """Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,"""
        pa_csv_mock += """Client Recv,p50 latency,p90 latency,p95 latency,p99 latency,Avg latency,request/response,response wait,"""
        pa_csv_mock += """Avg GPU Utilization,Avg GPU Power Usage,Max GPU Memory Usage,Total GPU Memory\n"""
        pa_csv_mock += """1,46.8,2,187,18,34,65,16,1,4600,4700,4800,4900,5000,3,314,"""
        pa_csv_mock += """,,,7:1500"""

        # Test Max GPU Memory
        gpu_metrics = [GPUUsedMemory]

        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            perf_analyzer.run(gpu_metrics)

        records = perf_analyzer.get_gpu_records()
        self.assertEqual(len(records), 0)

        # Test exception handling
        self.perf_mock.set_perf_analyzer_return_code(1)
        with patch(
            "model_analyzer.perf_analyzer.perf_analyzer.open",
            mock_open(read_data=pa_csv_mock),
        ), patch("model_analyzer.perf_analyzer.perf_analyzer.os.remove"):
            self.assertTrue(perf_analyzer.run(perf_metrics))
        self.server.stop()

    def test_measurement_interval_increase(self):
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Create server, client, PerfAnalyzer, and wait for server ready
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=self.gpus
        )
        perf_analyzer_config = PerfAnalyzerConfig()
        perf_analyzer_config["model-name"] = TEST_MODEL_NAME
        perf_analyzer_config["concurrency-range"] = TEST_CONCURRENCY_RANGE
        perf_analyzer_config["measurement-mode"] = "time_windows"
        perf_analyzer = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=self.run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )
        self.client = TritonClientFactory.create_grpc_client(server_url=TEST_GRPC_URL)
        self.server.start()

        # Test failure to stabilize for measurement windows
        self.client.wait_for_server_ready(num_retries=1)
        test_stabilize_output = "Please use a larger time window"
        self.perf_mock.set_perf_analyzer_result_string(test_stabilize_output)
        self.perf_mock.set_perf_analyzer_return_code(1)
        perf_metrics = [PerfThroughput, PerfLatencyP99]
        perf_analyzer.run(perf_metrics)
        self.assertEqual(self.perf_mock.get_perf_analyzer_popen_call_count(), 10)

    def test_measurement_request_count_increase(self):
        server_config = TritonServerConfig()
        server_config["model-repository"] = MODEL_REPOSITORY_PATH

        # Create server, client, PerfAnalyzer, and wait for server ready
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=self.gpus
        )
        perf_analyzer = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=self.run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )
        self.client = TritonClientFactory.create_grpc_client(server_url=TEST_GRPC_URL)
        self.server.start()

        # Test the timeout for count mode
        self.client.wait_for_server_ready(num_retries=1)
        test_both_output = "Please use a larger time window"
        self.perf_mock.set_perf_analyzer_result_string(test_both_output)
        self.perf_mock.set_perf_analyzer_return_code(1)
        perf_metrics = [PerfThroughput, PerfLatencyP99]
        perf_analyzer.run(perf_metrics)
        self.assertEqual(self.perf_mock.get_perf_analyzer_popen_call_count(), 10)

    def test_is_multi_model(self):
        """
        Test the functionality of the _is_multi_model() function

        If the provided run_config only has one ModelRunConfig, then is_multi_model is false. Any
        more than one ModelRunConfig and it should return true
        """
        run_config = RunConfig({})
        run_config.add_model_run_config(
            ModelRunConfig(MagicMock(), MagicMock(), MagicMock())
        )

        pa1 = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )
        self.assertFalse(pa1._is_multi_model())

        run_config.add_model_run_config(
            ModelRunConfig(MagicMock(), MagicMock(), MagicMock())
        )
        pa2 = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )
        self.assertTrue(pa2._is_multi_model())

    def test_get_cmd_single_model(self):
        """
        Test the functionality of _get_cmd() for single model
        """
        pa = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=self.run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )

        expected_cmd = [
            "perf_analyzer",
            "-m",
            "test_model",
            "--measurement-interval",
            "1000",
            "--measurement-request-count",
            "50",
        ]
        self.assertEqual(pa._get_cmd(), expected_cmd)

    def test_get_cmd_multi_model(self):
        """
        Test the functionality of _get_cmd() for multi model
        """
        pac1 = PerfAnalyzerConfig()
        pac1["model-name"] = "MultiModel1"
        pac1["measurement-interval"] = 1000
        pac1["measurement-request-count"] = 50

        pac2 = PerfAnalyzerConfig()
        pac2["model-name"] = "MultiModel2"
        pac2["batch-size"] = 16
        pac2["concurrency-range"] = 1024

        run_config = RunConfig({})
        run_config.add_model_run_config(ModelRunConfig(MagicMock(), MagicMock(), pac1))
        run_config.add_model_run_config(ModelRunConfig(MagicMock(), MagicMock(), pac2))

        pa = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )

        # yapf: disable
        expected_cmd = [
            'mpiexec', '--allow-run-as-root', '--tag-output',
            '-n', '1', 'perf_analyzer', '--enable-mpi',
                '-m', 'MultiModel1',
                '--measurement-interval', '1000',
                '--measurement-request-count', '50',
            ':', '-n', '1', 'perf_analyzer', '--enable-mpi',
                '-m', 'MultiModel2',
                '-b', '16',
                '--concurrency-range', '1024'
        ]
        # yapf: enable

        self.assertEqual(pa._get_cmd(), expected_cmd)

    def test_split_output_per_rank_for_single_model(self):
        """
        Test functionality of _get_output_per_rank() for single-model
        """

        output = """
[1,0]<stdout>:*** Measurement Settings ***
[1,1]<stdout>:*** Measurement Settings ***
[1,1]<stdout>:Request concurrency: 2
[1,0]<stdout>:Request concurrency: 1
"""

        # Expect no change for single-model
        expected_result = [output]

        run_config = RunConfig({})
        run_config.add_model_run_config(
            ModelRunConfig(MagicMock(), MagicMock(), MagicMock())
        )

        pa = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )

        pa._output = output
        result = pa._split_output_per_rank()
        self.assertEqual(result, expected_result)

    def test_split_output_per_rank_for_multi_model(self):
        """
        Test functionality of _get_output_per_rank() for multi-model
        """

        output = """
[1,0]<stdout>:*** Measurement Settings ***
[1,1]<stdout>:*** Measurement Settings ***
[1,1]<stdout>:Request concurrency: 2
[1,0]<stdout>:Request concurrency: 1
"""

        expected_result = [
            """[1,0]<stdout>:*** Measurement Settings ***
[1,0]<stdout>:Request concurrency: 1
""",
            """[1,1]<stdout>:*** Measurement Settings ***
[1,1]<stdout>:Request concurrency: 2
""",
        ]

        run_config = RunConfig({})
        run_config.add_model_run_config(
            ModelRunConfig(MagicMock(), MagicMock(), MagicMock())
        )
        run_config.add_model_run_config(
            ModelRunConfig(MagicMock(), MagicMock(), MagicMock())
        )

        pa = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )

        pa._output = output
        result = pa._split_output_per_rank()
        self.assertEqual(result, expected_result)

    def test_auto_adjust_parameters(self):
        """
        Test function _auto_adjust_parameters()

        In the case below:
            - Model0 fails to be stable in time_windows and should have measurement-interval increased
            - Model1 is fine and should have no changes
            - Model2 fails to be stable in count_windows and should have measurement-request-count increased
        """
        pac0 = PerfAnalyzerConfig()
        pac0["model-name"] = "MultiModel0"
        pac0["measurement-mode"] = "time_windows"

        pac1 = PerfAnalyzerConfig()
        pac1["model-name"] = "MultiModel1"
        pac1["measurement-mode"] = "time_windows"

        pac2 = PerfAnalyzerConfig()
        pac2["model-name"] = "MultiModel2"
        pac2["measurement-mode"] = "count_windows"

        run_config = RunConfig({})
        run_config.add_model_run_config(ModelRunConfig(MagicMock(), MagicMock(), pac0))
        run_config.add_model_run_config(ModelRunConfig(MagicMock(), MagicMock(), pac1))
        run_config.add_model_run_config(ModelRunConfig(MagicMock(), MagicMock(), pac2))

        pa = PerfAnalyzer(
            path=PERF_BIN_PATH,
            config=run_config,
            max_retries=10,
            timeout=100,
            max_cpu_util=50,
        )

        pa._output = """
[1,0]<stdout>:*** Measurement Settings ***
[1,1]<stdout>:*** Measurement Settings ***
[1,2]<stdout>:*** Measurement Settings ***
[1,2]<stdout>:Failed to obtain stable measurement
[1,0]<stdout>:Failed to obtain stable measurement
[1,1]<stdout>:Success
        """

        pa._auto_adjust_parameters(MagicMock())

        expected_measurement_interval = (
            PERF_ANALYZER_MEASUREMENT_WINDOW + MEASUREMENT_WINDOW_STEP
        )
        expected_request_count = (
            PERF_ANALYZER_MINIMUM_REQUEST_COUNT + MEASUREMENT_REQUEST_COUNT_STEP
        )

        self.assertEqual(
            expected_measurement_interval,
            pa._config.model_run_configs()[0].perf_config()["measurement-interval"],
        )
        self.assertEqual(
            None,
            pa._config.model_run_configs()[0].perf_config()[
                "measurement-request-count"
            ],
        )
        self.assertEqual(
            None,
            pa._config.model_run_configs()[1].perf_config()["measurement-interval"],
        )
        self.assertEqual(
            None,
            pa._config.model_run_configs()[1].perf_config()[
                "measurement-request-count"
            ],
        )
        self.assertEqual(
            None,
            pa._config.model_run_configs()[2].perf_config()["measurement-interval"],
        )
        self.assertEqual(
            expected_request_count,
            pa._config.model_run_configs()[2].perf_config()[
                "measurement-request-count"
            ],
        )

    def tearDown(self):
        # In case test raises exception
        if self.server is not None:
            self.server.stop()

        # Stop mocking
        self.server_local_mock.stop()
        self.perf_mock.stop()
        self.client_mock.stop()
        self.mock_psutil.stop()
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
