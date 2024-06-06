# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
from io import StringIO

import pytest
from genai_perf.export_data.console_exporter import ConsoleExporter
from genai_perf.export_data.data_exporter_factory import DataExporterType
from tests.test_data import triton_profile_data


class TestConsoleExporter:
    @pytest.fixture
    def mock_read_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This function will mock the open function for specific files.

        """

        original_open = open

        def custom_open(filename, *args, **kwargs):

            if str(filename) == "triton_profile_export.json":
                tmp_file = StringIO(json.dumps(triton_profile_data))
                return tmp_file
            else:
                return original_open(filename, *args, **kwargs)

        monkeypatch.setattr("builtins.open", custom_open)

    def test_pretty_print_output(self, capsys) -> None:
        config = {
            "type": DataExporterType.CONSOLE,
            "stats": stats,
        }
        exporter = ConsoleExporter(config)
        exporter.export()

        expected_content = (
            "                             LLM Metrics                              \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓\n"
            "┃                Statistic ┃  avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩\n"
            "│ Time to first token (ms) │ 2.00 │ 2.00 │ 3.00 │ 2.99 │ 2.90 │ 2.75 │\n"
            "│ Inter token latency (ms) │ 0.50 │ 0.00 │ 1.00 │ 0.99 │ 0.90 │ 0.75 │\n"
            "│     Request latency (ms) │ 3.00 │ 3.00 │ 4.00 │ 3.99 │ 3.90 │ 3.75 │\n"
            "│         Num output token │ 6.50 │ 6.00 │ 7.00 │ 6.99 │ 6.90 │ 6.75 │\n"
            "│          Num input token │ 7.50 │ 7.00 │ 8.00 │ 7.99 │ 7.90 │ 7.75 │\n"
            "└──────────────────────────┴──────┴──────┴──────┴──────┴──────┴──────┘\n"
            "Output token throughput (per sec): 123.00\n"
            "Request throughput (per sec): 456.00\n"
        )

        returned_data = capsys.readouterr().out

        assert returned_data == expected_content


stats = {
    "request_throughput": {"unit": "requests/sec", "avg": 456.0},
    "request_latency": {
        "unit": "ms",
        "avg": 3.0,
        "p99": 3.99,
        "p95": 3.95,
        "p90": 3.90,
        "p75": 3.75,
        "p50": 3.50,
        "p25": 3.25,
        "max": 4.0,
        "min": 3.0,
        "std": 3.50,
    },
    "time_to_first_token": {
        "unit": "ms",
        "avg": 2.0,
        "p99": 2.99,
        "p95": 2.95,
        "p90": 2.90,
        "p75": 2.75,
        "p50": 2.50,
        "p25": 2.25,
        "max": 3.00,
        "min": 2.00,
        "std": 2.50,
    },
    "inter_token_latency": {
        "unit": "ms",
        "avg": 0.50,
        "p99": 0.99,
        "p95": 0.95,
        "p90": 0.90,
        "p75": 0.75,
        "p50": 0.50,
        "p25": 0.25,
        "max": 1.00,
        "min": 0.00,
        "std": 0.50,
    },
    "output_token_throughput": {"unit": "tokens/sec", "avg": 123.0},
    "output_token_throughput_per_request": {
        "unit": "tokens/sec",
        "avg": 300.00,
        "p99": 300.00,
        "p95": 300.00,
        "p90": 300.00,
        "p75": 300.00,
        "p50": 300.00,
        "p25": 300.00,
        "max": 300.00,
        "min": 300.00,
        "std": 300.00,
    },
    "num_output_token": {
        "unit": "tokens",
        "avg": 6.5,
        "p99": 6.99,
        "p95": 6.95,
        "p90": 6.90,
        "p75": 6.75,
        "p50": 6.5,
        "p25": 6.25,
        "max": 7.0,
        "min": 6.0,
        "std": 6.5,
    },
    "num_input_token": {
        "unit": "tokens",
        "avg": 7.5,
        "p99": 7.99,
        "p95": 7.95,
        "p90": 7.90,
        "p75": 7.75,
        "p50": 7.5,
        "p25": 7.25,
        "max": 8.0,
        "min": 7.0,
        "std": 7.5,
    },
}
