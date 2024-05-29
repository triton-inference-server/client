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
import sys
from io import StringIO
from pathlib import Path

import pytest
from genai_perf.export_data.console_exporter import ConsoleExporter
from genai_perf.export_data.data_exporter_factory import DataExporterType
from genai_perf.llm_metrics import LLMProfileDataParser
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_tokenizer
from tests.test_data import triton_profile_data


class TestConsoleExporter:
    @pytest.fixture
    def mock_read_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This function will mock the open function for specific files:

        """

        original_open = open

        def custom_open(filename, *args, **kwargs):

            if str(filename) == "triton_profile_export.json":
                tmp_file = StringIO(json.dumps(triton_profile_data))
                return tmp_file
            else:
                return original_open(filename, *args, **kwargs)

        monkeypatch.setattr("builtins.open", custom_open)

    def test_pretty_print_output(self, mock_read_write: pytest.MonkeyPatch) -> None:
        tokenizer = get_tokenizer(DEFAULT_TOKENIZER)
        pd = LLMProfileDataParser(
            filename=Path("triton_profile_export.json"),
            tokenizer=tokenizer,
        )
        stat = pd.get_statistics(infer_mode="concurrency", load_level="10")

        capturedTable = StringIO()
        sys.stdout = capturedTable
        config = {
            "type": DataExporterType.CONSOLE,
            "stats": stat.stats_dict,
        }
        exporter = ConsoleExporter(config)
        exporter.export()
        sys.stdout = sys.__stdout__

        expected_content = (
            "                             LLM Metrics                              \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓\n"
            "┃                Statistic ┃  avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩\n"
            "│ Time to first token (ms) │ 2.00 │ 2.00 │ 2.00 │ 2.00 │ 2.00 │ 2.00 │\n"
            "│ Inter token latency (ms) │ 2.00 │ 1.00 │ 3.00 │ 2.97 │ 2.70 │ 2.25 │\n"
            "│     Request latency (ms) │ 8.00 │ 7.00 │ 9.00 │ 8.98 │ 8.80 │ 8.50 │\n"
            "│         Num output token │ 4.50 │ 3.00 │ 6.00 │ 5.97 │ 5.70 │ 5.25 │\n"
            "│          Num input token │ 3.50 │ 3.00 │ 4.00 │ 3.99 │ 3.90 │ 3.75 │\n"
            "└──────────────────────────┴──────┴──────┴──────┴──────┴──────┴──────┘\n"
            "Output token throughput (per sec): 900.00\n"
            "Request throughput (per sec): 200.00\n"
        )

        returned_data = capturedTable.getvalue()

        assert returned_data == expected_content
