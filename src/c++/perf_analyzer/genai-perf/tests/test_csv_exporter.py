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
from pathlib import Path
from typing import Any, List

import pytest
from genai_perf.export_data.csv_exporter import CsvExporter
from genai_perf.export_data.data_exporter_factory import DataExporterType
from genai_perf.llm_metrics import LLMProfileDataParser
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_tokenizer
from tests.test_data import triton_profile_data


class TestCsvExporter:
    @pytest.fixture
    def mock_read_write(self, monkeypatch: pytest.MonkeyPatch) -> List[str]:
        """
        This function will mock the open function for specific files:

        """

        written_data = []

        original_open = open

        def custom_open(filename, *args, **kwargs):
            def write(self: Any, content: str) -> int:
                written_data.append(content)
                return len(content)

            if str(filename) == "triton_profile_export.json":
                tmp_file = StringIO(json.dumps(triton_profile_data))
                return tmp_file
            elif str(filename) == "profile_export_genai_perf.csv":
                tmp_file = StringIO()
                tmp_file.write = write.__get__(tmp_file)
                return tmp_file
            else:
                return original_open(filename, *args, **kwargs)

        monkeypatch.setattr("builtins.open", custom_open)

        return written_data

    def test_csv_output(self, mock_read_write: pytest.MonkeyPatch) -> None:
        """
        Collect LLM metrics from profile export data and confirm correct values are
        printed in csv.
        """

        tokenizer = get_tokenizer(DEFAULT_TOKENIZER)
        pd = LLMProfileDataParser(
            filename=Path("triton_profile_export.json"),
            tokenizer=tokenizer,
        )
        stat = pd.get_statistics(infer_mode="concurrency", load_level="10")

        expected_content = [
            "Metric,avg,min,max,p99,p95,p90,p75,p50,p25\r\n",
            "Time To First Token (ms),2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00\r\n",
            "Inter Token Latency (ms),2.00,1.00,3.00,2.97,2.85,2.70,2.25,2.00,1.75\r\n",
            "Request Latency (ms),8.00,7.00,9.00,8.98,8.90,8.80,8.50,8.00,7.50\r\n",
            "Num Output Token,4.50,3.00,6.00,5.97,5.85,5.70,5.25,4.50,3.75\r\n",
            "Num Input Token,3.50,3.00,4.00,3.99,3.95,3.90,3.75,3.50,3.25\r\n",
            "\r\n",
            "Metric,Value\r\n",
            "Output Token Throughput (per sec),900.00\r\n",
            "Request Throughput (per sec),200.00\r\n",
        ]

        config = {
            "type": DataExporterType.CSV,
            "stats": stat.stats_dict,
            "artifact_dir": Path("."),
        }
        exporter = CsvExporter(config)
        exporter.export()

        returned_data = mock_read_write

        assert returned_data == expected_content
