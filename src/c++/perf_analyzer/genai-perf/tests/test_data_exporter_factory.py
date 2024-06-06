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


from argparse import Namespace
from pathlib import Path

import genai_perf.export_data.data_exporter_factory as factory
from genai_perf.export_data.console_exporter import ConsoleExporter
from genai_perf.export_data.csv_exporter import CsvExporter
from genai_perf.export_data.json_exporter import JsonExporter
from genai_perf.parser import get_extra_inputs_as_dict


class TestOutputReporter:
    stats = {
        "request_latency": {
            "unit": "ms",
            "avg": 1,
            "p99": 2,
            "p95": 3,
            "p90": 4,
            "p75": 5,
            "p50": 6,
            "p25": 7,
            "max": 8,
            "min": 9,
            "std": 0,
        },
    }
    args = {
        "model": ["gpt2_vllm"],
        "formatted_model_name": "gpt2_vllm",
        "model_selection_strategy": "round_robin",
        "func": "Should_be_removed",
        "output_format": "Should_be_removed",
        "profile_export_file": ".",
        "artifact_dir": ".",
        "extra_inputs": ["max_tokens:200"],
    }
    args_namespace = Namespace(**args)

    def test_return_json_exporter(self) -> None:
        config = {
            "type": factory.DataExporterType.JSON,
            "stats": self.stats,
            "args": self.args_namespace,
            "extra_inputs": get_extra_inputs_as_dict(self.args_namespace),
            "artifact_dir": Path("."),
        }
        f = factory.DataExporterFactory()
        exporter = f.create_data_exporter(config)
        assert isinstance(exporter, JsonExporter)

    def test_return_csv_exporter(self) -> None:
        config = {
            "type": factory.DataExporterType.CSV,
            "stats": self.stats,
            "artifact_dir": Path("."),
        }
        f = factory.DataExporterFactory()
        exporter = f.create_data_exporter(config)
        assert isinstance(exporter, CsvExporter)

    def test_return_console_exporter(self) -> None:
        config = {
            "type": factory.DataExporterType.CONSOLE,
            "stats": self.stats,
            "artifact_dir": Path("."),
        }
        f = factory.DataExporterFactory()
        exporter = f.create_data_exporter(config)
        assert isinstance(exporter, ConsoleExporter)
