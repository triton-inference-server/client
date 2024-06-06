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

from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.data_exporter_factory import (
    DataExporterFactory,
    DataExporterType,
)
from genai_perf.export_data.data_exporter_interface import DataExporterInterface
from genai_perf.llm_metrics import LLMProfileDataParser
from genai_perf.parser import get_extra_inputs_as_dict


class OutputReporter:
    """
    A class to orchestrate output generation.
    """

    def __init__(self, data_parser: LLMProfileDataParser, args: Namespace):
        self.args = args

        if args.concurrency:
            self.infer_mode = "concurrency"
            self.load_level = f"{args.concurrency}"
        elif args.request_rate:
            self.infer_mode = "request_rate"
            self.load_level = f"{args.request_rate}"
        else:
            raise GenAIPerfException("No valid infer mode specified")

        self.stats = data_parser.get_statistics(self.infer_mode, self.load_level)

    def report_output(self) -> None:
        factory = DataExporterFactory()
        data_exporters = []
        data_exporters.append(self._get_console_exporter(factory))
        data_exporters.append(self._get_json_exporter(factory))
        data_exporters.append(self._get_csv_exporter(factory))

        for exporter in data_exporters:
            exporter.export()

    def _get_console_exporter(
        self, factory: DataExporterFactory
    ) -> DataExporterInterface:
        config = {
            "type": DataExporterType.CONSOLE,
            "stats": self.stats.stats_dict,
        }
        return factory.create_data_exporter(config)

    def _get_json_exporter(self, factory: DataExporterFactory) -> DataExporterInterface:
        config = {
            "type": DataExporterType.JSON,
            "stats": self.stats.stats_dict,
            "args": self.args,
            "extra_inputs": get_extra_inputs_as_dict(self.args),
            "artifact_dir": self.args.artifact_dir,
        }
        return factory.create_data_exporter(config)

    def _get_csv_exporter(self, factory: DataExporterFactory) -> DataExporterInterface:
        config = {
            "type": DataExporterType.CSV,
            "stats": self.stats.stats_dict,
            "artifact_dir": self.args.artifact_dir,
        }
        return factory.create_data_exporter(config)
