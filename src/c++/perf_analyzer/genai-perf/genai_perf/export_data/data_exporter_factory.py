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

from enum import Enum
from typing import Any, Dict, cast

from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.console_exporter import ConsoleExporter
from genai_perf.export_data.csv_exporter import CsvExporter
from genai_perf.export_data.data_exporter_interface import DataExporterInterface
from genai_perf.export_data.json_exporter import JsonExporter


class DataExporterType(str, Enum):
    JSON = "json_exporter"
    CSV = "csv_exporter"
    CONSOLE = "console_exporter"


DataExporterMapping = {
    DataExporterType.JSON: JsonExporter,
    DataExporterType.CSV: CsvExporter,
    DataExporterType.CONSOLE: ConsoleExporter,
}


class DataExporterFactory:
    def create_data_exporter(self, config: Dict[str, Any]) -> DataExporterInterface:
        if config.get("type") is None:
            raise GenAIPerfException("No exporter type specified")
        exporter_class = DataExporterMapping[config["type"]]
        return cast(DataExporterInterface, exporter_class(config))
