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


from typing import List

from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.metrics import Metrics
from rich.console import Console
from rich.table import Table


class ConsoleExporter:
    """
    A class to export the statistics and arg values to the console.
    """

    STAT_COLUMNS = ["avg", "min", "max", "p99", "p90", "p75"]

    def __init__(self, config: ExporterConfig):
        self._stats = config.stats
        self._metrics = config.metrics
        self._args = config.args

    def _get_title(self):
        if self._args.endpoint_type == "embeddings":
            return "Embedding Metrics"
        return "LLM Metrics"

    def export(self) -> None:
        table = Table(title=self._get_title())

        table.add_column("Statistic", justify="right", style="cyan", no_wrap=True)
        for stat in self.STAT_COLUMNS:
            table.add_column(stat, justify="right", style="green")

        # Request metrics table
        self._construct_table(table)

        console = Console()
        console.print(table)

        # System metrics are printed after the table
        for metric, unit in self._metrics.system_metric_names:
            formatted_metric = metric.replace("_", " ").capitalize()
            value = self._stats[metric]["avg"]
            formatted_metric += f" ({unit}): {value:.2f}"
            print(formatted_metric)

    def _construct_table(self, table: Table) -> None:
        for metric, unit in self._metrics.request_metric_names:
            if self._should_skip(metric):
                continue

            formatted_metric = metric.replace("_", " ").capitalize()
            formatted_metric += f" ({unit})" if unit != "tokens" else ""
            row_values = [formatted_metric]
            for stat in self.STAT_COLUMNS:
                value = self._stats[metric][stat]
                row_values.append(f"{value:,.2f}")

            table.add_row(*row_values)

    def _should_skip(self, metric: str) -> bool:
        if self._args.endpoint_type in ["embeddings", "ranking"]:
            return False  # skip nothing

        # TODO (TMA-1712): need to decide if we need this metric. Remove
        # from statistics display for now.
        # TODO (TMA-1678): output_token_throughput_per_request is treated
        # separately since the current code treats all throughput metrics to
        # be displayed outside of the statistics table.
        if metric == "output_token_throughput_per_request":
            return True

        # When non-streaming, skip ITL and TTFT
        streaming_metrics = [
            "inter_token_latency",
            "time_to_first_token",
        ]
        if not self._args.streaming and metric in streaming_metrics:
            return True
        return False
