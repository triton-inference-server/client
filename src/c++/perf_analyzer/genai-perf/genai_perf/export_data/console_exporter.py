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


from typing import Dict

from genai_perf.llm_metrics import Metrics
from rich.console import Console
from rich.table import Table


class ConsoleExporter:
    """
    A class to export the statistics and arg values to the console.
    """

    def __init__(self, config: Dict):
        self._stats = config["stats"]

    def export(self) -> None:
        singular_metric_rows = []
        table = Table(title="LLM Metrics")

        table.add_column("Statistic", justify="right", style="cyan", no_wrap=True)
        stats = ["avg", "min", "max", "p99", "p90", "p75"]
        for stat in stats:
            table.add_column(stat, justify="right", style="green")

        for metric in Metrics.metric_labels:
            formatted_metric = metric.replace("_", " ").capitalize()

            # Throughput fields are printed after the table
            is_throughput_field = metric in Metrics.throughput_fields
            if is_throughput_field:
                value = self._stats.get(f"{metric}", -1).get(stats[0], -1)
                formatted_metric += f" (per sec): {value:.2f}"
                singular_metric_rows.append(formatted_metric)
                continue

            # TODO (TMA-1712): need to decide if we need this metric. Remove
            # from statistics display for now.
            # TODO (TMA-1678): output_token_throughput_per_request is treated
            # separately since the current code treats all throughput metrics to
            # be displayed outside of the statistics table.
            if metric == "output_token_throughput_per_request":
                formatted_metric += f" (per sec)"
                continue

            is_time_field = metric in Metrics.time_fields
            if is_time_field:
                formatted_metric += " (ms)"

            row_values = [formatted_metric]

            for stat in stats:
                value = self._stats.get(f"{metric}", -1).get(stat, -1)
                row_values.append(f"{value:,.2f}")

            # Without streaming, there is no inter-token latency available, so do not print it.
            if metric == "inter_token_latency":
                if all(value == "-1" for value in row_values[1:]):
                    continue
            # Without streaming, TTFT and request latency are the same, so do not print TTFT.
            elif metric == "time_to_first_token":
                unique_values = False
                for stat in stats:
                    value_ttft = self._stats.get(f"{metric}", -1).get(stat, -1)
                    value_req_latency = self._stats.get("request_latency", -1).get(
                        stat, -1
                    )
                    if value_ttft != value_req_latency:
                        unique_values = True
                        break
                if not unique_values:
                    continue

            table.add_row(*row_values)

        console = Console()
        console.print(table)

        for row in singular_metric_rows:
            print(row)
