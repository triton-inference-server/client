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


from genai_perf.export_data.exporter_config import ExporterConfig
from rich.console import Console
from rich.table import Table


class ConsoleExporter:
    """
    A class to export the statistics and arg values to the console.
    """

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p75"]

    def __init__(self, config: ExporterConfig):
        self._stats = config.stats
        self._metrics = config.metrics
        self._args = config.args
        self._benchmark_duration = config.benchmark_duration

    def _get_title(self):
        if self._args.endpoint_type == "embeddings":
            return "Embeddings Metrics"
        elif self._args.endpoint_type == "rankings":
            return "Rankings Metrics"
        else:
            return "LLM Metrics"

    def export(self) -> None:
        table = Table(title=self._get_title())

        table.add_column("Statistic", justify="right", style="cyan", no_wrap=True)
        for stat in self.STAT_COLUMN_KEYS:
            table.add_column(stat, justify="right", style="green")

        # Request metrics table
        self._construct_table(table)

        console = Console()
        console.print(table)

        # System metrics are printed after the table
        for metric in self._metrics.system_metrics:
            line = metric.name.replace("_", " ").capitalize()
            value = self._stats[metric.name]["avg"]
            line += f" ({metric.unit}): {value:.2f}"
            print(line)
        
        if self._args.goodput_constraints:
            total_count, good_count = self._count_good_req()
            ttft_constraint, itl_constraint = self._args.goodput_constraints
            line = f"Out of {total_count} requests, {good_count} are Good under the constraints of TTFT: {ttft_constraint:.2f}ms, ITL: {itl_constraint:.2f}ms"        
            print(line)
            goodput_value = good_count / self._benchmark_duration
            goodput_line = f"Request goodput (per sec): {goodput_value:.2f}"
            print(goodput_line)
            

    def _count_good_req(self):
        ttft_constraint_ms, itl_constraint_ms = self._args.goodput_constraints # List:[TTFT, ITL]
        # ms to ns
        ttft_constraint, itl_constraint = ttft_constraint_ms * 1e6, itl_constraint_ms * 1e6
        time_to_first_tokens = self._metrics.time_to_first_tokens
        inter_token_latencies = self._metrics.inter_token_latencies
        good_req_count = 0
        total_req = len(time_to_first_tokens)
        for ttft, itl in zip(time_to_first_tokens, inter_token_latencies):
            if ttft <= ttft_constraint and itl <= itl_constraint:
                good_req_count += 1
        return total_req, good_req_count

    def _construct_table(self, table: Table) -> None:
        for metric in self._metrics.request_metrics:
            if self._should_skip(metric.name):
                continue

            metric_str = metric.name.replace("_", " ").capitalize()
            metric_str += f" ({metric.unit})" if metric.unit != "tokens" else ""
            row_values = [metric_str]
            for stat in self.STAT_COLUMN_KEYS:
                value = self._stats[metric.name][stat]
                row_values.append(f"{value:,.2f}")

            table.add_row(*row_values)

    # (TMA-1976) Refactor this method as the csv exporter shares identical method.
    def _should_skip(self, metric_name: str) -> bool:
        if self._args.endpoint_type == "embeddings":
            return False  # skip nothing

        # TODO (TMA-1712): need to decide if we need this metric. Remove
        # from statistics display for now.
        # TODO (TMA-1678): output_token_throughput_per_request is treated
        # separately since the current code treats all throughput metrics to
        # be displayed outside of the statistics table.
        if metric_name == "output_token_throughput_per_request":
            return True

        # When non-streaming, skip ITL and TTFT
        streaming_metrics = [
            "inter_token_latency",
            "time_to_first_token",
        ]
        if not self._args.streaming and metric_name in streaming_metrics:
            return True
        return False
    
    def _count_good_req(self):
        ttft_constraint_ms, itl_constraint_ms = self._args.goodput_constraints # List:[TTFT, ITL]
        # ms to ns
        ttft_constraint, itl_constraint = ttft_constraint_ms * 1e6, itl_constraint_ms * 1e6
        time_to_first_tokens = self._metrics.time_to_first_tokens
        inter_token_latencies = self._metrics.inter_token_latencies
        good_req_count = 0
        total_req = len(time_to_first_tokens)
        for ttft, itl in zip(time_to_first_tokens, inter_token_latencies):
            if ttft <= ttft_constraint and itl <= itl_constraint:
                good_req_count += 1
        return total_req, good_req_count

