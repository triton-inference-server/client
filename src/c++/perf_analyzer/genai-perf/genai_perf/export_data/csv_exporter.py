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


import csv
from typing import Dict

import genai_perf.logging as logging
from genai_perf.llm_metrics import Metrics

DEFAULT_OUTPUT_DATA_CSV = "profile_export_genai_perf.csv"

logger = logging.getLogger(__name__)


class CsvExporter:
    """
    A class to export the statistics and arg values in a csv format.
    """

    def __init__(self, config: Dict):
        self._stats = config["stats"]
        self._output_dir = config["artifact_dir"]

    def export(self) -> None:
        csv_filename = self._output_dir / DEFAULT_OUTPUT_DATA_CSV
        logger.info(f"Generating {csv_filename}")

        multiple_metric_header = [
            "Metric",
            "avg",
            "min",
            "max",
            "p99",
            "p95",
            "p90",
            "p75",
            "p50",
            "p25",
        ]

        single_metric_header = [
            "Metric",
            "Value",
        ]

        with open(csv_filename, mode="w", newline="") as csvfile:
            singular_metric_rows = []

            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(multiple_metric_header)

            for metric in Metrics.metric_labels:
                formatted_metric = metric.replace("_", " ").title()

                is_throughput_field = metric in Metrics.throughput_fields
                is_time_field = metric in Metrics.time_fields

                if is_time_field:
                    formatted_metric += " (ms)"
                elif is_throughput_field:
                    formatted_metric += " (per sec)"
                # TODO (TMA-1712): need to decide if we need this metric. Do not
                # include in the csv for now.
                # TODO (TMA-1678): output_token_throughput_per_request is treated
                # separately since the current code treats all throughput metrics
                # to be displayed outside of the statistics table.
                elif metric == "output_token_throughput_per_request":
                    formatted_metric += " (per sec)"
                    continue

                row_values = [formatted_metric]

                if is_throughput_field:
                    value = self._stats.get(f"{metric}", -1).get(
                        multiple_metric_header[1], -1
                    )
                    row_values.append(f"{value:.2f}")
                    singular_metric_rows.append(row_values)
                    continue

                for stat in multiple_metric_header[1:]:
                    value = self._stats.get(f"{metric}", -1).get(stat, -1)
                    row_values.append(f"{value:.2f}")

                # Without streaming, there is no inter-token latency available, so do not print it.
                if metric == "inter_token_latency":
                    if all(value == "-1" for value in row_values[1:]):
                        continue
                # Without streaming, TTFT and request latency are the same, so do not print TTFT.
                elif metric == "time_to_first_token":
                    unique_values = False
                    for stat in multiple_metric_header[1:]:
                        value_ttft = self._stats.get(f"{metric}", -1).get(stat, -1)
                        value_req_latency = self._stats.get("request_latency", -1).get(
                            stat, -1
                        )
                        if value_ttft != value_req_latency:
                            unique_values = True
                            break
                    if not unique_values:
                        continue

                csv_writer.writerow(row_values)

            csv_writer.writerow([])
            csv_writer.writerow(single_metric_header)
            for row in singular_metric_rows:
                csv_writer.writerow(row)
