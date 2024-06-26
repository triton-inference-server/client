#!/usr/bin/env python3

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

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from genai_perf.metrics.metrics import Metrics


class Statistics:
    """A class that aggregates various statistics from given metrics class.

    The Statistics class goes through each metric in the metrics class and
    calculates several statistics such as:
      - average (arithmetic mean)
      - percentiles (p25, p50, p75, p90, p95, p99)
      - minimum & maximum
      - standard deviation
    The class will store each calculated statistics as part of its attribute.

    Example:

      >>> metrics = LLMMetrics(request_throughputs=[2, 4])
      >>> stats = Statistics(metrics)
      >>> print(stats.avg_request_throughput)  # output: 3
    """

    def __init__(self, metrics: Metrics):
        # iterate through Metrics to calculate statistics and set attributes
        self._metrics = metrics
        self._stats_dict: Dict = defaultdict(dict)
        for attr, data in metrics.data.items():
            if self._should_skip(data, attr):
                continue

            attr = metrics.get_base_name(attr)
            self._add_units(attr)
            self._calculate_mean(data, attr)
            if not self._is_system_metric(metrics, attr):
                self._calculate_percentiles(data, attr)
                self._calculate_minmax(data, attr)
                self._calculate_std(data, attr)

    def _should_skip(self, data: List[Union[int, float]], attr: str) -> bool:
        """Checks if some metrics should be skipped."""
        # No data points
        if len(data) == 0:
            return True
        # Skip ITL when non-streaming (all zero)
        elif attr == "inter_token_latencies" and sum(data) == 0:
            return True
        return False

    def _calculate_mean(self, data: List[Union[int, float]], attr: str) -> None:
        avg = np.mean(data)
        setattr(self, "avg_" + attr, avg)
        self._stats_dict[attr]["avg"] = float(avg)

    def _calculate_percentiles(self, data: List[Union[int, float]], attr: str) -> None:
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        p90, p95, p99 = np.percentile(data, [90, 95, 99])
        setattr(self, "p25_" + attr, p25)
        setattr(self, "p50_" + attr, p50)
        setattr(self, "p75_" + attr, p75)
        setattr(self, "p90_" + attr, p90)
        setattr(self, "p95_" + attr, p95)
        setattr(self, "p99_" + attr, p99)
        self._stats_dict[attr]["p99"] = float(p99)
        self._stats_dict[attr]["p95"] = float(p95)
        self._stats_dict[attr]["p90"] = float(p90)
        self._stats_dict[attr]["p75"] = float(p75)
        self._stats_dict[attr]["p50"] = float(p50)
        self._stats_dict[attr]["p25"] = float(p25)

    def _calculate_minmax(self, data: List[Union[int, float]], attr: str) -> None:
        min, max = np.min(data), np.max(data)
        setattr(self, "min_" + attr, min)
        setattr(self, "max_" + attr, max)
        self._stats_dict[attr]["max"] = float(max)
        self._stats_dict[attr]["min"] = float(min)

    def _calculate_std(self, data: List[Union[int, float]], attr: str) -> None:
        std = np.std(data)
        setattr(self, "std_" + attr, std)
        self._stats_dict[attr]["std"] = float(std)

    def scale_data(self, factor: float = 1 / 1e6) -> None:
        for k1, v1 in self.stats_dict.items():
            if self._is_time_metric(k1):
                for k2, v2 in v1.items():
                    if k2 != "unit":
                        self.stats_dict[k1][k2] = self._scale(v2, factor)

    def _scale(self, metric: float, factor: float = 1 / 1e6) -> float:
        """
        Scale metrics from nanoseconds by factor.
        Default is nanoseconds to milliseconds.
        """
        return metric * factor

    def _add_units(self, key) -> None:
        if self._is_time_metric(key):
            self._stats_dict[key]["unit"] = "ms"
        if key == "request_throughput":
            self._stats_dict[key]["unit"] = "requests/sec"
        if key.startswith("output_token_throughput"):
            self._stats_dict[key]["unit"] = "tokens/sec"
        if "sequence_length" in key:
            self._stats_dict[key]["unit"] = "tokens"

    def __repr__(self) -> str:
        attr_strs = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                attr_strs.append(f"{k}={v}")
        return f"Statistics({','.join(attr_strs)})"

    @property
    def data(self) -> dict:
        """Return all the aggregated statistics."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def metrics(self) -> Metrics:
        """Return the underlying metrics used to calculate the statistics."""
        return self._metrics

    @property
    def stats_dict(self) -> Dict:
        return self._stats_dict

    def _is_system_metric(self, metrics: Metrics, attr: str) -> bool:
        system_metrics = [m for m, _ in metrics.system_metric_names]
        return attr in system_metrics

    def _is_time_metric(self, field: str) -> bool:
        # TMA-?: Remove the hardcoded time metrics list
        time_metrics = [
            "inter_token_latency",
            "time_to_first_token",
            "request_latency",
        ]
        return field in time_metrics

    def export_parquet(self, artifact_dir: Path, filename: str) -> None:
        max_length = -1
        col_index = 0
        filler_list = []
        df = pd.DataFrame()

        # Data frames require all columns of the same length
        # find the max length column
        for key, value in self._metrics.data.items():
            max_length = max(max_length, len(value))

        # Insert None for shorter columns to match longest column
        for key, value in self._metrics.data.items():
            if len(value) < max_length:
                diff = max_length - len(value)
                filler_list = [None] * diff
            df.insert(col_index, key, value + filler_list)
            diff = 0
            filler_list = []
            col_index = col_index + 1

        filepath = artifact_dir / f"{filename}.gzip"
        df.to_parquet(filepath, compression="gzip")
