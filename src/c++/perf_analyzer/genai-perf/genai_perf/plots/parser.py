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

from pathlib import Path

# Ignore yaml import to avoid mypy error
# Issue: https://github.com/python/mypy/issues/10632
import yaml  # type: ignore
from genai_perf.llm_metrics import Statistics
from genai_perf.plots.config import PlotConfig, PlotType, ProfileRunData
from genai_perf.utils import scale


class PlotConfigParser:
    """Parses YAML configuration file to generate PlotConfigs."""

    def __init__(self, filename: str | None = None) -> None:
        if filename:
            self._plot_configs = self._parse_config(filename)

    def _parse_config(self, filename: str) -> list[PlotConfig]:
        """Load YAML configuration file and convert to PlotConfigs."""
        with open(filename) as f:
            configs = yaml.safe_load(f)

        plot_configs: list[PlotConfig] = []
        return plot_configs

    @staticmethod
    def create_default_configs(stats: Statistics, filename: Path) -> list[PlotConfig]:
        """Creates a set of default plot configurations for single run plots."""
        ttfts = stats.metrics.data["time_to_first_tokens"]
        req_latencies = stats.metrics.data["request_latencies"]

        itls = []
        token_positions = []
        for itls_per_req in stats.metrics.data["inter_token_latencies"]:
            itls += itls_per_req
            token_positions += list(range(1, len(itls_per_req) + 1))

        # scale to seconds
        scaled_ttfts = [scale(x, (1 / 1e9)) for x in ttfts]
        scaled_req_latencies = [scale(x, (1 / 1e9)) for x in req_latencies]
        scaled_itls = [scale(x, (1 / 1e9)) for x in itls]

        return [
            PlotConfig(
                title="Time to First Token",
                data=[
                    ProfileRunData(
                        name=filename.stem,
                        x_metric=[],
                        y_metric=scaled_ttfts,
                    )
                ],
                x_label="Time to First Token (seconds)",
                y_label="",
                type=PlotType.BOX,
                output=Path(""),
            ),
            PlotConfig(
                title="Request Latency",
                data=[
                    ProfileRunData(
                        name=filename.stem,
                        x_metric=[],
                        y_metric=scaled_req_latencies,
                    )
                ],
                x_label="Request Latency (seconds)",
                y_label="",
                type=PlotType.BOX,
                output=Path(""),
            ),
            PlotConfig(
                title="Distribution of Input Tokens to Generated Tokens",
                data=[
                    ProfileRunData(
                        name=filename.stem,
                        x_metric=stats.metrics.data["num_input_tokens"],
                        y_metric=stats.metrics.data["num_output_tokens"],
                    )
                ],
                x_label="Number of Input Tokens Per Request",
                y_label="Number of Generated Tokens Per Request",
                type=PlotType.HEATMAP,
                output=Path(""),
            ),
            PlotConfig(
                title="Time to First Token vs Number of Input Tokens",
                data=[
                    ProfileRunData(
                        name=filename.stem,
                        x_metric=stats.metrics.data["num_input_tokens"],
                        y_metric=scaled_ttfts,
                    )
                ],
                x_label="Number of Input Tokens",
                y_label="Time to First Token (seconds)",
                type=PlotType.SCATTER,
                output=Path(""),
            ),
            PlotConfig(
                title="Token-to-Token Latency vs Output Token Position",
                data=[
                    ProfileRunData(
                        name=filename.stem,
                        x_metric=token_positions,
                        y_metric=scaled_itls,
                    )
                ],
                x_label="Output Token Position",
                y_label="Token-to-Token Latency (seconds)",
                type=PlotType.SCATTER,
                output=Path(""),
            ),
        ]

    def generate_configs(self):
        pass
