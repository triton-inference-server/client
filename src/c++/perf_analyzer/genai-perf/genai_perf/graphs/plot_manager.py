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


from genai_perf.graphs.box_plot import BoxPlot
from genai_perf.graphs.heat_map import HeatMap
from genai_perf.graphs.scatter_plot import ScatterPlot
from genai_perf.llm_metrics import Statistics
from genai_perf.utils import scale


class PlotManager:
    """
    Manage details around plots generated
    """

    def __init__(self, stats: Statistics) -> None:
        self._stats = stats

    def create_default_graphs(self):
        y_metric = "time_to_first_tokens"
        y_key = "time_to_first_tokens_scaled"
        scaled_data = [scale(x, (1 / 1e9)) for x in self._stats.metrics.data[y_metric]]
        extra_data = {y_key: scaled_data}
        bp_ttft = BoxPlot(self._stats, extra_data)
        bp_ttft.create_plot(
            y_key=y_key,
            y_metric=y_metric,
            graph_title="Time to First Token",
            filename_root="time_to_first_token",
            x_label="Time to First Token (seconds)",
        )

        y_metric = "request_latencies"
        y_key = "request_latencies_scaled"
        scaled_data = [scale(x, (1 / 1e9)) for x in self._stats.metrics.data[y_metric]]
        extra_data = {y_key: scaled_data}
        bp_req_lat = BoxPlot(self._stats, extra_data)
        bp_req_lat.create_plot(
            y_key=y_key,
            y_metric=y_metric,
            graph_title="Request Latency",
            filename_root="request_latency",
            x_label="Request Latency (seconds)",
        )

        hm = HeatMap(self._stats)
        hm.create_plot(
            x_key="num_input_tokens",
            y_key="num_output_tokens",
            x_metric="input_tokens",
            y_metric="generated_tokens",
            graph_title="Distribution of Input Tokens to Generated Tokens",
            x_label="Number of Input Tokens Per Request",
            y_label="Number of Generated Tokens Per Request",
            filename_root="input_tokens_vs_generated_tokens",
        )

        x_metric = "num_input_tokens"
        y_metric = "time_to_first_tokens"
        y_key = "time_to_first_tokens_scaled"
        scaled_data = [scale(x, (1 / 1e9)) for x in self._stats.metrics.data[y_metric]]
        extra_data = {y_key: scaled_data}
        sp_ttft_vs_input_tokens = ScatterPlot(self._stats, extra_data)
        sp_ttft_vs_input_tokens.create_plot(
            x_key=x_metric,
            y_key=y_key,
            x_metric=x_metric,
            y_metric=y_metric,
            graph_title="Time to First Token vs Number of Input Tokens",
            x_label="Number of Input Tokens",
            y_label="Time to First Token (seconds)",
            filename_root="ttft_vs_input_tokens",
        )

        itl_latencies = self._stats.metrics.data["inter_token_latencies"]
        x_data = []
        y_data = []
        for itl_latency_list in itl_latencies:
            for index, latency in enumerate(itl_latency_list):
                x_data.append(index + 1)
                y_data.append(latency / 1e9)
        x_key = "token_position"
        y_key = "inter_token_latency"
        extra_data = {x_key: x_data, y_key: y_data}
        sp_tot_v_tok_pos = ScatterPlot(self._stats, extra_data)
        sp_tot_v_tok_pos.create_plot(
            x_key=x_key,
            y_key=y_key,
            graph_title="Token-to-Token Latency vs Output Token Position",
            x_label="Output Token Position",
            y_label="Token-to-Token Latency (seconds)",
            filename_root="token_to_token_vs_output_position",
        )
