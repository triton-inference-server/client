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


import os

from genai_perf.graphs.box_plot import BoxPlot
from genai_perf.graphs.heat_map import HeatMap
from genai_perf.graphs.scatter_plot import ScatterPlot
from genai_perf.llm_metrics import Statistics


class PlotManager:
    """
    Manage details around plots generated
    """

    def __init__(self, stats: Statistics) -> None:
        self._stats = stats

    def create_default_graphs(self):
        bp = BoxPlot(self._stats)
        bp.create_box_plot(
            y_key="time_to_first_tokens",
            graph_title="Time to First Token Analysis",
            filename_root="ttft",
            x_label="Time to First Token of Individual Requests (seconds)",
        )
        bp.create_box_plot(
            y_key="request_latencies",
            graph_title="Total Output Time Analysis",
            filename_root="tot",
            x_label="Time to Completion of Individual Requests (seconds)",
        )

        hm = HeatMap(self._stats)
        hm.create_heat_map(
            x_key="num_input_tokens",
            y_key="num_output_tokens",
            x_metric="input_tokens",
            y_metric="generated_tokens",
            graph_title="Distribution of Input Tokens to Generated Tokens (Over All Requests)",
            x_label="Number of Input Tokens Per Request",
            y_label="Number of Generated Tokens Per Request",
            filename_root="heatmap",
        )

        sp = ScatterPlot(self._stats)
        sp.create_scatter_plot(
            x_key="num_input_tokens",
            y_key="time_to_first_tokens",
            scale_y=True,
            graph_title="Time to First Token vs Size of Initial Prompt",
            x_label="Number of Input Tokens",
            y_label="Time to First Token (seconds)",
            filename_root="ttft_v_input_tokens",
        )

        itl_latencies = self._stats.metrics.data["inter_token_latencies"]
        x_data = []
        y_data = []
        for itl_latency_list in itl_latencies:
            for index, latency in enumerate(itl_latency_list):
                x_data.append(index + 1)
                y_data.append(latency / 1e9)

        sp.create_scatter_plot(
            x_key="token_position",
            y_key="inter_token_latencies",
            graph_title="Token-to-Token Latencies vs Token Position",
            x_label="Token Position",
            y_label="Token to Token Latency (seconds)",
            filename_root="itl_v_position",
            preprocessed_x_data=x_data,
            preprocessed_y_data=y_data,
        )
