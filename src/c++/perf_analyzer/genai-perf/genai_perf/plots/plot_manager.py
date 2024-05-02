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


from genai_perf.llm_metrics import Statistics
from genai_perf.plots.box_plot import BoxPlot
from genai_perf.plots.heat_map import HeatMap
from genai_perf.plots.scatter_plot import ScatterPlot
from genai_perf.utils import scale
from genai_perf.plots.config import PlotConfig, PlotType, ProfileRunData


class PlotManager:
    """
    Manage details around plots generated
    """

    def __init__(self, plot_configs: list[PlotConfig]) -> None:
        self._plot_configs = plot_configs

    def create_default_plots(self):
        bp_ttft = BoxPlot(self._plot_configs[0].data)
        bp_ttft.create_plot(
            #y_metric=y_metric,
            graph_title=self._plot_configs[0].title,
            filename_root="time_to_first_token",
            x_label=self._plot_configs[0].x_label,
        )

        bp_req_lat = BoxPlot(self._plot_configs[1].data)
        bp_req_lat.create_plot(
            graph_title=self._plot_configs[1].title,
            filename_root="request_latency",
            x_label=self._plot_configs[1].x_label,
        )

        hm = HeatMap(self._plot_configs[2].data)
        hm.create_plot(
            x_metric="input_tokens",
            y_metric="generated_tokens",
            graph_title=self._plot_configs[2].title,
            x_label=self._plot_configs[2].x_label,
            y_label=self._plot_configs[2].y_label,
            filename_root="input_tokens_vs_generated_tokens",
        )

        x_metric = "num_input_tokens"
        y_metric = "time_to_first_tokens"
        sp_ttft_vs_input_tokens = ScatterPlot(self._plot_configs[3].data)
        sp_ttft_vs_input_tokens.create_plot(
            x_metric=x_metric,
            y_metric=y_metric,
            graph_title=self._plot_configs[3].title,
            x_label=self._plot_configs[3].x_label,
            y_label=self._plot_configs[3].y_label,
            filename_root="ttft_vs_input_tokens",
        )

        x_metric = "token_position"
        y_metric = "inter_token_latency"
        sp_tot_v_tok_pos = ScatterPlot(self._plot_configs[4].data)
        sp_tot_v_tok_pos.create_plot(
            x_metric=x_metric,
            y_metric=y_metric,
            graph_title=self._plot_configs[4].title,
            x_label=self._plot_configs[4].x_label,
            y_label=self._plot_configs[4].y_label,
            filename_root="token_to_token_vs_output_position",
        )
