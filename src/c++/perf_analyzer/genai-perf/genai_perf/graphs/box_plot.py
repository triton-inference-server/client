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


import copy
from typing import Dict

import pandas as pd
import plotly.express as px
from genai_perf.graphs.base_plot import BasePlot
from genai_perf.llm_metrics import Statistics
from genai_perf.utils import scale
from plotly.graph_objects import Figure


class BoxPlot(BasePlot):
    """
    Generate a box plot in jpeg and html format.
    """

    def __init__(self, stats: Statistics, extra_data: Dict | None = None) -> None:
        super().__init__(stats, extra_data)

    def create_plot(
        self,
        x_key: str = "",
        y_key: str = "",
        x_metric: str = "",
        y_metric: str = "",
        graph_title: str = "",
        x_label: str = "",
        y_label: str = "",
        filename_root: str = "",
    ):
        df = pd.DataFrame({y_metric: self._metrics_data[y_key]})
        fig = px.box(
            df,
            y=y_metric,
            points="all",
            title=graph_title,
        )
        fig.update_layout(title_x=0.5)
        fig.update_xaxes(title_text=x_label)

        fig.update_yaxes(title_text="")

        # create a copy to avoid annotations on html file
        fig_jpeg = copy.deepcopy(fig)
        self._add_annotations(fig_jpeg, y_metric)

        self._generate_parquet(df, filename_root)
        self._generate_graph_file(fig, filename_root + ".html", graph_title)
        self._generate_graph_file(fig_jpeg, filename_root + ".jpeg", graph_title)

    def _add_annotations(self, fig: Figure, y_metric: str) -> None:
        """
        Add annotations to the non html version of the box plot
        to replace the missing hovertext
        """
        stat_root_name = self._stats.metrics.get_base_name(y_metric)

        val = scale(self._stats.data[f"max_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"max: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )

        val = scale(self._stats.data[f"p75_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"q3: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )

        val = scale(self._stats.data[f"p50_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"median: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )

        val = scale(self._stats.data[f"p25_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"q1: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )

        val = scale(self._stats.data[f"min_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"min: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )
