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

import pandas as pd
import plotly.express as px
from genai_perf.llm_metrics import Statistics


class ScatterPlot:
    """
    Generate a scatter plot in jpeg and html format.
    """

    def __init__(self, stats: Statistics) -> None:
        self._stats = stats.data
        self._metrics = stats.metrics

    def _scale_ns_to_s(self, value):
        return value / 1000_000_000

    def create_scatter_plot(
        self,
        x_key: str = "",
        scale_x: bool = False,
        y_key: str = "",
        scale_y: bool = False,
        graph_title: str = "",
        x_label: str = "",
        y_label: str = "",
        filename_root: str = "",
        preprocessed_x_data=[],
        preprocessed_y_data=[],
    ):
        if not preprocessed_x_data and not preprocessed_y_data:
            if scale_x:
                x_values = list(map(self._scale_ns_to_s, self._metrics.data[x_key]))
            else:
                x_values = self._metrics.data[x_key]
            if scale_y:
                y_values = list(map(self._scale_ns_to_s, self._metrics.data[y_key]))
            else:
                y_values = self._metrics.data[y_key]
        else:
            x_values = preprocessed_x_data
            y_values = preprocessed_y_data

        df = pd.DataFrame(
            {
                x_key: x_values,
                y_key: y_values,
            }
        )

        fig = px.scatter(
            df,
            x=x_key,
            y=y_key,
            trendline="ols",
        )

        fig.update_layout(
            title={
                "text": f"{graph_title}",
                "xanchor": "center",
                "x": 0.5,
            }
        )
        fig.update_xaxes(title_text=f"{x_label}")
        fig.update_yaxes(title_text=f"{y_label}")

        if not os.path.exists("images"):
            os.mkdir("images")
        print(f"Generating '{graph_title}' html and jpeg files")
        fig.write_html(f"images/{filename_root}.html")
        fig.write_image(f"images/{filename_root}.jpeg")
