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


class HeatMap:
    """
    Generate a heat map in jpeg and html format.
    """

    def __init__(self, stats: Statistics) -> None:
        self._stats = stats.data
        self._metrics = stats.metrics

    def create_heat_map(
        self,
        x_key: str,
        y_key: str,
        x_metric: str,
        y_metric: str,
        graph_title: str,
        x_label: str,
        y_label: str,
        filename_root: str,
    ):
        x_values = self._metrics.data[x_key]
        y_values = self._metrics.data[y_key]
        df = pd.DataFrame(
            {
                x_metric: x_values,
                y_metric: y_values,
            }
        )
        fig = px.density_heatmap(
            df,
            x=x_metric,
            y=y_metric,
        )
        fig.update_layout(
            title={
                "text": graph_title,
                "xanchor": "center",
                "x": 0.5,
            }
        )
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)

        if not os.path.exists("images"):
            os.mkdir("images")
        print(f"Generating '{graph_title}' html and jpeg files")
        fig.write_html(f"images/{filename_root}.html")
        fig.write_image(f"images/{filename_root}.jpeg")
