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
from genai_perf.graphs.base_plot import BasePlot
from genai_perf.llm_metrics import Statistics


class BoxPlot(BasePlot):
    """
    Generate a box plot in jpeg and html format.
    """

    def __init__(self, stats: Statistics) -> None:
        super().__init__(stats)

    def create_box_plot(
        self,
        y_key: str,
        graph_title: str,
        filename_root: str,
        x_label: str,
    ):
        scaled_data = [self._scale(x, (1 / 1e9)) for x in self._metrics.data[y_key]]
        df = pd.DataFrame(scaled_data, columns=[y_key])

        fig = px.box(
            df,
            y=y_key,
            points="all",
            title=graph_title,
        )
        fig.update_layout(title_x=0.5)
        fig.update_xaxes(
            title_text=x_label,
        )

        fig.update_yaxes(title_text="")

        if not os.path.exists("images"):
            os.mkdir("images")
        print(f"Generating '{graph_title}' html and jpeg files")
        fig.write_html(f"images/{filename_root}.html")

        self._add_annotations(fig, y_key)

        fig.write_image(f"images/{filename_root}.jpeg")

    def _add_annotations(self, fig, y_key):
        stat_root_name = self._metrics.get_base_name(y_key)

        val = self._scale(self._stats[f"max_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"max: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )

        val = self._scale(self._stats[f"p75_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"q3: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )

        val = self._scale(self._stats[f"p50_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"median: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )

        val = self._scale(self._stats[f"p25_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"q1: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )

        val = self._scale(self._stats[f"min_{stat_root_name}"], (1 / 1e9))
        fig.add_annotation(
            x=0.5,
            y=val,
            text=f"min: {round(val, 2)}",
            showarrow=False,
            yshift=10,
        )
