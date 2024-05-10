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
from typing import List

import plotly.graph_objects as go
from genai_perf.plots.base_plot import BasePlot
from genai_perf.plots.plot_config import ProfileRunData


class BoxPlot(BasePlot):
    """
    Generate a box plot in jpeg and html format.
    """

    def __init__(self, data: List[ProfileRunData]) -> None:
        super().__init__(data)

    def create_plot(
        self,
        graph_title: str = "",
        x_label: str = "",
        y_label: str = "",
        width: int = 700,
        height: int = 450,
        filename_root: str = "",
        output_dir: Path = Path(""),
    ) -> None:
        fig = go.Figure()
        for pd in self._profile_data:
            fig.add_trace(go.Box(y=pd.y_metric, name=pd.name))

        # Update layout and axis labels
        fig.update_layout(
            title={
                "text": f"{graph_title}",
                "xanchor": "center",
                "x": 0.5,
            },
            width=width,
            height=height,
        )
        fig.update_traces(boxpoints="all")
        fig.update_xaxes(title_text=x_label, showticklabels=False)
        fig.update_yaxes(title_text=y_label)

        # Save dataframe as parquet file
        df = self._create_dataframe(x_label, y_label)
        self._generate_parquet(df, output_dir, filename_root)

        self._generate_graph_file(fig, output_dir, filename_root + ".html")
        self._generate_graph_file(fig, output_dir, filename_root + ".jpeg")
