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


from copy import deepcopy
from typing import Dict
from pathlib import Path

from genai_perf.constants import DEFAULT_ARTIFACT_DIR
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_metrics import Statistics
from pandas import DataFrame
from plotly.graph_objects import Figure
from genai_perf.plots.plot_config import ProfileRunData


class BasePlot:
    """
    Base class for plots
    """

    def __init__(self, data: list[ProfileRunData]) -> None:
        self._profile_data = data

    def create_plot(
        self,
        x_metric: str,
        y_metric: str,
        graph_title: str,
        x_label: str,
        y_label: str,
        filename_root: str,
        output_dir: Path,
    ) -> None:
        """
        Create plot for specific graph type
        """
        raise NotImplementedError

    def _generate_parquet(self, df: DataFrame, output_dir: Path, file: str) -> None:
        filepath = output_dir / f"{file}.gzip"
        df.to_parquet(filepath, compression="gzip")

    def _generate_graph_file(self, fig: Figure, output_dir: Path, file: str, title: str) -> None:
        if file.endswith("jpeg"):
            print(f"Generating '{title}' jpeg")
            filepath = output_dir / f"{file}.jpeg"
            fig.write_image(filepath)
        elif file.endswith("html"):
            print(f"Generating '{title}' html")
            filepath = output_dir / f"{file}.html"
            fig.write_html(filepath)
        else:
            extension = file.split(".")[-1]
            raise GenAIPerfException(f"image file type {extension} is not supported")
