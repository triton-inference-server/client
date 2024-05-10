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

from typing import List

import genai_perf.logging as logging
from genai_perf.plots.box_plot import BoxPlot
from genai_perf.plots.heat_map import HeatMap
from genai_perf.plots.plot_config import PlotConfig, PlotType
from genai_perf.plots.scatter_plot import ScatterPlot

logger = logging.getLogger(__name__)


class PlotManager:
    """
    Manage details around plots generated
    """

    def __init__(self, plot_configs: List[PlotConfig]) -> None:
        self._plot_configs = plot_configs

    def _generate_filename(self, title: str) -> str:
        filename = "_".join(title.lower().split())
        return filename

    def generate_plots(self) -> None:
        for plot_config in self._plot_configs:
            logger.info(f"Generating '{plot_config.title}' plot")
            if plot_config.type == PlotType.BOX:
                bp = BoxPlot(plot_config.data)
                bp.create_plot(
                    graph_title=plot_config.title,
                    x_label=plot_config.x_label,
                    width=plot_config.width,
                    height=plot_config.height,
                    filename_root=self._generate_filename(plot_config.title),
                    output_dir=plot_config.output,
                )

            elif plot_config.type == PlotType.HEATMAP:
                hm = HeatMap(plot_config.data)
                hm.create_plot(
                    graph_title=plot_config.title,
                    x_label=plot_config.x_label,
                    y_label=plot_config.y_label,
                    width=plot_config.width,
                    height=plot_config.height,
                    filename_root=self._generate_filename(plot_config.title),
                    output_dir=plot_config.output,
                )

            elif plot_config.type == PlotType.SCATTER:
                sp = ScatterPlot(plot_config.data)
                sp.create_plot(
                    graph_title=plot_config.title,
                    x_label=plot_config.x_label,
                    y_label=plot_config.y_label,
                    width=plot_config.width,
                    height=plot_config.height,
                    filename_root=self._generate_filename(plot_config.title),
                    output_dir=plot_config.output,
                )
