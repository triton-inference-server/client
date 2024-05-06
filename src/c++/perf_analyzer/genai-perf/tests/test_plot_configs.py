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

# Skip type checking to avoid mypy error
# Issue: https://github.com/python/mypy/issues/10632
import yaml  # type: ignore
from genai_perf.plots.config import PlotType
from genai_perf.plots.parser import PlotConfigParser


class TestPlotConfigParser:
    yaml_config = """
    plot1:
      title: TTFT vs ITL
      x_metric: time_to_first_tokens
      y_metric: inter_token_latencies
      x_label: TTFT (ms)
      y_label: ITL (ms)
      type: box
      paths:
        - run1/concurrency32.json
        - run2/concurrency32.json
        - run3/concurrency32.json
      output: test_output_1

    plot2:
      title: Num Input Token vs Num Output Token
      x_metric: num_input_tokens
      y_metric: num_output_tokens
      x_label: Input Tokens
      y_label: Output Tokens
      type: scatter
      paths:
        - run4/concurrency1.json
      output: test_output_2
    """

    def test_generate_configs(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "genai_perf.plots.parser.load_yaml",
            lambda _: yaml.safe_load(self.yaml_config),
        )
        monkeypatch.setattr(PlotConfigParser, "_get_statistics", lambda *_: {})
        monkeypatch.setattr(PlotConfigParser, "_get_metric", lambda *_: [1, 2, 3])

        config_parser = PlotConfigParser(Path("test_config.yaml"))
        plot_configs = config_parser.generate_configs()

        assert len(plot_configs) == 2
        pc1, pc2 = plot_configs

        # plot config 1
        assert pc1.title == "TTFT vs ITL"
        assert pc1.x_label == "TTFT (ms)"
        assert pc1.y_label == "ITL (ms)"
        assert pc1.type == PlotType.BOX
        assert pc1.output == Path("test_output_1")

        assert len(pc1.data) == 3  # profile run data
        prd1, prd2, prd3 = pc1.data
        assert prd1.name == "run1/concurrency32"
        assert prd2.name == "run2/concurrency32"
        assert prd3.name == "run3/concurrency32"
        for prd in pc1.data:
            assert prd.x_metric == [1, 2, 3]
            assert prd.y_metric == [1, 2, 3]

        # plot config 2
        assert pc2.title == "Num Input Token vs Num Output Token"
        assert pc2.x_label == "Input Tokens"
        assert pc2.y_label == "Output Tokens"
        assert pc2.type == PlotType.SCATTER
        assert pc2.output == Path("test_output_2")

        assert len(pc2.data) == 1  # profile run data
        prd = pc2.data[0]
        assert prd.name == "run4/concurrency1"
        assert prd.x_metric == [1, 2, 3]
        assert prd.y_metric == [1, 2, 3]
