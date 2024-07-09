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

import json
#from yaml import safe_load

import genai_perf.parser as parser
from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.export_data.yaml_exporter import YamlExporter

class TestYamlExporter:
    def test_generate_yaml(self, monkeypatch) -> None:
        cli_cmd = [
            "genai-perf",
            "-m",
            "gpt2_vllm",
            "--backend",
            "vllm",
            "--streaming",
            "--extra-inputs",
            "max_tokens:256",
            "--extra-inputs",
            "ignore_eos:true",
        ]
        config = ExporterConfig()
        monkeypatch.setattr("sys.argv", cli_cmd)
        args, _ = parser.parse_args()
        config.stats = self.stats
        config.args = args
        config.extra_inputs = parser.get_extra_inputs_as_dict(args)
        config.artifact_dir = args.artifact_dir
        yaml_exporter = YamlExporter(config)
        assert yaml_exporter._stats_and_args == self.expected_yaml_output

    stats = {}
    expected_yaml_output = {'name': 'Silenthand Olleander', 'race': 'Human','traits': ['ONE_HAND', 'ONE_EYE']}