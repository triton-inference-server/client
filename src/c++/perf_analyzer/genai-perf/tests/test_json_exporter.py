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
from pathlib import PosixPath

from genai_perf.export_data.json_exporter import JsonExporter
from genai_perf.llm_inputs.llm_inputs import OutputFormat, PromptSource


class TestJsonExporter:
    stats = {
        "request_throughput": {"unit": "request per sec", "avg": "12.222026283281268"},
        "request_latency": {
            "unit": "ns",
            "avg": "81694262.3143508",
            "p99": "101977365.62",
            "p95": "98139925.99999999",
            "p90": "93780014.0",
            "p75": "87927373.5",
            "p50": "81988918.0",
            "p25": "76813208.5",
            "max": "180952000",
            "min": "37899188",
            "std": "11531917.69933574",
        },
        "time_to_first_token": {
            "unit": "ns",
            "avg": "20757771.22095672",
            "p99": "29047316.52",
            "p95": "27985451.7",
            "p90": "25826318.599999998",
            "p75": "22925623.5",
            "p50": "20901523.0",
            "p25": "18963453.5",
            "max": "44816559",
            "min": "10820287",
            "std": "4220849.59851806",
        },
        "inter_token_latency": {
            "unit": "ns",
            "avg": "3860101.1100274306",
            "p99": "5263723.220000001",
            "p95": "5097144.25",
            "p90": "5036146.1",
            "p75": "4063794.25",
            "p50": "3988754.5",
            "p25": "3866785.25",
            "max": "104056336",
            "min": "619744",
            "std": "1539307.1111368367",
        },
        "output_token_throughput": {
            "unit": "tokens per sec",
            "avg": "217.1010500160076",
        },
        "output_token_throughput_per_request": {
            "unit": "tokens per sec",
            "avg": "221.26925312949",
            "p99": "320.769756629708",
            "p95": "297.8740207961169",
            "p90": "277.716572304638",
            "p75": "237.54220501729532",
            "p50": "214.42163973683745",
            "p25": "198.40160210092222",
            "max": "362.2711821112862",
            "min": "79.15736875418017",
            "std": "36.9078750549695",
        },
        "num_output_token": {
            "unit": "tokens",
            "avg": "17.763097949886106",
            "p99": "22.0",
            "p95": "20.0",
            "p90": "20.0",
            "p75": "19.0",
            "p50": "18.0",
            "p25": "17.0",
            "max": "25",
            "min": "3",
            "std": "1.7190889855652978",
        },
        "num_input_token": {
            "unit": "tokens",
            "avg": "550.0523917995445",
            "p99": "552.24",
            "p95": "550.0",
            "p90": "550.0",
            "p75": "550.0",
            "p50": "550.0",
            "p25": "550.0",
            "max": "553",
            "min": "550",
            "std": "0.34348803287081675",
        },
    }
    args = {
        "model": "gpt2_vllm",
        "backend": OutputFormat.VLLM,
        "endpoint": None,
        "endpoint_type": None,
        "service_kind": "triton",
        "streaming": True,
        "u": None,
        "extra_inputs": ["max_tokens:256", "ignore_eos:true"],
        "input_dataset": None,
        "input_file": None,
        "num_prompts": 100,
        "output_tokens_mean": -1,
        "output_tokens_mean_deterministic": False,
        "output_tokens_stddev": 0,
        "random_seed": 0,
        "synthetic_input_tokens_mean": 550,
        "synthetic_input_tokens_stddev": 0,
        "concurrency": 1,
        "measurement_interval": 10000,
        "request_rate": None,
        "stability_percentage": 999,
        "generate_plots": False,
        "profile_export_file": PosixPath(
            "artifacts/gpt2_vllm-triton-vllm-concurrency1/profile_export.json"
        ),
        "artifact_dir": PosixPath("artifacts/gpt2_vllm-triton-vllm-concurrency1"),
        "tokenizer": "hf-internal-testing/llama-tokenizer",
        "verbose": False,
        "subcommand": None,
        "prompt_source": PromptSource.SYNTHETIC,
        "output_format": OutputFormat.VLLM,
        "func": None,
    }
    extra_inputs = {"max_tokens": 256, "ignore_eos": True}

    expected_json_output = """
      {
        "request_throughput": {
          "unit": "request per sec",
          "avg": "12.222026283281268"
        },
        "request_latency": {
          "unit": "ns",
          "avg": "81694262.3143508",
          "p99": "101977365.62",
          "p95": "98139925.99999999",
          "p90": "93780014.0",
          "p75": "87927373.5",
          "p50": "81988918.0",
          "p25": "76813208.5",
          "max": "180952000",
          "min": "37899188",
          "std": "11531917.69933574"
        },
        "time_to_first_token": {
          "unit": "ns",
          "avg": "20757771.22095672",
          "p99": "29047316.52",
          "p95": "27985451.7",
          "p90": "25826318.599999998",
          "p75": "22925623.5",
          "p50": "20901523.0",
          "p25": "18963453.5",
          "max": "44816559",
          "min": "10820287",
          "std": "4220849.59851806"
        },
        "inter_token_latency": {
          "unit": "ns",
          "avg": "3860101.1100274306",
          "p99": "5263723.220000001",
          "p95": "5097144.25",
          "p90": "5036146.1",
          "p75": "4063794.25",
          "p50": "3988754.5",
          "p25": "3866785.25",
          "max": "104056336",
          "min": "619744",
          "std": "1539307.1111368367"
        },
        "output_token_throughput": {
          "unit": "tokens per sec",
          "avg": "217.1010500160076"
        },
        "output_token_throughput_per_request": {
          "unit": "tokens per sec",
          "avg": "221.26925312949",
          "p99": "320.769756629708",
          "p95": "297.8740207961169",
          "p90": "277.716572304638",
          "p75": "237.54220501729532",
          "p50": "214.42163973683745",
          "p25": "198.40160210092222",
          "max": "362.2711821112862",
          "min": "79.15736875418017",
          "std": "36.9078750549695"
        },
        "num_output_token": {
          "unit": "tokens",
          "avg": "17.763097949886106",
          "p99": "22.0",
          "p95": "20.0",
          "p90": "20.0",
          "p75": "19.0",
          "p50": "18.0",
          "p25": "17.0",
          "max": "25",
          "min": "3",
          "std": "1.7190889855652978"
        },
        "num_input_token": {
          "unit": "tokens",
          "avg": "550.0523917995445",
          "p99": "552.24",
          "p95": "550.0",
          "p90": "550.0",
          "p75": "550.0",
          "p50": "550.0",
          "p25": "550.0",
          "max": "553",
          "min": "550",
          "std": "0.34348803287081675"
        },
        "input_config": {
          "model": "gpt2_vllm",
          "backend": "vllm",
          "endpoint": null,
          "endpoint_type": null,
          "service_kind": "triton",
          "streaming": true,
          "u": null,
          "input_dataset": null,
          "input_file": null,
          "num_prompts": 100,
          "output_tokens_mean": -1,
          "output_tokens_mean_deterministic": false,
          "output_tokens_stddev": 0,
          "random_seed": 0,
          "synthetic_input_tokens_mean": 550,
          "synthetic_input_tokens_stddev": 0,
          "concurrency": 1,
          "measurement_interval": 10000,
          "request_rate": null,
          "stability_percentage": 999,
          "generate_plots": false,
          "profile_export_file": "artifacts/gpt2_vllm-triton-vllm-concurrency1/profile_export.json",
          "artifact_dir": "artifacts/gpt2_vllm-triton-vllm-concurrency1",
          "tokenizer": "hf-internal-testing/llama-tokenizer",
          "verbose": false,
          "subcommand": null,
          "prompt_source": "synthetic",
          "extra_inputs": {
            "max_tokens": 256,
            "ignore_eos": true
          }
        }
      }
    """

    def test_generate_json(self) -> None:
        json_exporter = JsonExporter(self.stats, self.args, self.extra_inputs)
        assert json_exporter._stats_and_args == json.loads(self.expected_json_output)
