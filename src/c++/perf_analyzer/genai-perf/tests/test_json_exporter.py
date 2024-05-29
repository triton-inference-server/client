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

import genai_perf.parser as parser
from genai_perf.export_data.data_exporter_factory import DataExporterType
from genai_perf.export_data.json_exporter import JsonExporter


class TestJsonExporter:
    stats = {
        "request_throughput": {"unit": "requests/sec", "avg": "7"},
        "request_latency": {
            "unit": "ms",
            "avg": 1,
            "p99": 2,
            "p95": 3,
            "p90": 4,
            "p75": 5,
            "p50": 6,
            "p25": 7,
            "max": 8,
            "min": 9,
            "std": 0,
        },
        "time_to_first_token": {
            "unit": "ms",
            "avg": 11,
            "p99": 12,
            "p95": 13,
            "p90": 14,
            "p75": 15,
            "p50": 16,
            "p25": 17,
            "max": 18,
            "min": 19,
            "std": 10,
        },
        "inter_token_latency": {
            "unit": "ms",
            "avg": 21,
            "p99": 22,
            "p95": 23,
            "p90": 24,
            "p75": 25,
            "p50": 26,
            "p25": 27,
            "max": 28,
            "min": 29,
            "std": 20,
        },
        "output_token_throughput": {
            "unit": "tokens/sec",
            "avg": 31,
        },
        "output_token_throughput_per_request": {
            "unit": "tokens/sec",
            "avg": 41,
            "p99": 42,
            "p95": 43,
            "p90": 44,
            "p75": 45,
            "p50": 46,
            "p25": 47,
            "max": 48,
            "min": 49,
            "std": 40,
        },
        "num_output_token": {
            "unit": "tokens",
            "avg": 51,
            "p99": 52,
            "p95": 53,
            "p90": 54,
            "p75": 55,
            "p50": 56,
            "p25": 57,
            "max": 58,
            "min": 59,
            "std": 50,
        },
        "num_input_token": {
            "unit": "tokens",
            "avg": 61,
            "p99": 62,
            "p95": 63,
            "p90": 64,
            "p75": 65,
            "p50": 66,
            "p25": 67,
            "max": 68,
            "min": 69,
            "std": 60,
        },
    }

    expected_json_output = """
      {
        "request_throughput": {
          "unit": "requests/sec",
          "avg": "7"
          },
          "request_latency": {
              "unit": "ms",
              "avg": 1,
              "p99": 2,
              "p95": 3,
              "p90": 4,
              "p75": 5,
              "p50": 6,
              "p25": 7,
              "max": 8,
              "min": 9,
              "std": 0
          },
          "time_to_first_token": {
              "unit": "ms",
              "avg": 11,
              "p99": 12,
              "p95": 13,
              "p90": 14,
              "p75": 15,
              "p50": 16,
              "p25": 17,
              "max": 18,
              "min": 19,
              "std": 10
          },
          "inter_token_latency": {
              "unit": "ms",
              "avg": 21,
              "p99": 22,
              "p95": 23,
              "p90": 24,
              "p75": 25,
              "p50": 26,
              "p25": 27,
              "max": 28,
              "min": 29,
              "std": 20
          },
          "output_token_throughput": {
              "unit": "tokens/sec",
              "avg": 31
          },
          "output_token_throughput_per_request": {
              "unit": "tokens/sec",
              "avg": 41,
              "p99": 42,
              "p95": 43,
              "p90": 44,
              "p75": 45,
              "p50": 46,
              "p25": 47,
              "max": 48,
              "min": 49,
              "std": 40
          },
          "num_output_token": {
              "unit": "tokens",
              "avg": 51,
              "p99": 52,
              "p95": 53,
              "p90": 54,
              "p75": 55,
              "p50": 56,
              "p25": 57,
              "max": 58,
              "min": 59,
              "std": 50
          },
          "num_input_token": {
              "unit": "tokens",
              "avg": 61,
              "p99": 62,
              "p95": 63,
              "p90": 64,
              "p75": 65,
              "p50": 66,
              "p25": 67,
              "max": 68,
              "min": 69,
              "std": 60
          },
        "input_config": {
          "model": ["gpt2_vllm"],
          "formatted_model_name": "gpt2_vllm",
          "model_selection_strategy": "round_robin",
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

    def test_generate_json(self, monkeypatch) -> None:
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
        monkeypatch.setattr("sys.argv", cli_cmd)
        args, _ = parser.parse_args()
        config = {
            "type": DataExporterType.JSON,
            "stats": self.stats,
            "args": args,
            "extra_inputs": parser.get_extra_inputs_as_dict(args),
            "artifact_dir": args.artifact_dir,
        }
        json_exporter = JsonExporter(config)
        assert json_exporter._stats_and_args == json.loads(self.expected_json_output)
