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

from genai_perf import parser
from genai_perf.export_data.console_exporter import ConsoleExporter
from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.metrics import LLMMetrics, Metrics, Statistics


class TestConsoleExporter:

    def test_streaming_llm_output(self, monkeypatch, capsys) -> None:
        argv = [
            "genai-perf",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
            "--streaming",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        metrics = LLMMetrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
            time_to_first_tokens=[7, 8, 9],
            inter_token_latencies=[10, 11, 12],
            output_token_throughputs=[456],
            output_sequence_lengths=[1, 2, 3],
            input_sequence_lengths=[5, 6, 7],
        )
        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        expected_content = (
            "                                LLM Metrics                                 \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓\n"
            "┃                Statistic ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩\n"
            "│     Request latency (ms) │  5.00 │  4.00 │  6.00 │  5.98 │  5.80 │  5.50 │\n"
            "│ Time to first token (ms) │  8.00 │  7.00 │  9.00 │  8.98 │  8.80 │  8.50 │\n"
            "│ Inter token latency (ms) │ 11.00 │ 10.00 │ 12.00 │ 11.98 │ 11.80 │ 11.50 │\n"
            "│   Output sequence length │  2.00 │  1.00 │  3.00 │  2.98 │  2.80 │  2.50 │\n"
            "│    Input sequence length │  6.00 │  5.00 │  7.00 │  6.98 │  6.80 │  6.50 │\n"
            "└──────────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┘\n"
            "Request throughput (requests/sec): 123.00\n"
            "Output token throughput (tokens/sec): 456.00\n"
        )

        returned_data = capsys.readouterr().out
        assert returned_data == expected_content

    def test_nonstreaming_llm_output(self, monkeypatch, capsys) -> None:
        argv = [
            "genai-perf",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        metrics = LLMMetrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
            time_to_first_tokens=[4, 5, 6],  # same as request_latency
            inter_token_latencies=[],  # no ITL
            output_token_throughputs=[456],
            output_sequence_lengths=[1, 2, 3],
            input_sequence_lengths=[5, 6, 7],
        )
        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        # No TTFT and ITL in the output
        expected_content = (
            "                            LLM Metrics                             \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓\n"
            "┃              Statistic ┃  avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩\n"
            "│   Request latency (ms) │ 5.00 │ 4.00 │ 6.00 │ 5.98 │ 5.80 │ 5.50 │\n"
            "│ Output sequence length │ 2.00 │ 1.00 │ 3.00 │ 2.98 │ 2.80 │ 2.50 │\n"
            "│  Input sequence length │ 6.00 │ 5.00 │ 7.00 │ 6.98 │ 6.80 │ 6.50 │\n"
            "└────────────────────────┴──────┴──────┴──────┴──────┴──────┴──────┘\n"
            "Request throughput (requests/sec): 123.00\n"
            "Output token throughput (tokens/sec): 456.00\n"
        )

        returned_data = capsys.readouterr().out
        assert returned_data == expected_content

    def test_embedding_output(self, monkeypatch, capsys) -> None:
        argv = [
            "genai-perf",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "embeddings",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        metrics = Metrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
        )
        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        expected_content = (
            "                        Embedding Metrics                         \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓\n"
            "┃            Statistic ┃  avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩\n"
            "│ Request latency (ms) │ 5.00 │ 4.00 │ 6.00 │ 5.98 │ 5.80 │ 5.50 │\n"
            "└──────────────────────┴──────┴──────┴──────┴──────┴──────┴──────┘\n"
            "Request throughput (requests/sec): 123.00\n"
        )

        returned_data = capsys.readouterr().out
        assert returned_data == expected_content
