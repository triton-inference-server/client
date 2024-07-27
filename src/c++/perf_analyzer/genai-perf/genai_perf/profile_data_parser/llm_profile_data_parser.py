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

import json
from itertools import tee
from pathlib import Path
from typing import Dict, List, Tuple

from genai_perf.metrics import LLMMetrics, Metrics
from genai_perf.profile_data_parser.profile_data_parser import (
    ProfileDataParser,
    ResponseFormat,
)
from genai_perf.tokenizer import Tokenizer
from genai_perf.utils import load_json_str, remove_sse_prefix


class LLMProfileDataParser(ProfileDataParser):
    """A class that calculates and aggregates all the LLM performance statistics
    across the Perf Analyzer profile results.

    The LLMProfileDataParser class parses profile export JSON file, collects the
    core LLM performance metrics, and calculates summary statistics for each
    different Perf Analyzer runs/experiments.

    Example:

      >>> ... # run Perf Analyzer with concurrency level 10
      >>>
      >>> from transformers import AutoTokenizer
      >>>
      >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
      >>> pd = LLMProfileDataParser(
      >>>     filename="profile_export.json",
      >>>     tokenizer=tokenizer,
      >>> )
      >>> stats = pd.get_statistics(infer_mode="concurrency", level=10)
      >>>
      >>> print(stats)  # output: Statistics(avg_time_to_first_token=...)
      >>> stats.pretty_print()  # Output: time_to_first_token_s: ...
    """

    def __init__(
        self,
        filename: Path,
        tokenizer: Tokenizer,
    ) -> None:
        self._tokenizer = tokenizer
        super().__init__(filename)

    def _parse_requests(self, requests: dict) -> Metrics:
        """Parse each requests in profile export data to extract key metrics."""
        min_req_timestamp, max_res_timestamp = float("inf"), 0
        request_latencies = []
        time_to_first_tokens = []
        inter_token_latencies = []
        output_token_throughputs_per_request = []
        input_sequence_lengths = []
        output_sequence_lengths = []
        chunked_inter_token_latencies = []

        for request in requests:
            req_timestamp = request["timestamp"]
            req_inputs = request["request_inputs"]
            res_timestamps = request["response_timestamps"]
            res_outputs = request["response_outputs"]

            self._preprocess_response(res_timestamps, res_outputs)

            # Skip requests with empty response. This happens sometimes when the
            # model returns a single response with empty string.
            if not res_timestamps:
                continue

            # track entire benchmark duration
            min_req_timestamp = min(min_req_timestamp, req_timestamp)
            max_res_timestamp = max(max_res_timestamp, res_timestamps[-1])

            # request latencies
            req_latency_ns = res_timestamps[-1] - req_timestamp
            request_latencies.append(req_latency_ns)  # nanosec
            req_latency_s = req_latency_ns / 1e9  # sec

            # time to first token
            ttft = res_timestamps[0] - req_timestamp
            time_to_first_tokens.append(ttft)

            # number of input tokens
            input_seq_len = self._get_input_token_count(req_inputs)
            input_sequence_lengths.append(input_seq_len)

            # output token throughput per request
            output_token_counts, total_output_token = self._get_output_token_counts(
                res_outputs
            )
            output_token_throughputs_per_request.append(
                total_output_token / req_latency_s
            )
            output_sequence_lengths.append(total_output_token)

            # inter token latencies
            if total_output_token > 1:
                inter_token_latency = (req_latency_ns - ttft) / (total_output_token - 1)
                inter_token_latencies.append(round(inter_token_latency))

            # The new ITL calculation above loses all token-level ITL information
            # and as a result breaks ITL vs token position visualization. Keep
            # the old version of inter token latency as a WAR to preserve the
            # visualization.
            chunked_inter_token_latency = []
            for (t1, _), (t2, n2) in self._pairwise(
                zip(res_timestamps, output_token_counts)
            ):
                # TMA-1676: handle empty first/last responses
                # if the latter response has zero token (e.g. empty string),
                # then set it default to one for the sake of inter token latency
                # calculation and to avoid divide by zero.
                num_token = 1 if n2 == 0 else n2
                chunked_inter_token_latency.append(round((t2 - t1) / num_token))
            chunked_inter_token_latencies.append(chunked_inter_token_latency)

        # request & output token throughput
        benchmark_duration = (max_res_timestamp - min_req_timestamp) / 1e9  # nanosec
        request_throughputs = [len(requests) / benchmark_duration]
        output_token_throughputs = [sum(output_sequence_lengths) / benchmark_duration]

        return LLMMetrics(
            request_throughputs,
            request_latencies,
            time_to_first_tokens,
            inter_token_latencies,
            output_token_throughputs,
            output_token_throughputs_per_request,
            output_sequence_lengths,
            input_sequence_lengths,
            chunked_inter_token_latencies,
        )

    def _pairwise(self, iterable):
        """Generate pairs of consecutive elements from the given iterable."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def _preprocess_response(
        self, res_timestamps: List[int], res_outputs: List[Dict[str, str]]
    ) -> None:
        """Helper function to preprocess responses of a request."""
        if self._service_kind == "openai":
            # PA sometimes receives multiple SSE responses at once (as a single
            # response). Handle these responses by merging into a single response.
            for i in range(len(res_outputs)):
                response = res_outputs[i]["response"]
                responses = response.strip().split("\n\n")
                if len(responses) > 1:
                    merged_response = load_json_str(remove_sse_prefix(responses[0]))
                    if (
                        merged_response["choices"][0]["delta"].get("content", None)
                        is None
                    ):
                        merged_response["choices"][0]["delta"]["content"] = ""
                    for r in responses[1:]:
                        text = self._extract_openai_text_output(r)
                        merged_response["choices"][0]["delta"]["content"] += text

                    res_outputs[i] = {"response": json.dumps(merged_response)}

            # Remove responses without any content
            indices_to_remove = []
            for idx, out in enumerate(res_outputs):
                if self._is_openai_empty_response(out["response"]):
                    indices_to_remove.append(idx)
            indices_to_remove.sort(reverse=True)
            for index in indices_to_remove:
                res_timestamps.pop(index)
                res_outputs.pop(index)

    def _get_input_token_count(self, req_inputs: dict) -> int:
        """Deserialize the request input and return tokenized inputs."""
        if self._service_kind == "triton":
            input_text = req_inputs["text_input"]
        elif self._service_kind == "openai":
            input_text = self._get_openai_input_text(req_inputs)
        else:
            raise ValueError(f"Unknown service kind: '{self._service_kind}'.")

        return len(self._tokenizer.encode(input_text))

    def _get_openai_input_text(self, req_inputs: dict) -> str:
        """Tokenize the OpenAI request input texts."""
        payload = load_json_str(req_inputs["payload"])
        if self._response_format == ResponseFormat.OPENAI_CHAT_COMPLETIONS:
            return payload["messages"][0]["content"]
        elif self._response_format == ResponseFormat.OPENAI_COMPLETIONS:
            return payload["prompt"]
        elif self._response_format == ResponseFormat.OPENAI_VISION:
            content = payload["messages"][0]["content"]
            return " ".join(c["text"] for c in content if c["type"] == "text")
        else:
            raise ValueError(
                "Failed to parse OpenAI request input in profile export file."
            )

    def _get_output_token_counts(
        self, res_outputs: List[Dict]
    ) -> Tuple[List[int], int]:
        """Return response-level token counts and total token count."""
        if self._service_kind == "triton":
            output_texts = self._get_triton_output_tokens(res_outputs)
        elif self._service_kind == "openai":
            output_texts = self._get_openai_output_tokens(res_outputs)
        else:
            raise ValueError(f"Unknown service kind: '{self._service_kind}'.")

        full_text_token_count = len(self._tokenizer.encode("".join(output_texts)))

        output_tokens = self._get_response_output_tokens(output_texts)
        output_token_counts = list(map(len, output_tokens))
        return output_token_counts, full_text_token_count

    def _get_triton_output_tokens(self, res_outputs: List[Dict]) -> List[str]:
        """Return a list of Triton response texts."""
        return [r["text_output"] for r in res_outputs]

    def _get_openai_output_tokens(self, res_outputs: List[Dict]) -> List[str]:
        """Return a list of OpenAI response texts."""
        output_texts = []
        for output in res_outputs:
            text = self._extract_openai_text_output(output["response"])
            output_texts.append(text)
        return output_texts

    def _get_response_output_tokens(self, output_texts: List[str]) -> List[List[int]]:
        """Return a list of response output tokens."""
        # Exclamation mark trick forces the llama tokenization to consistently
        # start each output with a specific token which allows us to safely skip
        # the first token of every tokenized output and get only the ones that
        # are returned by the model
        encodings = self._tokenizer(["!" + txt for txt in output_texts])
        return [out[1:] for out in encodings.data["input_ids"]]

    def _extract_openai_text_output(self, response: str) -> str:
        """Extracts text/content of the OpenAI response object."""
        response = remove_sse_prefix(response)

        if response == "[DONE]":
            return ""

        data = load_json_str(response)
        completions = data["choices"][0]

        text_output = ""
        if "object" not in data:
            # FIXME: TPA-47 workaround for vLLM not following OpenAI Completions
            # API specification when streaming, missing 'object' field:
            # https://platform.openai.com/docs/api-reference/completions
            text_output = completions.get("text", "")
        elif data["object"] == "text_completion":  # legacy
            text_output = completions.get("text", "")
        elif data["object"] == "chat.completion":  # non-streaming
            text_output = completions["message"].get("content", "")
        elif data["object"] == "chat.completion.chunk":  # streaming
            text_output = completions["delta"].get("content", "")
        else:
            obj_type = data["object"]
            raise ValueError(f"Unknown OpenAI response object type '{obj_type}'.")
        return text_output

    def _is_openai_empty_response(self, response: str) -> bool:
        """Returns true if the response is an openai response with no content (or empty content)"""
        text = self._extract_openai_text_output(response)
        if text:
            return False
        return True
