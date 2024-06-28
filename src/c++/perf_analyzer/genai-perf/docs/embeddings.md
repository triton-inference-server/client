<!--
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Profiling Embeddings Models with GenAI-Perf

GenAI-Perf allows you to profile embedding models running on an
[OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)-compatible server.

## Creating a Sample Embeddings Input File

To create a sample embeddings input file, use the following command:

```bash
echo '{"text": "What was the first car ever driven?"}
{"text": "Who served as the 5th President of the United States of America?"}
{"text": "Is the Sydney Opera House located in Australia?"}
{"text": "In what state did they film Shrek 2?"}' > embeddings.jsonl
```

This will generate a file named embeddings.jsonl with the following content:
```jsonl
{"text": "What was the first car ever driven?"}
{"text": "Who served as the 5th President of the United States of America?"}
{"text": "Is the Sydney Opera House located in Australia?"}
{"text": "In what state did they film Shrek 2?"}
```

## Starting an OpenAI Embeddings-Compatible Server
To start an OpenAI embeddings-compatible server, run the following command:
```bash
docker run -it --net=host --rm --gpus=all vllm/vllm-openai:latest --model intfloat/e5-mistral-7b-instruct --dtype float16 --max-model-len 1024
```

## Running GenAI-Perf
To profile embeddings models using GenAI-Perf, use the following command:

```bash
genai-perf \
-m intfloat/e5-mistral-7b-instruct \
--service-kind openai \
--endpoint-type embeddings \
--batch-size 2 \
--input-file embeddings.jsonl
```

This will use default values for optional arguments. You can also pass in
additional arguments with the `--extra-inputs` [flag](../README.md#input-options).
For example, you could use this command:

```bash
genai-perf \
-m intfloat/e5-mistral-7b-instruct \
--service-kind openai \
--endpoint-type embeddings \
--extra-inputs user:sample_user
```

Example output:

```
                          Embeddings Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Statistic            ┃ avg   ┃ min   ┃ max    ┃ p99   ┃ p90   ┃ p75   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ Request latency (ms) │ 42.21 │ 28.18 │ 318.61 │ 56.50 │ 49.21 │ 43.07 │
└──────────────────────┴───────┴───────┴────────┴───────┴───────┴───────┘
Request throughput (per sec): 23.63
```