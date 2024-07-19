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

# Profile Vision-Language Models with GenAI-Perf

GenAI-Perf allows you to profile Vison-Language Models (VLM) running on
[OpenAI Chat Completions API](https://platform.openai.com/docs/guides/chat-completions)-compatible server
by sending [multi-modal contents](https://platform.openai.com/docs/guides/vision) to the server.
Currently, you can send multi-modal contents with GenAI-Perf using the following two approaches:
1. The synthetic data generation approach, where GenAI-Perf generates the multi-modal data for you.
2. The Bring Your Own Data (BYOD) approach, where you provide GenAI-Perf with the data to send.


## Approach 1: Synthetic Multi-Modal Data Generation

GenAI-Perf can generate synthetic multi-modal data such as texts or images using
the parameters provide by the user through CLI.

```bash
genai-perf profile \
    -m llava-hf/llava-v1.6-mistral-7b-hf \
    --service-kind openai \
    --endpoint-type vision \
    --image-width-mean 512 \
    --image-width-stddev 30 \
    --image-height-mean 512 \
    --image-height-stddev 30 \
    --image-format png \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --streaming
```

> [!Note]
> Under the hood, the GenAI-Perf generates synthetic images using few source images
> under the `genai_perf/llm_inputs/source_images` directory.
> If you would like to add/remove/edit the source images,
> you can do so by directly editing the source images under the directory.
> GenAI-Perf will pickup the images under the directory automatically when
> generating the synthetic images.


## Approach 2: Bring Your Own Data (BYOD)

Instead of letting GenAI-Perf create the synthetic data,
you can also provide GenAI-Perf with your own data using
[`--input-file`](../README.md#--input-file-path) CLI option.
The file needs to be in JSONL format and should contain both the prompt and
the filepath to the image to send.

For instance, an example of input file would look something as following:
```bash
// input.jsonl
{"text_input": "What is in this image?", "image": "path/to/image1.png"}
{"text_input": "What is the color of the dog?", "image": "path/to/image2.jpeg"}
{"text_input": "Describe the scene in the picture.", "image": "path/to/image3.png"}
...
```

After you create the file, you can run GenAI-Perf using the following command:

```bash
genai-perf profile \
    -m llava-hf/llava-v1.6-mistral-7b-hf \
    --service-kind openai \
    --endpoint-type vision \
    --input-file input.jsonl \
    --streaming
```

Running GenAI-Perf using either approach will give you an example output that
looks like below:

```bash
                                         LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to first token (ms) │   321.05 │   291.30 │   537.07 │   497.88 │   318.46 │   317.35 │
│ Inter token latency (ms) │    12.28 │    11.44 │    12.88 │    12.87 │    12.81 │    12.53 │
│     Request latency (ms) │ 1,866.23 │ 1,044.70 │ 2,832.22 │ 2,779.63 │ 2,534.64 │ 2,054.03 │
│   Output sequence length │   126.68 │    59.00 │   204.00 │   200.58 │   177.80 │   147.50 │
│    Input sequence length │   100.00 │   100.00 │   100.00 │   100.00 │   100.00 │   100.00 │
└──────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
Output token throughput (per sec): 67.40
Request throughput (per sec): 0.53
```
