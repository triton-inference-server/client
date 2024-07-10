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

# Profile Ranking Models with GenAI-Perf


GenAI-Perf allows you to profile ranking models compatible with Hugging Face's
[Text Embeddings Inference's re-ranker API](https://huggingface.co/docs/text-embeddings-inference/en/quick_tour#re-rankers).

## Create a Sample Rankings Input Directory

To create a sample rankings input directory, follow these steps:

Create a directory called rankings_jsonl:
```bash
mkdir rankings_jsonl
```

Inside this directory, create a JSONL file named queries.jsonl with queries data:

```bash
echo '{"text": "What was the first car ever driven?"}
{"text": "Who served as the 5th President of the United States of America?"}
{"text": "Is the Sydney Opera House located in Australia?"}
{"text": "In what state did they film Shrek 2?"}' > rankings_jsonl/queries.jsonl
```

Create another JSONL file named passages.jsonl with passages data:

```bash
echo '{"text": "Eric Anderson (born January 18, 1968) is an American sociologist and sexologist."}
{"text": "Kevin Loader is a British film and television producer."}
{"text": "Francisco Antonio Zea Juan Francisco Antonio Hilari was a Colombian journalist, botanist, diplomat, politician, and statesman who served as the 1st Vice President of Colombia."}
{"text": "Daddys Home 2 Principal photography on the film began in Massachusetts in March 2017 and it was released in the United States by Paramount Pictures on November 10, 2017. Although the film received unfavorable reviews, it has grossed over $180 million worldwide on a $69 million budget."}' > rankings_jsonl/passages.jsonl
```

## Start a Hugging Face Re-Ranker-Compatible Server
To start a Hugging Face re-ranker-compatible server, run the following commands:

```bash
model=BAAI/bge-reranker-large
revision=refs/pr/4
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.3 --model-id $model --revision $revision
```

## Run GenAI-Perf
To profile ranking models using GenAI-Perf, use the following command:

```bash
genai-perf \
    -m BAAI/bge-reranker-large \
    --service-kind openai \
    --endpoint-type rankings \
    --endpoint rerank \
    --input-file rankings_jsonl/ \
    -u localhost:8080 \
    --extra-inputs rankings:tei \
    --batch-size 2
```

This command specifies the use of Hugging Face's ranking API with `--endpoint rerank` and `--extra-inputs rankings:tei`.

Example output:

```
                          Rankings Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┓
┃            Statistic ┃  avg ┃  min ┃   max ┃   p99 ┃  p90 ┃  p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━┩
│ Request latency (ms) │ 5.48 │ 2.50 │ 23.91 │ 10.27 │ 8.34 │ 6.07 │
└──────────────────────┴──────┴──────┴───────┴───────┴──────┴──────┘
Request throughput (per sec): 180.11
```
