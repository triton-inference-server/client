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

# Generated File Structures

## Overview

This document serves as a guide to understanding the structure and contents of
the generated files by GenAi-Perf. The files are organized as follows.

## Directory Structure

```
genai-perf/
├── artifacts/
│   ├── data/
│   └── images/
└── docs/
```

## File Types
Within the artifacts and docs directories, several file types are generated,
including .gzip, .csv, .json, .html, and .jpeg. Below is a detailed
explanation of each file and its purpose.

### Artifacts Directory

#### Data Subdirectory

The data subdirectory contains the raw and processed performance data files.

##### GZIP Files

- all_data.gzip: Aggregated performance data from all collected metrics.
- input_tokens_vs_generated_tokens.gzip: Data on the number of input tokens
versus the number of generated tokens.
- request_latency.gzip: Latency measurements for each request to the LLM.
- time_to_first_token.gzip: Time metrics from input to the generation of the
first token.
- token_to_token_vs_output_position.gzip: Time from one token generation to
the next, plotted against the position of the output token.
- ttft_vs_input_tokens.gzip: Time to the first token in relation to the
number of input tokens.

##### JSON Files

- llm_inputs.json: Input prompts provided to the LLM during testing.
- profile_export.json: Exported performance profile in JSON format.
- profile_json.json: *Confirm with team. This looks similar to profile_export.json but a bit difference.*

##### CSV File

- profile_export_genai_perf.csv: A CSV of the output tables printed
in the GenAi-Perf output. These may have more detail than the printed tables.

#### Images Subdirectory

The images subdirectory contains visual representations of the performance
data. All images are in both HTML and JPEG formats.

##### HTML and JPEG Files
- input_tokens_vs_generated_tokens: A heat map showing the relationship
between input and generated tokens.
- request_latency: A box plot showing request latency.
- time_to_first_token: A box plot showing time to first token.
- token_to_token_vs_output_position: A scatterplot showing token-to-token
time versus output token position.
- ttft_vs_input_tokens: A scatterplot showing token-to-token time versus the
number of input tokens.

## Usage Instructions

To use the generated files:
1. Navigate to the artifacts/data directory to access the raw data files for
detailed analysis.
2. Decompress any GZIP files with your choice of software. In Linux, you can
use `gzip -d "filepath"` to unzip the file at "filepath."
3. Open .csv and .json files with spreadsheet or JSON parsing tools for structured
data analysis.
4. View .html visualizations in a web browser for interactive data exploration.
5. Use an image software to open .jpeg images for static visual representations.