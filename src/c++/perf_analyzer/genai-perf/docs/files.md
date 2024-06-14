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
the files generated  by GenAi-Perf.

## Directory Structure

After running GenAi-Perf, your file tree should contain the following:

```
genai-perf/
├── artifacts/
│   ├── data/
│   └── images/
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
- input_sequence_lengths_vs_output_sequence_lengths.gzip: This contains data on
the input sequence lengths versus the output sequence lengths for each request.
- request_latency.gzip: This contains the latency for each request.
- time_to_first_token.gzip: This contains the time to first token for each request.
- token_to_token_vs_output_position.gzip: This contains the time from one token
generation to the next versus the position of the output token for each token.
- ttft_vs_input_sequence_lengths.gzip: This contains the time to first token
versus the input sequence length for each request.

##### JSON Files

- llm_inputs.json: This contains the input prompts provided to the LLM during testing.
- profile_export.json: This is provided by Perf Analyzer and contains the timestamps
for each event in the lifecycle of each request. This is low-level data used to calculate
metrics by GenAi-Perf.

##### CSV File

- profile_export_genai_perf.csv: A CSV of the output tables printed
in the GenAi-Perf output. These may have more detail than the printed tables.

#### Images Subdirectory

The images subdirectory contains visual representations of the performance
data. All images are in both HTML and JPEG formats.

##### HTML and JPEG Files
- input_sequence_lengths_vs_output_sequence_lengths: A heat map showing the
relationship between input and generated tokens.
- request_latency: A box plot showing request latency.
- time_to_first_token: A box plot showing time to first token.
- token_to_token_vs_output_position: A scatterplot showing token-to-token
time versus output token position.
- ttft_vs_input_sequence_lengths: A scatterplot showing token-to-token time
versus the input sequence lengths.

## Usage Instructions

To use the generated files, navigate to the artifacts/data directory. Then,
the next steps depend on the file format you wish to work with.

### GZIP Files

The GZIP files contain Parquet files with calculated data, which can be read
with Pandas in Python. For example, you can create a dataframe with these files:

```
import pandas
df = pandas.read_partquet(path_to_file)`
```

You can then use Pandas to work with the data.

```
print(df.head())     # See the first few rows of the data.
print(df.describe()) # Get summary statistics for the data
```

### CSV and JSON Files
Open .csv and .json files with spreadsheet or JSON parsing tools for structured
data analysis. These can also be read via a text editor, like Vim.

### HTML Files

View .html visualizations in a web browser for interactive data exploration.

### JPEG Files

Use an image software to open .jpeg images for static visual representations.
