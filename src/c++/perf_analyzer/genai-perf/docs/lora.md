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

# Profile Multiple LoRA Adapters
GenAI-Perf allows you to profile multiple LoRA adapters on top of a base model.

## Select LoRA Adapters
To do this, list multiple adapters after the model name option `-m`:

```bash
genai-perf -m lora_adapter1 lora_adapter2 lora_adapter3
```

## Choose a Strategy for Selecting Models
When profiling with multiple models, you can specify how the models should be
assigned to prompts using the `--model-selection-strategy` option:

```bash
genai-perf \
    profile \
    -m lora_adapter1 lora_adapter2 lora_adapter3 \
    --model-selection-strategy round_robin
```

This setup will cycle through the lora_adapter1, lora_adapter2, and
lora_adapter3 models in a round-robin manner for each prompt.

For more details on additional options and configurations, refer to the
[Command Line Options section](../README.md#command-line-options) in the README.