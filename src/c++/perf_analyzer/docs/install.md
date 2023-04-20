<!--
Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Recommended Installation Method

## Triton SDK Container

The recommended way to "install" Perf Analyzer is to run the pre-built
executable from within the Triton SDK docker container available on the
[NVIDIA GPU Cloud Catalog](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver).
As long as the SDK container has its network exposed to the address and port of
the inference server, Perf Analyzer will be able to run.

```bash
export RELEASE=<yy.mm> # e.g. to use the release from the end of February of 2023, do `export RELEASE=23.02`

docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# inside container
perf_analyzer -m <model>
```

# Alternative Installation Methods

- [Pip](#pip)
- [Build from Source](#build-from-source)

## Pip

```bash
pip install tritonclient

perf_analyzer -m <model>
```

**Warning**: If any runtime dependencies are missing, Perf Analyzer will produce
errors showing which ones are missing. You will need to manually install them.

## Build from Source

The Triton SDK container is used for building, so some build and runtime
dependencies are already installed.

```bash
export RELEASE=<yy.mm> # e.g. to use the release from the end of February of 2023, do `export RELEASE=23.02`

docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# inside container
# prep installing newer version of cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null ; apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'

# install build/runtime dependencies
apt update ; apt install -y cmake-data=3.21.1-0kitware1ubuntu20.04.1 cmake=3.21.1-0kitware1ubuntu20.04.1 libcurl4-openssl-dev rapidjson-dev

rm -rf client ; git clone --depth 1 https://github.com/triton-inference-server/client

mkdir client/build ; cd client/build

cmake -DTRITON_ENABLE_PERF_ANALYZER=ON ..

make -j8 cc-clients

perf_analyzer -m <model>
```

- To enable
  [CUDA shared memory](input_data.md#shared-memory), add
  `-DTRITON_ENABLE_GPU=ON` to the `cmake` command.
- To enable
  [C API mode](benchmarking.md#benchmarking-triton-directly-via-c-api), add
  `-DTRITON_ENABLE_PERF_ANALYZER_C_API=ON` to the `cmake` command.
- To enable [TorchServe backend](benchmarking.md#benchmarking-torchserve), add
  `-DTRITON_ENABLE_PERF_ANALYZER_TS=ON` to the `cmake` command.
- To enable
  [Tensorflow Serving backend](benchmarking.md#benchmarking-tensorflow-serving),
  add `-DTRITON_ENABLE_PERF_ANALYZER_TFS=ON` to the `cmake` command.
