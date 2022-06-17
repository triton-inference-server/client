<!--
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Triton Java API

This is a Triton Java API contributed by Alibaba Cloud PAI Team.
It's based on Triton's HTTP/REST Protocols and for both easy of use and performance.

This Java API mimics Triton's official Python API. It has similar classes and methods.
- `triton.client.InferInput` describes each input to model. 
- `triton.client.InferRequestedOutput` describes each output from model.
- `triton.client.InferenceServerClient` is the main inference class.

Currently the Java API supports only a subset of the entire Triton
protocol. 

A minimal example would be like:

```java
package triton.client.example;

import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Lists;
import triton.client.InferInput;
import triton.client.InferRequestedOutput;
import triton.client.InferResult;
import triton.client.InferenceServerClient;
import triton.client.pojo.DataType;

public class MinExample {
    public static void main(String[] args) throws Exception {
        boolean isBinary = true;
        InferInput inputIds = new InferInput("input_ids", new long[] {1L, 32}, DataType.INT32);
        int[] inputIdsData = new int[32];
        Arrays.fill(inputIdsData, 1); // fill with some data.
        inputIds.setData(inputIdsData, isBinary);

        InferInput inputMask = new InferInput("input_mask", new long[] {1, 32}, DataType.INT32);
        int[] inputMaskData = new int[32];
        Arrays.fill(inputMaskData, 1);
        inputMask.setData(inputMaskData, isBinary);

        InferInput segmentIds = new InferInput("segment_ids", new long[] {1, 32}, DataType.INT32);
        int[] segmentIdsData = new int[32];
        Arrays.fill(segmentIdsData, 0);
        segmentIds.setData(segmentIdsData, isBinary);
        List<InferInput> inputs = Lists.newArrayList(inputIds, inputMask, segmentIds);
        List<InferRequestedOutput> outputs = Lists.newArrayList(new InferRequestedOutput("logits", isBinary));

        InferenceServerClient client = new InferenceServerClient("0.0.0.0:8000", 5000, 5000);
        InferResult result = client.infer("roberta", inputs, outputs);
        float[] logits = result.getOutputAsFloat("logits");
        System.out.println(Arrays.toString(logits));
    }
}
```

## Supported and Unsupported Java client features

### Supported Java client features:
HTTP client is supported with limited capabilty. Currently supported:
- Synchronous inference requests

GRPC has very limited support. Please see [grpc generated Java client](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/java) for details 

### *Unsupported* Java client features:
GRPC client:
- Full feature Java GRPC client and corresponding tests

HTTP client:
1. Asynchronous inference requests
2. Streaming inference requests
3. SSL or HTTPS protocol communications
4. Requesting/Recieving Server Metadata Information
5. Requesting/Recieving Model Metadata Information
6. Requesting/Recieving Model Inference Statistics
7. Sending inference requests using Shared Memory (System, GPU)
8. Sending multiple synchronous inferences on server
9. [Extensions](https://github.com/triton-inference-server/server/tree/main/docs/protocol) are not supported

## Building Java Examples

The Java examples can be found in [examples folder](src/main/java/triton/client/examples/). To compile these examples,
simply run:

```bash
$ cd client/src/java 
$ mvn clean install -Ddir=examples
```

Then you will be able to find the examples located in your target folder: `examples` and the 
compiled jar at `target/java-api-0.0.1.jar`.
