# Triton Java SDK

This is a Triton Java SDK contributed by Alibaba Cloud PAI Team.
It's based on Triton's HTTP/REST Protocols and for both easy of use and performance.

This Java SDK mimics Triton's official Python SDK. It has similar classes and methods.
- `com.nvidia.triton.contrib.InferInput` describes each input to model. 
- `com.nvidia.triton.contrib.InferRequestedOutput` describes each output from model.
- `com.nvidia.triton.contrib.InferenceServerClient` is the main inference class.

A minimal example would be like:

```java
package com.nvidia.triton.contrib.example;

import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Lists;
import com.nvidia.triton.contrib.InferInput;
import com.nvidia.triton.contrib.InferRequestedOutput;
import com.nvidia.triton.contrib.InferResult;
import com.nvidia.triton.contrib.InferenceServerClient;
import com.nvidia.triton.contrib.pojo.DataType;

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

See more examples in `src/main/java/com/nvidia/triton/contrib/example/`.