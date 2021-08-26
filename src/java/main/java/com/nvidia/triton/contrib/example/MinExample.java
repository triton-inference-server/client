package com.nvidia.triton.contrib.example;

import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Lists;
import com.nvidia.triton.contrib.InferInput;
import com.nvidia.triton.contrib.InferRequestedOutput;
import com.nvidia.triton.contrib.InferResult;
import com.nvidia.triton.contrib.InferenceServerClient;
import com.nvidia.triton.contrib.pojo.DataType;

/**
 * @author xiafei.qiuxf
 * @date 2021/7/28
 */
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
    }
}
