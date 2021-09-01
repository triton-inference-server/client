// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package triton.client.examples;

import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Lists;
import triton.client.InferInput;
import triton.client.InferRequestedOutput;
import triton.client.InferResult;
import triton.client.InferenceServerClient;
import triton.client.pojo.DataType;

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
