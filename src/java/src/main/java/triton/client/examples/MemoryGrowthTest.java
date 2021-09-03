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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.google.common.collect.Lists;
import triton.client.InferInput;
import triton.client.InferRequestedOutput;
import triton.client.InferResult;
import triton.client.InferenceServerClient;
import triton.client.pojo.DataType;

public class MemoryGrowthTest {
  public static void main(String[] args) throws Exception
  {
    // Number of inferences to run.
    int repetitions = 100;
    for (int i = 0; i < repetitions; i++) {
      // Create the data for the input tensor.
      boolean isBinary = true;
      InferInput input0 = new InferInput("INPUT0", new long[] {1L, 16}, DataType.INT32);
      List<Integer> lst_0 = IntStream.rangeClosed(1, 16).boxed().collect(Collectors.toList());
      int[] input0_data = lst_0.stream().mapToInt(j -> j).toArray();
      input0.setData(input0_data, isBinary);

      List<InferInput> inputs = Lists.newArrayList(input0);
      List<InferRequestedOutput> outputs =
          Lists.newArrayList(new InferRequestedOutput("OUTPUT0", isBinary));

      InferenceServerClient client = new InferenceServerClient("0.0.0.0:8000", 5000, 5000);
      InferResult result = client.infer("custom_identity_int32", inputs, outputs);

      // Get the output arrays from the results and verify
      int[] output0 = result.getOutputAsInt("OUTPUT0");

      if (!Arrays.equals(input0_data, output0)) {
        System.out.println("incorrect output");
        System.exit(1);
      }
    }
    System.exit(0);
  }
}
