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

package com.nvidia.triton.contrib.examples;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import com.nvidia.triton.contrib.InferInput;
import com.nvidia.triton.contrib.InferRequestedOutput;
import com.nvidia.triton.contrib.InferResult;
import com.nvidia.triton.contrib.InferenceServerClient;
import com.nvidia.triton.contrib.InferenceServerClient.InferArguments;
import com.nvidia.triton.contrib.endpoint.FixedEndpoint;
import com.nvidia.triton.contrib.pojo.DataType;

/**
 * @author xiafei.qiuxf
 * @date 2021/5/7
 */
public class TritonInsideEASExample {
    public static void main(String[] args) throws Exception {
        // Prepare input.
        InferInput input = new InferInput("input", new long[] {1L, 299L, 299L, 3L}, DataType.FP32);
        int size = 1 * 299 * 299 * 3;
        float[] data = new float[size];
        Random rand = new Random(43);
        for (int i = 0; i < size; i++) {
            data[i] = rand.nextFloat();
        }
        input.setData(data, true);

        // Prepare output.
        InferRequestedOutput output = new InferRequestedOutput("InceptionV3/Predictions/Softmax", true);

        // The service's Internet/Intranet Endpoint in EASR could be used to send request to. Since the endpoint has a
        // fixed URL, FixedEndpoint is used here.
        // The URL may look like:
        //   - 1111111111111111.ap-south-1.pai-eas.aliyuncs.com/api/predict/<eas_service_name>
        // or
        //   - 1111111111111111.vpc.ap-south-1.pai-eas.aliyuncs.com/api/predict/<eas_service_name>
        FixedEndpoint endpoint = new FixedEndpoint("xxxxxx.ap-south-1.pai-eas.aliyuncs.com/api/predict/test_triton");
        try (InferenceServerClient client = new InferenceServerClient(endpoint, 50000, 50000)) {

            InferArguments inferArg = new InferArguments("inception_graphdef",
                Collections.singletonList(input),
                Collections.singletonList(output))
                // To send request to your service, set token of your EAS service's here.
                .setHeader("Authorization", "<EAS Token>");

            InferResult result = client.infer(inferArg);
            for (String name : result.getOutputs()) {
                float[] outputAsFloat = result.getOutputAsFloat(name);
                System.out.println("--------- " + name);
                System.out.println(Arrays.toString(outputAsFloat));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
