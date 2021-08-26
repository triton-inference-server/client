package com.nvidia.triton.contrib.example;

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
