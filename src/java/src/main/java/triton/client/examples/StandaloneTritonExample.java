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

import triton.client.InferInput;
import triton.client.InferRequestedOutput;
import triton.client.InferResult;
import triton.client.InferenceServerClient;
import triton.client.InferenceServerClient.InferArguments;
import triton.client.endpoint.FixedEndpoint;
import triton.client.pojo.DataType;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.AtomicDouble;

/**
 * Do inference via a stand-alone (which is not hosted inside EAS) triton server.
 *
 * @author xiafei.qiuxf
 * @date 2021/5/7
 */
public class StandaloneTritonExample {
     private static List<InferInput> getInputs(String name, boolean isBinary) {
         switch (name) {
             case "roberta": {
                 InferInput inputIds = new InferInput("input_ids", new long[] {1L, 32}, DataType.INT32);
                 int[] inputIdsData = new int[] {101, 2923, 966, 2533, 6579, 743, 4638, 8024, 3300, 1259, 6163, 743, 1726, 1343,
                     6843,
                     2157, 782, 8024, 3688, 2353, 6574, 7030, 679, 7231, 511, 2207, 1779, 4638, 1377, 809, 2897, 102};
                 inputIds.setData(inputIdsData, isBinary);

                 InferInput inputMask = new InferInput("input_mask", new long[] {1, 32}, DataType.INT32);
                 int[] inputMaskData = new int[32];
                 Arrays.fill(inputMaskData, 1);
                 inputMask.setData(inputMaskData, isBinary);

                 InferInput segmentIds = new InferInput("segment_ids", new long[] {1, 32}, DataType.INT32);
                 int[] segmentIdsData = new int[32];
                 Arrays.fill(segmentIdsData, 0);
                 segmentIds.setData(segmentIdsData, isBinary);
                 return Lists.newArrayList(inputIds, inputMask, segmentIds);
             }
             default:
                 throw new UnsupportedOperationException(name);
         }
     }
     private static List<InferRequestedOutput> getOutputs(String name, boolean isBinary) {
         switch (name) {
             case "roberta":
                 return Lists.newArrayList(new InferRequestedOutput("logits", isBinary));
             default:
                 throw new UnsupportedOperationException(name);
         }
     }

    public static void main(String[] args) throws Exception {
        final String name = args[0];
        final int nThreads = Integer.parseInt(args[1]);

        System.out.printf("Testing %s with %d threads.%n", name, nThreads);

        List<InferInput> inputs = getInputs(name, true);
        List<InferRequestedOutput> outputs = getOutputs(name, true);

        List<Thread> threads = Lists.newArrayList();
        AtomicDouble totalQps = new AtomicDouble(0);
        AtomicDouble avgLatency = new AtomicDouble(0);
        for (int t = 0; t < nThreads; t++) {
            Thread thread = new Thread(() -> {
                long tid = Thread.currentThread().getId();
                // For a stand-alone triton server, FixedEndpoint is used to connected to it.
                FixedEndpoint endpoint = new FixedEndpoint("0.0.0.0:8000");
                try (InferenceServerClient client = new InferenceServerClient(endpoint, 5000, 5000)) {
                    InferArguments inferArg = new InferArguments("roberta", inputs, outputs);
                    final int N = 1000;
                    final int GAP = 100;
                    long start = System.currentTimeMillis();
                    long lastGapStart = start;
                    for (int i = 0; i < N; i++) {
                        InferResult result = client.infer(inferArg);
                        if ((i + 1) % GAP == 0) {
                            long now = System.currentTimeMillis();
                            long gapElapsedMs = now - lastGapStart;
                            double latency = 1.0 * gapElapsedMs / GAP;
                            System.out.printf("[%d][GAP] Requests: %d, avg latency(ms): %.2f%n", tid, i + 1, latency);
                            lastGapStart = now;
                        }
                    }
                    long totalMs = System.currentTimeMillis() - start;
                    double latency = 1.0 * totalMs / N;
                    double qps = 1000.0 * N / totalMs;
                    System.out.printf("[%d][TOTAL] avg latency(ms): %.2f, qps: %.2f%n", tid, latency, qps);
                    totalQps.addAndGet(qps);
                    avgLatency.addAndGet(latency);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
            thread.start();
            threads.add(thread);
        }
        for (Thread thread : threads) {
            thread.join();
        }
        System.out.println("==================================");
        System.out.printf("[ALL]         QPS: %.2f\n", totalQps.get());
        System.out.printf("[ALL] Latency(ms): %.2f\n", avgLatency.get() / nThreads);
        System.out.println("==================================");
    }

}
