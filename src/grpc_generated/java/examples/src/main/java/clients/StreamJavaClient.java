// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE


package clients;


import inference.GRPCInferenceServiceGrpc;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceStub;
import inference.GrpcService.*;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;

import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class StreamJavaClient {

	public static void main(String[] args) throws InterruptedException {

		String host = args.length > 0 ? args[0] : "localhost";
		int port = args.length > 1 ? Integer.parseInt(args[1]) : 8001;

		String model_name = "simple";
		String model_version = "";

		// Create channel for communicating with the server, gRPC stub
		ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
		// Create gRPC blocking stub for synchronous check server,
		GRPCInferenceServiceBlockingStub grpc_blocking_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);
		// Create gRPC stub for asynchronous streaming call server
		GRPCInferenceServiceStub grpc_stub = GRPCInferenceServiceGrpc.newStub(channel);

		// check server is live
		ServerLiveRequest serverLiveRequest = ServerLiveRequest.getDefaultInstance();
		ServerLiveResponse r = grpc_blocking_stub.serverLive(serverLiveRequest);
		System.out.println(r);

		// Generate the request
		ModelInferRequest.Builder request = ModelInferRequest.newBuilder();
		request.setModelName(model_name);
		request.setModelVersion(model_version);

		// Input data
		List<Integer> lst_0 = IntStream.rangeClosed(1, 16).boxed().collect(Collectors.toList());
		List<Integer> lst_1 = IntStream.rangeClosed(1, 16).boxed().collect(Collectors.toList());
		InferTensorContents.Builder input0_data = InferTensorContents.newBuilder();
		InferTensorContents.Builder input1_data = InferTensorContents.newBuilder();
		input0_data.addAllIntContents(lst_0);
		input1_data.addAllIntContents(lst_1);

		// Populate the inputs in inference request
		ModelInferRequest.InferInputTensor.Builder input0 = ModelInferRequest.InferInputTensor
				.newBuilder();
		input0.setName("INPUT0");
		input0.setDatatype("INT32");
		input0.addShape(1);
		input0.addShape(16);
		input0.setContents(input0_data);

		ModelInferRequest.InferInputTensor.Builder input1 = ModelInferRequest.InferInputTensor
				.newBuilder();
		input1.setName("INPUT1");
		input1.setDatatype("INT32");
		input1.addShape(1);
		input1.addShape(16);
		input1.setContents(input1_data);

		// request.inputs.extend([input0, input1])
		request.addInputs(0, input0);
		request.addInputs(1, input1);

		// Populate the outputs in the inference request
		ModelInferRequest.InferRequestedOutputTensor.Builder output0 = ModelInferRequest.InferRequestedOutputTensor
				.newBuilder();
		output0.setName("OUTPUT0");

		ModelInferRequest.InferRequestedOutputTensor.Builder output1 = ModelInferRequest.InferRequestedOutputTensor
				.newBuilder();
		output1.setName("OUTPUT1");

		// request.outputs.extend([output0, output1])
		request.addOutputs(0, output0);
		request.addOutputs(1, output1);

		// Define a CountDownLatch to block this asynchronous streaming request, allowing the request to complete the call.
		// The CountDownLatch is not necessary and should be removed if a callback has already been defined in the responseObserver to receive outputs.
		CountDownLatch finishLatch = new CountDownLatch(1);

		// Define streaming responseObserver and accept streaming outputs in onNext, and process the logic to complete the response in onCompleted
		StreamObserver<ModelStreamInferResponse> responseObserver = new StreamObserver<ModelStreamInferResponse>() {
			@Override
			public void onNext(ModelStreamInferResponse response) {

				// Get response outputs
				int[] op0 = toArray(response.getInferResponse().getRawOutputContentsList().get(0).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer());
				int[] op1 = toArray(response.getInferResponse().getRawOutputContentsList().get(1).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer());

			}

			@Override
			public void onCompleted() {
				System.out.println("the stream request on completed");
				// Count down the latch, signaling that this asynchronous streaming request has been completed.
				finishLatch.countDown();
			}

			@Override
			public void onError(Throwable t) {
				System.out.println("the stream request on error");
				// Also count down the latch when on error
				finishLatch.countDown();
			}
		};

		StreamObserver<ModelInferRequest> requestObserver = grpc_stub.modelStreamInfer(responseObserver);
		requestObserver.onNext(request.build());
		// Can continue sending requests...
		// requestObserver.onNext(request2.build());

		// Indicates that the request has been completed and no new requests will be sent
		requestObserver.onCompleted();

		// Wait for the asynchronous operation to complete or timeout after 60 seconds.
		if (!finishLatch.await(60, TimeUnit.SECONDS)) {
			System.out.println("request timeout.");
		}
		
	}

	public static int[] toArray(IntBuffer b) {
		if (b.hasArray()) {
			if (b.arrayOffset() == 0)
				return b.array();

			return Arrays.copyOfRange(b.array(), b.arrayOffset(), b.array().length);
		}

		b.rewind();
		int[] tmp = new int[b.remaining()];
		b.get(tmp);

		return tmp;
	}

}
