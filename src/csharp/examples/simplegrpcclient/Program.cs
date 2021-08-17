/**
// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
 */

using Inference;
using System;
using Grpc.Net.Client;
using System.Linq;
using System.Threading.Tasks;

namespace simplegrpcclient
{
    class Program
    {
        static async Task Main(string[] args)
        {
			var host = args.Length > 0 ? args[0] : "localhost";
			int port = args.Length > 1 ? int.Parse(args[1]) : 8001;

            var model_name = "simple";
			var model_version = "";

			// # Create gRPC stub for communicating with the server
			using var channel = GrpcChannel.ForAddress($"http://{host}:{port}");
			var grpcClient = new GRPCInferenceService.GRPCInferenceServiceClient(channel);

			// check server is live
			var serverLiveRequest = new ServerLiveRequest();
			var r = await grpcClient.ServerLiveAsync(serverLiveRequest);
			Console.WriteLine(r.Live);

			// # Generate the request
			var request = new ModelInferRequest();
			request.ModelName = model_name;
			request.ModelVersion = model_version;

			// # Input data
			var lst_0 = Enumerable.Range(1, 16).ToArray();
			var lst_1 = Enumerable.Range(1, 16).ToArray();
			var input0_data = new InferTensorContents();
			var input1_data = new InferTensorContents();
			input0_data.IntContents.AddRange(lst_0);
			input1_data.IntContents.AddRange(lst_1);

			// # Populate the inputs in inference request
			var input0 = new ModelInferRequest.Types.InferInputTensor();
			input0.Name = "INPUT0";
			input0.Datatype = "INT32";
			input0.Shape.Add(1);
			input0.Shape.Add(16);
			input0.Contents = input0_data;

			var input1 = new ModelInferRequest.Types.InferInputTensor();
			input0.Name = "INPUT1";
			input0.Datatype = "INT32";
			input0.Shape.Add(1);
			input0.Shape.Add(16);
			input0.Contents = input1_data;

			// request.inputs.extend([input0, input1])
			request.Inputs.Add(input0);
			request.Inputs.Add(input1);

			// # Populate the outputs in the inference request
			var output0 = new ModelInferRequest.Types.InferRequestedOutputTensor();
			output0.Name = "OUTPUT0";

			var output1 = new ModelInferRequest.Types.InferRequestedOutputTensor();
			output1.Name = "OUTPUT1";

			// request.outputs.extend([output0, output1])
			request.Outputs.Add(output0);
			request.Outputs.Add(output1);

			var response = await grpcClient.ModelInferAsync(request);
			Console.WriteLine(response);

			// Get the response outputs
			int[] op0 = response.Outputs[0].Contents.IntContents.ToArray();
			int[] op1 = response.Outputs[1].Contents.IntContents.ToArray();

			// Validate response outputs
			for (int i = 0; i < op0.Length; i++)
			{
				Console.WriteLine($"{lst_0[i]} + {lst_1[i]} = {op0[i]}");
				Console.WriteLine($"{lst_0[i]} + {lst_1[i]} = {op1[i]}");

				if (op0[i] != (lst_0[i] + lst_1[i]))
				{
					Console.WriteLine("OUTPUT0 contains incorrect sum");
					Environment.Exit(-1);
				}

				if (op1[i] != (lst_0[i] - lst_1[i]))
				{
					Console.WriteLine("OUTPUT1 contains incorrect difference");
					Environment.Exit(-1);
				}
			}
		}
    }
}
