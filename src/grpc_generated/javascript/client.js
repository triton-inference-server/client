// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const util = require('util');
const PROTO_PATH = __dirname + '/proto/grpc_service.proto';
const PROTO_IMPORT_PATH = __dirname + '/proto'

const GRPCServicePackageDefinition = protoLoader.loadSync(PROTO_PATH, {
    includeDirs: [PROTO_IMPORT_PATH],
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true
});

function BufferToInt32Array(buf) {
    newArray = new Int32Array(buf.byteLength / 4)
    dv = new DataView(buf.buffer)
    for (let i = 0; i < newArray.length; i++) {
        newArray[i] = dv.getInt32(buf.byteOffset + (i * 4), littleEndian = true);
    }
    return newArray
}

const inference = grpc.loadPackageDefinition(GRPCServicePackageDefinition).inference;

async function main() {
    const argv = process.argv.slice(2)
    const host = argv.length > 0 ? argv[0] : "localhost";
    const port = argv.length > 1 ? argv[1] : "8001";

    const model_name = "simple";
    const model_version = "";
    const batch_size = 1
    const dimension = 16

    const client = new inference.GRPCInferenceService(host + ':' + port, grpc.credentials.createInsecure());


    const serverLive = util.promisify(client.serverLive).bind(client);
    const serverReady = util.promisify(client.serverReady).bind(client);
    const modelMetadata = util.promisify(client.modelMetadata).bind(client);
    const modelInfer = util.promisify(client.modelInfer).bind(client);

    serverLiveResponse = await serverLive({});
    console.log("Triton Health - Live:", serverLiveResponse.live);
    if(!serverLiveResponse.live) {
        console.error("Triton is not Live")
        process.exit(1);
    }

    serverReadyResponse = await serverReady({});
    console.log("Triton Health - Ready:", serverReadyResponse.ready);
    if(!serverReadyResponse.ready) {
        console.error("Triton is not Ready")
        process.exit(1);
    }

    modelMetadataResponse = await modelMetadata({ name: model_name, version: model_version });
    console.log("\nModel Info:", modelMetadataResponse)

    // Input Data
    // Use input dimension [batch_size, dimension]

    const input0_data = Array(batch_size * dimension).fill().map((element, index) => index)
    const input1_data = Array(batch_size * dimension).fill().map((element, index) => index)

    const input0 = {
        name: "INPUT0",
        datatype: "INT32",
        shape: [batch_size, dimension],
        contents: {
            int_contents: input0_data
        }
    }

    const input1 = {
        name: "INPUT1",
        datatype: "INT32",
        shape: [batch_size, dimension],
        contents: {
            int_contents: input1_data
        }
    }

    const modelInferRequest = {
        model_name: model_name,
        model_version: model_version,
        inputs: [input0, input1],
        outputs: [
            { name: "OUTPUT0" },
            { name: "OUTPUT1" }
        ]
    }

    const outputs = await modelInfer(modelInferRequest);

    output_data = outputs.raw_output_contents.map(BufferToInt32Array);

    console.log("\nChecking Inference Output")
    console.log("-------------------------")
    for (let i = 0; i < dimension; i++) {
        console.log(input0_data[i] + " + " + input1_data[i] + " = " + output_data[0][i])
        console.log(input0_data[i] + " - " + input1_data[i] + " = " + output_data[1][i])
        if (((input0_data[i] + input1_data[i]) != output_data[0][i]) ||
                ((input0_data[i] - input1_data[i]) != output_data[1][i])) {
            console.error("Unexpected results encountered")
            process.exit(1);
        }
    }

}

main()
