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

    serverReadyResponse = await serverReady({});
    console.log("Triton Health - Ready:", serverReadyResponse.ready);

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
    }

}

main()