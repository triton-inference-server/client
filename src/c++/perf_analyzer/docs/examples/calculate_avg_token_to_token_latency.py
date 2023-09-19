import json

with open("profile_export.json") as f:
    # example json demonstrating format:
    # https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/examples/decoupled_output_file.json
    requests = json.load(f)["experiments"][0]["requests"]
    latencies = []
    for request in requests:
        prev_response = request["response_timestamps"][0]
        for response in request["response_timestamps"][1:]:
            latencies.append(response - prev_response)
            prev_response = response
    avg_latency_s = sum(latencies) / len(latencies) / 1000000000

    print("Average token-to-token latency: " + str(avg_latency_s) + " s")
