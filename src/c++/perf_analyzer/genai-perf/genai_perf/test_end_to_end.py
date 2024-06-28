import itertools
import os
import subprocess
import sys

# How to run:
#   test_end_to_end.py <target>
#     Where target is "nim_chat" or "nim_completions" or "vllm_openai" or "triton_tensorrtllm"
#
# For all cases but vllm_openai, it assumes that the server will be on port 9999
#
# This script will run a sweep of all combinations of values in the testing matrix
# by appending those options on to the genai-pa base command
#


testing_matrix = [
    ["--concurrency 1", "--concurrency 32", "--request-rate 1", "--request-rate 32"],
    ["--streaming", ""],
]

base_commands = {
    "nim_chat": "genai-perf -s 999 -p 20000 -m llama-2-7b-chat -u http://localhost:9999 --service-kind openai --endpoint-type chat",
    "nim_completions": "genai-perf -s 999 -p 20000 -m llama-2-7b -u http://localhost:9999 --service-kind openai --endpoint-type completions",
    "nim_vision": "genai-perf -s 999 -p 20000 -m llava16-mistral-7b -u http://localhost:9999 --service-kind openai --endpoint-type vision",
    "vllm_openai": "genai-perf -s 999 -p 20000 -m mistralai/Mistral-7B-v0.1 --service-kind openai --endpoint-type chat",
    "triton_tensorrtllm": "genai-perf -s 999 -p 20000 -m llama-2-7b -u 0.0.0.0:9999 --service-kind triton --backend tensorrtllm",
    "triton_vllm": "genai-perf -s 999 -p 20000 -m gpt2_vllm --service-kind triton --backend vllm",
}
testname = ""

if len(sys.argv) == 2:
    # The second element in sys.argv is the input string
    testname = sys.argv[1]
else:
    options = " ".join(base_commands.keys())
    print(f"This script requires exactly one argument. It must be one of {options}")
    exit(1)

base_command = base_commands[testname]


def rename_files(files: list, substr: str) -> None:
    for f in files:
        name, ext = f.rsplit(".", 1)
        # Insert the substring and reassemble the filename
        new_filename = f"{testname}__{name}__{substr}.{ext}"
        try:
            os.rename(f, new_filename)
        except FileNotFoundError:
            # Just ignore the error, since if PA failed these files may not exist
            pass


def print_summary():
    # FIXME -- print out a few basic metrics. Maybe from the csv?
    pass


def sanity_check():
    # FIXME -- add in some sanity checking? Throughput isn't 0?
    pass


# Loop through all combinations
for combination in itertools.product(*testing_matrix):
    options_string = " ".join(combination)
    command_with_options = f"{base_command} {options_string}"
    command_array = command_with_options.split()

    file_options_string = "__".join(combination)
    file_options_string = file_options_string.replace(" ", "")
    file_options_string = file_options_string.replace("-", "")
    output_file = testname + "__" + file_options_string + ".log"

    with open(output_file, "w") as outfile:
        print(f"\nCMD: {command_with_options}")
        print(f"  Output log is {output_file}")
        proc = subprocess.run(command_array, stdout=outfile, stderr=subprocess.STDOUT)

        if proc.returncode != 0:
            print(f"  Command failed with return code: {proc.returncode}")
        else:
            print(f"  Command executed successfully!")
            print_summary()
            sanity_check()

        files = [
            "profile_export.json",
            "profile_export_genai_pa.csv",
            "llm_inputs.json",
        ]
        rename_files(files, file_options_string)
