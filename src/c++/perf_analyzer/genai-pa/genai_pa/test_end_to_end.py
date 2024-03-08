import subprocess
import os
import shutil
import itertools

import sys
    

# Define the options for option1 and option2
testing_matrix = [
    ["--concurrency 1", "--concurrency 32", "--request-rate 1", "--request-rate 32"],
    ["--streaming", ""]
]

base_commands = {
    "nim_chat": "genai-pa -u http://localhost:9999 -m llama-2-7b-chat --output-format openai_chat_completions --service-kind openai --endpoint v1/chat/completions",
    "nim_completions": "genai-pa -u http://localhost:9999 -m llama-2-7b --output-format openai_completions --service-kind openai --endpoint v1/completions",
    "vllm_openai": "genai-pa -m mistralai/Mistral-7B-v0.1 --output-format openai_chat_completions --service-kind openai --endpoint v1/chat/completions"
}
testname = ""

if len(sys.argv) == 2:
    # The second element in sys.argv is the input string
    testname = sys.argv[1]
else:
    options = ' '.join(base_commands.keys())
    print(f"This script requires exactly one argument. It must be one of {options}")
    exit(1)

base_command = base_commands[testname]

# Loop through all combinations
for combination in itertools.product(*testing_matrix):

    options_string = ' '.join(combination)
    command_with_options = f"{base_command} {options_string}"
    command_array = command_with_options.split()

    file_options_string = '__'.join(combination)
    file_options_string = file_options_string.replace(" ", "")
    file_options_string = file_options_string.replace("-", "")
    output_file = testname + "__" + file_options_string + ".log"

    with open(output_file, 'w') as outfile:
        print(f"Running {command_with_options}")
        subprocess.run(command_array, stdout=outfile, stderr=subprocess.STDOUT)
        # FIXME return value

        files = ["profile_export.json", "profile_export_genai_pa.csv", "llm_inputs.json"]
        for f in files:
            name, ext = f.rsplit('.', 1)
            # Insert the substring and reassemble the filename
            new_filename = f"{testname}__{name}__{file_options_string}.{ext}"
            try:
                os.rename(f, new_filename)
            except FileNotFoundError:
                print(f"  Warning: {f} does not exist to be renamed") 

