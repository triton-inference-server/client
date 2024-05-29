# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


triton_profile_data = {
    "service_kind": "triton",
    "endpoint": "",
    "experiments": [
        {
            "experiment": {
                "mode": "concurrency",
                "value": 10,
            },
            "requests": [
                {
                    "timestamp": 1000000,
                    "request_inputs": {"text_input": "This is test"},
                    "response_timestamps": [3000000, 5000000, 8000000],
                    "response_outputs": [
                        {"text_output": "I"},
                        {"text_output": " like"},
                        {"text_output": " dogs"},
                    ],
                },
                {
                    "timestamp": 2000000,
                    "request_inputs": {"text_input": "This is test too"},
                    "response_timestamps": [4000000, 7000000, 11000000],
                    "response_outputs": [
                        {"text_output": "I"},
                        {"text_output": " don't"},
                        {"text_output": " cook food"},
                    ],
                },
            ],
        },
        {
            "experiment": {
                "mode": "request_rate",
                "value": 2.0,
            },
            "requests": [
                {
                    "timestamp": 5000000,
                    "request_inputs": {"text_input": "This is test"},
                    "response_timestamps": [7000000, 8000000, 13000000, 18000000],
                    "response_outputs": [
                        {"text_output": "cat"},
                        {"text_output": " is"},
                        {"text_output": " cool"},
                        {"text_output": " too"},
                    ],
                },
                {
                    "timestamp": 3000000,
                    "request_inputs": {"text_input": "This is test too"},
                    "response_timestamps": [6000000, 8000000, 11000000],
                    "response_outputs": [
                        {"text_output": "it's"},
                        {"text_output": " very"},
                        {"text_output": " simple work"},
                    ],
                },
            ],
        },
    ],
}
