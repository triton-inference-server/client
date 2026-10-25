#!/usr/bin/env python3

# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import unittest

import numpy as np
from google.protobuf.internal import api_implementation
from tritonclient.grpc import InferInput

try:
    import resource
except ImportError:  # Windows
    resource = None

_SHM_KEYS = (
    "shared_memory_region",
    "shared_memory_byte_size",
    "shared_memory_offset",
)


class TestGrpcInferInput(unittest.TestCase):
    def test_set_data_from_numpy_leaves_parameters_empty(self):
        infer_input = InferInput("INPUT0", [1, 4], "FP32")
        data = np.arange(4, dtype=np.float32).reshape(1, 4)

        for _ in range(3):
            infer_input.set_data_from_numpy(data)

        self.assertEqual(len(infer_input._get_tensor().parameters), 0)
        self.assertEqual(infer_input._get_content(), data.tobytes())

    def test_set_data_from_numpy_clears_shared_memory_parameters(self):
        for offset in (0, 16):
            with self.subTest(offset=offset):
                infer_input = InferInput("INPUT0", [1, 4], "FP32")
                infer_input.set_shared_memory("region", 16, offset=offset)
                infer_input.set_data_from_numpy(np.zeros((1, 4), dtype=np.float32))

                parameters = infer_input._get_tensor().parameters
                for key in _SHM_KEYS:
                    self.assertNotIn(key, parameters)

    def test_set_data_from_numpy_keeps_unrelated_parameters(self):
        infer_input = InferInput("INPUT0", [1, 4], "FP32")
        infer_input._get_tensor().parameters["custom"].string_param = "value"

        infer_input.set_data_from_numpy(np.zeros((1, 4), dtype=np.float32))

        parameters = infer_input._get_tensor().parameters
        self.assertEqual(list(parameters.keys()), ["custom"])
        self.assertEqual(parameters["custom"].string_param, "value")

    @unittest.skipIf(resource is None, "needs the resource module")
    @unittest.skipUnless(
        api_implementation.Type() == "upb", "leak is specific to the upb backend"
    )
    def test_reused_input_does_not_grow_memory(self):
        # Before the fix each call leaked ~190 B of upb arena memory, so
        # 200k calls grew peak RSS by ~38 MB.
        infer_input = InferInput("INPUT0", [1, 4], "FP32")
        data = np.zeros((1, 4), dtype=np.float32)
        for _ in range(10000):
            infer_input.set_data_from_numpy(data)

        rss_unit = 1 if sys.platform == "darwin" else 1024
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
        for _ in range(200000):
            infer_input.set_data_from_numpy(data)
        growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit - before

        self.assertLess(growth, 8 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
