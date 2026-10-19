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

import gzip
import unittest
import zlib

from tritonclient.http import InferenceServerClient
from tritonclient.http._infer_result import InferResult
from tritonclient.utils import InferenceServerException


class TestHttpResponseDecompression(unittest.TestCase):
    _BODY = b'{"model_name":"m","outputs":[]}'

    def test_compressed_response_at_limit(self):
        body = self._BODY + b" " * (128 * 1024)
        for encoding, compress in (("gzip", gzip.compress), ("deflate", zlib.compress)):
            with self.subTest(encoding=encoding):
                result = InferResult.from_response_body(
                    compress(body),
                    content_encoding=encoding,
                    max_response_size=len(body),
                )
                self.assertEqual(result.get_response()["model_name"], "m")

    def test_compressed_response_over_limit(self):
        body = self._BODY + b" " * 1024 * 1024
        for encoding, compress in (("gzip", gzip.compress), ("deflate", zlib.compress)):
            with self.subTest(encoding=encoding):
                with self.assertRaisesRegex(
                    InferenceServerException, "exceeds max_response_size"
                ):
                    InferenceServerClient.parse_response_body(
                        compress(body),
                        content_encoding=encoding,
                        max_response_size=1024,
                    )

    def test_binary_response(self):
        header = b'{"outputs":[{"name":"o","parameters":{"binary_data_size":4}}]}'
        body = header + b"\x00\x01\x02\x03"
        for encoding, compress in (("gzip", gzip.compress), ("deflate", zlib.compress)):
            with self.subTest(encoding=encoding):
                result = InferResult.from_response_body(
                    compress(body),
                    header_length=len(header),
                    content_encoding=encoding,
                    max_response_size=len(body),
                )
                self.assertEqual(result._buffer, b"\x00\x01\x02\x03")

    def test_concatenated_gzip_members_share_limit(self):
        payload = gzip.compress(self._BODY) + gzip.compress(b" " * 1024)
        with self.assertRaisesRegex(
            InferenceServerException, "exceeds max_response_size"
        ):
            InferResult.from_response_body(
                payload,
                content_encoding="gzip",
                max_response_size=len(self._BODY),
            )


if __name__ == "__main__":
    unittest.main()
