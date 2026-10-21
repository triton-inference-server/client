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

import asyncio
import unittest
from concurrent import futures

import grpc
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as aiogrpcclient
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import InferenceServerException


class _FlakyInferService(service_pb2_grpc.GRPCInferenceServiceServicer):
    """Records every ModelInfer request and fails the first 'failures' calls."""

    def __init__(self, failures=0):
        self.failures = failures
        self.requests = []
        self.metadata = []

    def ModelInfer(self, request, context):
        self.requests.append(request)
        self.metadata.append(dict(context.invocation_metadata()))
        if len(self.requests) <= self.failures:
            context.abort(grpc.StatusCode.UNAVAILABLE, "model is reloading")
        response = service_pb2.ModelInferResponse()
        response.model_name = request.model_name
        response.id = request.id
        return response


def _make_inputs():
    data = np.arange(16, dtype=np.float32).reshape(1, 16)
    infer_input = grpcclient.InferInput("INPUT0", data.shape, "FP32")
    infer_input.set_data_from_numpy(data)
    return [infer_input]


class TestPrebuiltInferRequest(unittest.TestCase):
    def setUp(self):
        self.service = _FlakyInferService()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
            self.service, self.server
        )
        port = self.server.add_insecure_port("localhost:0")
        self.server.start()
        self.url = "localhost:{}".format(port)
        self.client = grpcclient.InferenceServerClient(self.url)

    def tearDown(self):
        self.client.close()
        self.server.stop(None)

    def test_build_matches_infer(self):
        """infer() and build_infer_request() + infer_request() send the same request"""
        kwargs = dict(
            model_version="2",
            request_id="abc",
            sequence_id=7,
            sequence_start=True,
            priority=1,
            timeout=1000,
            parameters={"custom": "value"},
        )
        self.client.infer("simple", _make_inputs(), **kwargs)
        request = self.client.build_infer_request("simple", _make_inputs(), **kwargs)
        self.client.infer_request(request)

        self.assertEqual(len(self.service.requests), 2)
        self.assertEqual(self.service.requests[0], self.service.requests[1])
        self.assertEqual(self.service.requests[1], request)

    def test_retry_reuses_request(self):
        """A failed send can be retried with the same request after the inputs are gone"""
        self.service.failures = 1
        inputs = _make_inputs()
        request = self.client.build_infer_request("simple", inputs, request_id="r1")
        built = service_pb2.ModelInferRequest()
        built.CopyFrom(request)
        del inputs

        with self.assertRaises(InferenceServerException) as ctx:
            self.client.infer_request(request)
        self.assertEqual(ctx.exception.status(), "StatusCode.UNAVAILABLE")

        result = self.client.infer_request(request)
        self.assertEqual(result.get_response().id, "r1")
        self.assertEqual(self.service.requests, [built, built])
        self.assertEqual(request, built)

    def test_infer_request_headers(self):
        request = self.client.build_infer_request("simple", _make_inputs())
        self.client.infer_request(request, headers={"x-retry-attempt": "2"})
        self.assertEqual(self.service.metadata[0].get("x-retry-attempt"), "2")

    def test_build_rejects_non_string_version(self):
        with self.assertRaises(InferenceServerException):
            grpcclient.InferenceServerClient.build_infer_request(
                "simple", _make_inputs(), model_version=1
            )


class TestPrebuiltInferRequestAio(unittest.TestCase):
    def setUp(self):
        self.service = _FlakyInferService(failures=1)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
            self.service, self.server
        )
        port = self.server.add_insecure_port("localhost:0")
        self.server.start()
        self.url = "localhost:{}".format(port)

    def tearDown(self):
        self.server.stop(None)

    def test_retry_reuses_request(self):
        async def run():
            async with aiogrpcclient.InferenceServerClient(self.url) as client:
                request = client.build_infer_request(
                    "simple", _make_inputs(), request_id="r1"
                )
                with self.assertRaises(InferenceServerException):
                    await client.infer_request(request)
                result = await client.infer_request(request)
                await client.infer("simple", _make_inputs(), request_id="r1")
                return request, result

        request, result = asyncio.run(run())
        self.assertEqual(result.get_response().id, "r1")
        self.assertEqual(len(self.service.requests), 3)
        for received in self.service.requests:
            self.assertEqual(received, request)


if __name__ == "__main__":
    unittest.main()
