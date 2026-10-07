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

import importlib.util
import os
import unittest
from types import SimpleNamespace

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

# Load InferResult from the source file. The generated gRPC protobuf stubs
# are produced at wheel-build time and are not present in a source checkout.
_INFER_RESULT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "tritonclient", "grpc", "_infer_result.py"
    )
)
_spec = importlib.util.spec_from_file_location(
    "tritonclient_grpc_infer_result_under_test", _INFER_RESULT_PATH
)
_infer_result_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer_result_mod)
InferResult = _infer_result_mod.InferResult


def _output_tensor_class():
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "infer_output.proto"
    file_proto.syntax = "proto3"
    msg = file_proto.message_type.add()
    msg.name = "InferOutputTensor"
    optional = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    for number, name, ftype, label in (
        (1, "name", descriptor_pb2.FieldDescriptorProto.TYPE_STRING, optional),
        (2, "datatype", descriptor_pb2.FieldDescriptorProto.TYPE_STRING, optional),
        (
            3,
            "shape",
            descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
            descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        ),
    ):
        field = msg.field.add()
        field.name = name
        field.number = number
        field.type = ftype
        field.label = label
    file_desc = descriptor_pool.DescriptorPool().Add(file_proto)
    return message_factory.GetMessageClass(
        file_desc.message_types_by_name["InferOutputTensor"]
    )


_OutputTensor = _output_tensor_class()


def _infer_result(name="OUTPUT0", datatype="FP32", shape=(1, 16)):
    output = _OutputTensor(name=name, datatype=datatype)
    output.shape.extend(shape)
    return InferResult(SimpleNamespace(outputs=[output])), output


class TestGrpcInferResult(unittest.TestCase):
    """gRPC InferResult output retrieval"""

    def test_get_output_as_json_returns_dict(self):
        """as_json=True must return the named output as a dict, not None."""
        infer_result, _ = _infer_result()
        got = infer_result.get_output("OUTPUT0", as_json=True)
        self.assertIsInstance(got, dict)
        self.assertEqual(got["name"], "OUTPUT0")
        self.assertEqual(got["datatype"], "FP32")
        # MessageToJson represents int64 values as strings.
        self.assertEqual(got["shape"], ["1", "16"])

    def test_get_output_returns_protobuf_by_default(self):
        infer_result, output = _infer_result()
        got = infer_result.get_output("OUTPUT0")
        self.assertIs(got, output)

    def test_get_output_unknown_name_returns_none(self):
        infer_result, _ = _infer_result()
        self.assertIsNone(infer_result.get_output("MISSING"))
        self.assertIsNone(infer_result.get_output("MISSING", as_json=True))


if __name__ == "__main__":
    unittest.main()
