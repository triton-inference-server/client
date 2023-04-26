# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    import grpc
    from tritonclient.grpc import model_config_pb2
    from tritonclient.grpc import service_pb2
    from tritonclient.grpc import service_pb2_grpc
    from ._client import InferenceServerClient
    from ._client import KeepAliveOptions
    from ._infer_input import InferInput
    from ._infer_result import InferResult
    from ._requested_output import InferRequestedOutput
    from ._client import MAX_GRPC_MESSAGE_SIZE
    from tritonclient.utils import *
    from ._utils import raise_error, raise_error_grpc
    from .._plugin import InferenceServerClientPlugin
except ModuleNotFoundError as error:
    raise RuntimeError(
        'The installation does not include grpc support. '
        'Specify \'grpc\' or \'all\' while installing the tritonclient '
        'package to include the support') from error

from packaging import version
import warnings

# Check grpc version and issue warnings if grpc version is known to have
# memory leakage issue.
if version.parse(grpc.__version__) >= version.parse('1.43.0') and version.parse(
        grpc.__version__) < version.parse('1.51.1'):
    warnings.warn(
        f"Imported version of grpc is {grpc.__version__}. There is a memory "
        "leak in certain Python GRPC versions (1.43.0 to be specific). Please "
        "use versions <1.43.0 or >=1.51.1 to avoid leaks "
        "(see https://github.com/grpc/grpc/issues/28513).")
