#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
import struct

import grpc
import rapidjson as json
from google.protobuf.json_format import MessageToJson
from tritonclient.grpc import service_pb2, service_pb2_grpc

from .._client import InferenceServerClientBase
from .._request import Request
from ._infer_result import InferResult
from ._infer_stream import _InferStream, _RequestIterator
from ._utils import (
    _get_inference_request,
    _grpc_compression_type,
    get_cancelled_error,
    get_error_grpc,
    raise_error,
    raise_error_grpc,
)

# Should be kept consistent with the value specified in
# src/core/constants.h, which specifies MAX_GRPC_MESSAGE_SIZE
# as INT32_MAX.
INT32_MAX = 2 ** (struct.Struct("i").size * 8 - 1) - 1
MAX_GRPC_MESSAGE_SIZE = INT32_MAX


class KeepAliveOptions:
    """A KeepAliveOptions object is used to encapsulate GRPC KeepAlive
    related parameters for initiating an InferenceServerclient object.

    See the https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    documentation for more information.

    Parameters
    ----------
    keepalive_time_ms: int
        The period (in milliseconds) after which a keepalive ping is sent on
        the transport. Default is INT32_MAX.

    keepalive_timeout_ms: int
        The period (in milliseconds) the sender of the keepalive ping waits
        for an acknowledgement. If it does not receive an acknowledgment
        within this time, it will close the connection. Default is 20000
        (20 seconds).

    keepalive_permit_without_calls: bool
        Allows keepalive pings to be sent even if there are no calls in flight.
        Default is False.

    http2_max_pings_without_data: int
        The maximum number of pings that can be sent when there is no
        data/header frame to be sent. gRPC Core will not continue sending
        pings if we run over the limit. Setting it to 0 allows sending pings
        without such a restriction. Default is 2.

    """

    def __init__(
        self,
        keepalive_time_ms=INT32_MAX,
        keepalive_timeout_ms=20000,
        keepalive_permit_without_calls=False,
        http2_max_pings_without_data=2,
    ):
        self.keepalive_time_ms = keepalive_time_ms
        self.keepalive_timeout_ms = keepalive_timeout_ms
        self.keepalive_permit_without_calls = keepalive_permit_without_calls
        self.http2_max_pings_without_data = http2_max_pings_without_data


class CallContext:
    """This is a wrapper over grpc future call which can be used to
    issue cancellation on an ongoing RPC call.

    Parameters
    ----------
    grpc_future : gRPC.Future
        The future tracking gRPC call.
    """

    def __init__(self, grpc_future):
        self.__grpc_future = grpc_future

    def cancel(self):
        """Issues cancellation on the underlying request."""
        self.__grpc_future.cancel()


class InferenceServerClient(InferenceServerClientBase):
    """An InferenceServerClient object is used to perform any kind of
    communication with the InferenceServer using gRPC protocol. Most
    of the methods are thread-safe except start_stream, stop_stream
    and async_stream_infer. Accessing a client stream with different
    threads will cause undefined behavior.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. 'localhost:8001'.

    verbose : bool
        If True generate verbose output. Default value is False.

    ssl : bool
        If True use SSL encrypted secure channel. Default is False.

    root_certificates : str
        File holding the PEM-encoded root certificates as a byte
        string, or None to retrieve them from a default location
        chosen by gRPC runtime. The option is ignored if `ssl`
        is False. Default is None.

    private_key : str
        File holding the PEM-encoded private key as a byte string,
        or None if no private key should be used. The option is
        ignored if `ssl` is False. Default is None.

    certificate_chain : str
        File holding PEM-encoded certificate chain as a byte string
        to use or None if no certificate chain should be used. The
        option is ignored if `ssl` is False. Default is None.

    creds: grpc.ChannelCredentials
        A grpc.ChannelCredentials object to use for the connection.
        The ssl, root_certificates, private_key and certificate_chain
        options will be ignored when using this option. Default is None.

    keepalive_options: KeepAliveOptions
        Object encapsulating various GRPC KeepAlive options. See
        the class definition for more information. Default is None.

    channel_args: List[Tuple]
        List of Tuple pairs ("key", value) to be passed directly to the GRPC
        channel as the channel_arguments. If this argument is provided, it is
        expected the channel arguments are correct and complete, and the
        keepalive_options parameter will be ignored since the corresponding
        keepalive channel arguments can be set directly in this parameter. See
        https://grpc.github.io/grpc/python/glossary.html#term-channel_arguments
        for more details. Default is None.

    Raises
    ------
    Exception
        If unable to create a client.
    """

    def __init__(
        self,
        url,
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None,
        creds=None,
        keepalive_options=None,
        channel_args=None,
    ):
        super().__init__()
        # Explicitly check "is not None" here to support passing an empty
        # list to specify setting no channel arguments.
        if channel_args is not None:
            channel_opt = channel_args
        else:
            # Use GRPC KeepAlive client defaults if unspecified
            if not keepalive_options:
                keepalive_options = KeepAliveOptions()

            # To specify custom channel_opt, see the channel_args parameter.
            channel_opt = [
                ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_SIZE),
                ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_SIZE),
                ("grpc.keepalive_time_ms", keepalive_options.keepalive_time_ms),
                ("grpc.keepalive_timeout_ms", keepalive_options.keepalive_timeout_ms),
                (
                    "grpc.keepalive_permit_without_calls",
                    keepalive_options.keepalive_permit_without_calls,
                ),
                (
                    "grpc.http2.max_pings_without_data",
                    keepalive_options.http2_max_pings_without_data,
                ),
            ]

        if creds:
            self._channel = grpc.secure_channel(url, creds, options=channel_opt)
        elif ssl:
            rc_bytes = pk_bytes = cc_bytes = None
            if root_certificates is not None:
                with open(root_certificates, "rb") as rc_fs:
                    rc_bytes = rc_fs.read()
            if private_key is not None:
                with open(private_key, "rb") as pk_fs:
                    pk_bytes = pk_fs.read()
            if certificate_chain is not None:
                with open(certificate_chain, "rb") as cc_fs:
                    cc_bytes = cc_fs.read()
            creds = grpc.ssl_channel_credentials(
                root_certificates=rc_bytes,
                private_key=pk_bytes,
                certificate_chain=cc_bytes,
            )
            self._channel = grpc.secure_channel(url, creds, options=channel_opt)
        else:
            self._channel = grpc.insecure_channel(url, options=channel_opt)

        self._client_stub = service_pb2_grpc.GRPCInferenceServiceStub(self._channel)
        self._verbose = verbose
        self._stream = None

    def _get_metadata(self, headers):
        request = Request(headers)
        self._call_plugin(request)

        request_metadata = (
            request.headers.items() if request.headers is not None else ()
        )
        return request_metadata

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Close the client. Any future calls to server
        will result in an Error.

        """
        self.stop_stream()
        self._channel.close()

    def is_server_live(self, headers=None, client_timeout=None):
        """Contact the inference server and get liveness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        InferenceServerException
            If unable to get liveness or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.ServerLiveRequest()
            if self._verbose:
                print("is_server_live, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.ServerLive(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            return response.live
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def is_server_ready(self, headers=None, client_timeout=None):
        """Contact the inference server and get readiness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        InferenceServerException
            If unable to get readiness or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.ServerReadyRequest()
            if self._verbose:
                print("is_server_ready, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.ServerReady(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            return response.ready
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def is_model_ready(
        self, model_name, model_version="", headers=None, client_timeout=None
    ):
        """Contact the inference server and get the readiness of specified model.

        Parameters
        ----------
        model_name: str
            The name of the model to check for readiness.
        model_version: str
            The version of the model to check for readiness. The default value
            is an empty string which means then the server will choose a version
            based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        bool
            True if the model is ready, False if not ready.

        Raises
        ------
        InferenceServerException
            If unable to get model readiness or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            if type(model_version) != str:
                raise_error("model version must be a string")
            request = service_pb2.ModelReadyRequest(
                name=model_name, version=model_version
            )
            if self._verbose:
                print("is_model_ready, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.ModelReady(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            return response.ready
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_server_metadata(self, headers=None, as_json=False, client_timeout=None):
        """Contact the inference server and get its metadata.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns server metadata as a json dict,
            otherwise as a protobuf message. Default value is
            False. The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.


        Returns
        -------
        dict or protobuf message
            The JSON dict or ServerMetadataResponse message
            holding the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get server metadata or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.ServerMetadataRequest()
            if self._verbose:
                print("get_server_metadata, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.ServerMetadata(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_metadata(
        self,
        model_name,
        model_version="",
        headers=None,
        as_json=False,
        client_timeout=None,
    ):
        """Contact the inference server and get the metadata for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get metadata. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns model metadata as a json dict,
            otherwise as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        dict or protobuf message
            The JSON dict or ModelMetadataResponse message holding
            the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get model metadata or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            if type(model_version) != str:
                raise_error("model version must be a string")
            request = service_pb2.ModelMetadataRequest(
                name=model_name, version=model_version
            )
            if self._verbose:
                print("get_model_metadata, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.ModelMetadata(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_config(
        self,
        model_name,
        model_version="",
        headers=None,
        as_json=False,
        client_timeout=None,
    ):
        """Contact the inference server and get the configuration for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get configuration. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns configuration as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        dict or protobuf message
            The JSON dict or ModelConfigResponse message holding
            the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get model configuration or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            if type(model_version) != str:
                raise_error("model version must be a string")
            request = service_pb2.ModelConfigRequest(
                name=model_name, version=model_version
            )
            if self._verbose:
                print("get_model_config, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.ModelConfig(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_repository_index(
        self, headers=None, as_json=False, client_timeout=None
    ):
        """Get the index of model repository contents

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns model repository index
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        dict or protobuf message
            The JSON dict or RepositoryIndexResponse message holding
            the model repository index.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.RepositoryIndexRequest()
            if self._verbose:
                print(
                    "get_model_repository_index, metadata {}\n{}".format(
                        metadata, request
                    )
                )
            response = self._client_stub.RepositoryIndex(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def load_model(
        self,
        model_name,
        headers=None,
        config=None,
        files=None,
        client_timeout=None,
    ):
        """Request the inference server to load or reload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        config: str
            Optional JSON representation of a model config provided for
            the load request, if provided, this config will be used for
            loading the model.
        files: dict
            Optional dictionary specifying file path (with "file:" prefix) in
            the override model directory to the file content as bytes.
            The files will form the model directory that the model will be
            loaded from. If specified, 'config' must be provided to be
            the model configuration of the override model directory.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Raises
        ------
        InferenceServerException
            If unable to load the model or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.RepositoryModelLoadRequest(model_name=model_name)
            if config is not None:
                request.parameters["config"].string_param = config
            if self._verbose:
                # Don't print file content which can be large
                print(
                    "load_model, metadata {}\noverride files omitted:\n{}".format(
                        metadata, request
                    )
                )
            if files is not None:
                for path, content in files.items():
                    request.parameters[path].bytes_param = content
            self._client_stub.RepositoryModelLoad(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print("Loaded model '{}'".format(model_name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def unload_model(
        self,
        model_name,
        headers=None,
        unload_dependents=False,
        client_timeout=None,
    ):
        """Request the inference server to unload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be unloaded.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        unload_dependents : bool
            Whether the dependents of the model should also be unloaded.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Raises
        ------
        InferenceServerException
            If unable to unload the model or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.RepositoryModelUnloadRequest(model_name=model_name)
            request.parameters["unload_dependents"].bool_param = unload_dependents
            if self._verbose:
                print("unload_model, metadata {}\n{}".format(metadata, request))
            self._client_stub.RepositoryModelUnload(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print("Unloaded model '{}'".format(model_name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_inference_statistics(
        self,
        model_name="",
        model_version="",
        headers=None,
        as_json=False,
        client_timeout=None,
    ):
        """Get the inference statistics for the specified model name and
        version.

        Parameters
        ----------
        model_name : str
            The name of the model to get statistics. The default value is
            an empty string, which means statistics of all models will
            be returned.
        model_version: str
            The version of the model to get inference statistics. The
            default value is an empty string which means then the server
            will return the statistics of all available model versions.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns inference statistics
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Raises
        ------
        InferenceServerException
            If unable to get the model inference statistics or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            if type(model_version) != str:
                raise_error("model version must be a string")
            request = service_pb2.ModelStatisticsRequest(
                name=model_name, version=model_version
            )
            if self._verbose:
                print(
                    "get_inference_statistics, metadata {}\n{}".format(
                        metadata, request
                    )
                )
            response = self._client_stub.ModelStatistics(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def update_trace_settings(
        self,
        model_name=None,
        settings={},
        headers=None,
        as_json=False,
        client_timeout=None,
    ):
        """Update the trace settings for the specified model name, or
        global trace settings if model name is not given.
        Returns the trace settings after the update.

        Parameters
        ----------
        model_name : str
            The name of the model to update trace settings. Specifying None or
            empty string will update the global trace settings.
            The default value is None.
        settings: dict
            The new trace setting values. Only the settings listed will be
            updated. If a trace setting is listed in the dictionary with
            a value of 'None', that setting will be cleared.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns trace settings
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        dict or protobuf message
            The JSON dict or TraceSettingResponse message holding
            the updated trace settings.

        Raises
        ------
        InferenceServerException
            If unable to update the trace settings or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.TraceSettingRequest()
            if (model_name is not None) and (model_name != ""):
                request.model_name = model_name
            for key, value in settings.items():
                if value is None:
                    request.settings[key]
                else:
                    request.settings[key].value.extend(
                        value if isinstance(value, list) else [value]
                    )

            if self._verbose:
                print(
                    "update_trace_settings, metadata {}\n{}".format(metadata, request)
                )
            response = self._client_stub.TraceSetting(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_trace_settings(
        self, model_name=None, headers=None, as_json=False, client_timeout=None
    ):
        """Get the trace settings for the specified model name, or global trace
        settings if model name is not given

        Parameters
        ----------
        model_name : str
            The name of the model to get trace settings. Specifying None or
            empty string will return the global trace settings.
            The default value is None.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns trace settings
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        dict or protobuf message
            The JSON dict or TraceSettingResponse message holding
            the trace settings.

        Raises
        ------
        InferenceServerException
            If unable to get the trace settings or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.TraceSettingRequest()
            if (model_name is not None) and (model_name != ""):
                request.model_name = model_name
            if self._verbose:
                print("get_trace_settings, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.TraceSetting(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def update_log_settings(
        self, settings, headers=None, as_json=False, client_timeout=None
    ):
        """Update the global log settings.
        Returns the log settings after the update.
        Parameters
        ----------
        settings: dict
            The new log setting values. Only the settings listed will be
            updated.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns trace settings
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        Returns
        -------
        dict or protobuf message
            The JSON dict or LogSettingsResponse message holding
            the updated log settings.
        Raises
        ------
        InferenceServerException
            If unable to update the log settings or has timed out.
        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.LogSettingsRequest()
            for key, value in settings.items():
                if value is None:
                    request.settings[key]
                else:
                    if key == "log_file" or key == "log_format":
                        request.settings[key].string_param = value
                    elif key == "log_verbose_level":
                        request.settings[key].uint32_param = value
                    else:
                        request.settings[key].bool_param = value

            if self._verbose:
                print("update_log_settings, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.LogSettings(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_log_settings(self, headers=None, as_json=False, client_timeout=None):
        """Get the global log settings.
        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns log settings
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        Returns
        -------
        dict or protobuf message
            The JSON dict or LogSettingsResponse message holding
            the log settings.
        Raises
        ------
        InferenceServerException
            If unable to get the log settings or has timed out.
        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.LogSettingsRequest()
            if self._verbose:
                print("get_log_settings, metadata {}\n{}".format(metadata, request))
            response = self._client_stub.LogSettings(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_system_shared_memory_status(
        self, region_name="", headers=None, as_json=False, client_timeout=None
    ):
        """Request system shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active system shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns system shared memory status as a json
            dict, otherwise as a protobuf message. Default value is
            False.  The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        dict or protobuf message
            The JSON dict or SystemSharedMemoryStatusResponse message holding
            the system shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.SystemSharedMemoryStatusRequest(name=region_name)
            if self._verbose:
                print(
                    "get_system_shared_memory_status, metadata {}\n{}".format(
                        metadata, request
                    )
                )
            response = self._client_stub.SystemSharedMemoryStatus(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def register_system_shared_memory(
        self, name, key, byte_size, offset=0, headers=None, client_timeout=None
    ):
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        key : str
            The key of the underlying memory object that contains the
            system shared memory region.
        byte_size : int
            The size of the system shared memory region, in bytes.
        offset : int
            Offset, in bytes, within the underlying memory object to
            the start of the system shared memory region. The default
            value is zero.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Raises
        ------
        InferenceServerException
            If unable to register the specified system shared memory or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.SystemSharedMemoryRegisterRequest(
                name=name, key=key, offset=offset, byte_size=byte_size
            )
            if self._verbose:
                print(
                    "register_system_shared_memory, metadata {}\n{}".format(
                        metadata, request
                    )
                )
            self._client_stub.SystemSharedMemoryRegister(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print("Registered system shared memory with name '{}'".format(name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def unregister_system_shared_memory(
        self, name="", headers=None, client_timeout=None
    ):
        """Request the server to unregister a system shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the system shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified system shared memory region or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.SystemSharedMemoryUnregisterRequest(name=name)
            if self._verbose:
                print(
                    "unregister_system_shared_memory, metadata {}\n{}".format(
                        metadata, request
                    )
                )
            self._client_stub.SystemSharedMemoryUnregister(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                if name != "":
                    print(
                        "Unregistered system shared memory with name '{}'".format(name)
                    )
                else:
                    print("Unregistered all system shared memory regions")
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_cuda_shared_memory_status(
        self, region_name="", headers=None, as_json=False, client_timeout=None
    ):
        """Request cuda shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active cuda shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns cuda shared memory status as a json
            dict, otherwise as a protobuf message. Default value is
            False.  The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Returns
        -------
        dict or protobuf message
            The JSON dict or CudaSharedMemoryStatusResponse message holding
            the cuda shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory or has timed out.

        """

        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.CudaSharedMemoryStatusRequest(name=region_name)
            if self._verbose:
                print(
                    "get_cuda_shared_memory_status, metadata {}\n{}".format(
                        metadata, request
                    )
                )
            response = self._client_stub.CudaSharedMemoryStatus(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True)
                )
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def register_cuda_shared_memory(
        self,
        name,
        raw_handle,
        device_id,
        byte_size,
        headers=None,
        client_timeout=None,
    ):
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        raw_handle : bytes
            The raw serialized cudaIPC handle in base64 encoding.
        device_id : int
            The GPU device ID on which the cudaIPC handle was created.
        byte_size : int
            The size of the cuda shared memory region, in bytes.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Raises
        ------
        InferenceServerException
            If unable to register the specified cuda shared memory or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.CudaSharedMemoryRegisterRequest(
                name=name,
                raw_handle=base64.b64decode(raw_handle),
                device_id=device_id,
                byte_size=byte_size,
            )
            if self._verbose:
                print(
                    "register_cuda_shared_memory, metadata {}\n{}".format(
                        metadata, request
                    )
                )
            self._client_stub.CudaSharedMemoryRegister(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                print("Registered cuda shared memory with name '{}'".format(name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def unregister_cuda_shared_memory(self, name="", headers=None, client_timeout=None):
        """Request the server to unregister a cuda shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the cuda shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified cuda shared memory region or has timed out.

        """
        metadata = self._get_metadata(headers)
        try:
            request = service_pb2.CudaSharedMemoryUnregisterRequest(name=name)
            if self._verbose:
                print(
                    "unregister_cuda_shared_memory, metadata {}\n{}".format(
                        metadata, request
                    )
                )
            self._client_stub.CudaSharedMemoryUnregister(
                request=request, metadata=metadata, timeout=client_timeout
            )
            if self._verbose:
                if name != "":
                    print("Unregistered cuda shared memory with name '{}'".format(name))
                else:
                    print("Unregistered all cuda shared memory regions")
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def infer(
        self,
        model_name,
        inputs,
        model_version="",
        outputs=None,
        request_id="",
        sequence_id=0,
        sequence_start=False,
        sequence_end=False,
        priority=0,
        timeout=None,
        client_timeout=None,
        headers=None,
        compression_algorithm=None,
        parameters=None,
    ):
        """Run synchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        model_version : str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start : bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end : bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model. This option is only respected by the model that is
            configured with dynamic batching. See here for more details:
            https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher
        client_timeout : float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        headers : dict
            Optional dictionary specifying additional HTTP headers to include
            in the request.
        compression_algorithm : str
            Optional grpc compression algorithm to be used on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.
        parameters : dict
            Optional custom parameters to be included in the inference
            request.

        Returns
        -------
        InferResult
            The object holding the result of the inference.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """
        metadata = self._get_metadata(headers)

        if type(model_version) != str:
            raise_error("model version must be a string")

        request = _get_inference_request(
            model_name=model_name,
            inputs=inputs,
            model_version=model_version,
            request_id=request_id,
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout,
            parameters=parameters,
        )
        if self._verbose:
            print("infer, metadata {}\n{}".format(metadata, request))

        try:
            response = self._client_stub.ModelInfer(
                request=request,
                metadata=metadata,
                timeout=client_timeout,
                compression=_grpc_compression_type(compression_algorithm),
            )
            if self._verbose:
                print(response)
            result = InferResult(response)
            return result
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def async_infer(
        self,
        model_name,
        inputs,
        callback,
        model_version="",
        outputs=None,
        request_id="",
        sequence_id=0,
        sequence_start=False,
        sequence_end=False,
        priority=0,
        timeout=None,
        client_timeout=None,
        headers=None,
        compression_algorithm=None,
        parameters=None,
    ):
        """Run asynchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        callback : function
            Python function that is invoked once the request is completed.
            The function must reserve the last two arguments (result, error)
            to hold InferResult and InferenceServerException
            objects respectively which will be provided to the function when
            executing the callback. The ownership of these objects will be given
            to the user. The 'error' would be None for a successful inference.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end: bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model. This option is only respected by the model that is
            configured with dynamic batching. See here for more details:
            https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and provide
            error with message "Deadline Exceeded" in the callback when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        client_timeout: float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        compression_algorithm : str
            Optional grpc compression algorithm to be used on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.
        parameters : dict
            Optional custom parameters to be included in the inference
            request.

        Returns
        -------
        CallContext
            A representation of a computation in another control flow.
            Computations represented by a Future may be yet to be begun,
            ongoing, or have already completed.

            This object can be used to cancel the inference request like
            below:
            ----------
            future = async_infer(...)
            ret = future.cancel()
            ----------


        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """

        def wrapped_callback(call_future):
            error = result = None
            try:
                response = call_future.result()
                if self._verbose:
                    print(response)
                result = InferResult(response)
            except grpc.RpcError as rpc_error:
                error = get_error_grpc(rpc_error)
            except grpc.FutureCancelledError as err:
                error = get_cancelled_error()
            callback(result=result, error=error)

        metadata = self._get_metadata(headers)

        if type(model_version) != str:
            raise_error("model version must be a string")

        request = _get_inference_request(
            model_name=model_name,
            inputs=inputs,
            model_version=model_version,
            request_id=request_id,
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout,
            parameters=parameters,
        )
        if self._verbose:
            print("async_infer, metadata {}\n{}".format(metadata, request))

        try:
            self._call_future = self._client_stub.ModelInfer.future(
                request=request,
                metadata=metadata,
                timeout=client_timeout,
                compression=_grpc_compression_type(compression_algorithm),
            )
            if self._verbose:
                verbose_message = "Sent request"
                if request_id != "":
                    verbose_message = verbose_message + " '{}'".format(request_id)
                print(verbose_message)
            self._call_future.add_done_callback(wrapped_callback)
            return CallContext(self._call_future)
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def start_stream(
        self, callback, stream_timeout=None, headers=None, compression_algorithm=None
    ):
        """Starts a grpc bi-directional stream to send streaming inferences.
        Note: When using stream, user must ensure the InferenceServerClient.close()
        gets called at exit.

        Parameters
        ----------
        callback : function
            Python function that is invoked upon receiving response from
            the underlying stream. The function must reserve the last two
            arguments (result, error) to hold InferResult and
            InferenceServerException objects respectively
            which will be provided to the function when executing the callback.
            The ownership of these objects will be given to the user. The 'error'
            would be None for a successful inference.

        stream_timeout : float
            Optional stream timeout (in seconds). The stream will be closed
            once the specified timeout expires.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        compression_algorithm : str
            Optional grpc compression algorithm to be used on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.

        Raises
        ------
        InferenceServerException
            If unable to start a stream or a stream was already running
            for this client or has timed out.

        """
        if self._stream is not None:
            raise_error(
                "cannot start another stream with one already running. "
                "'InferenceServerClient' supports only a single active "
                "stream at a given time."
            )
        metadata = self._get_metadata(headers)

        self._stream = _InferStream(callback, self._verbose)

        try:
            response_iterator = self._client_stub.ModelStreamInfer(
                _RequestIterator(self._stream),
                metadata=metadata,
                timeout=stream_timeout,
                compression=_grpc_compression_type(compression_algorithm),
            )
            self._stream._init_handler(response_iterator)
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def stop_stream(self, cancel_requests=False):
        """Stops a stream if one available.

        Parameters
        ----------
        cancel_requests : bool
            If set True, then client cancels all the pending requests
            and closes the stream. If set False, the call blocks till
            all the pending requests on the stream are processed.

        """
        if self._stream is not None:
            self._stream.close(cancel_requests)
        self._stream = None

    def async_stream_infer(
        self,
        model_name,
        inputs,
        model_version="",
        outputs=None,
        request_id="",
        sequence_id=0,
        sequence_start=False,
        sequence_end=False,
        enable_empty_final_response=False,
        priority=0,
        timeout=None,
        parameters=None,
    ):
        """Runs an asynchronous inference over gRPC bi-directional streaming
        API. A stream must be established with a call to start_stream()
        before calling this function. All the results will be provided to the
        callback function associated with the stream.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int or str
            The unique identifier for the sequence being represented by the
            object.  A value of 0 or "" means that the request does not
            belong to a sequence. Default is 0.
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0 or "".
        sequence_end: bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0 or "".
        enable_empty_final_response: bool
            Indicates whether "empty" responses should be generated and sent
            back to the client from the server during streaming inference when
            they contain the TRITONSERVER_RESPONSE_COMPLETE_FINAL flag.
            This strictly relates to the case of models/backends that send
            flags-only responses (use TRITONBACKEND_ResponseFactorySendFlags(TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            or InferenceResponseSender.send(flags=TRITONSERVER_RESPONSE_COMPLETE_FINAL))
            Currently, this only occurs for decoupled models, and can be
            used to communicate to the client when a request has received
            its final response from the model. If the backend sends the final
            flag along with a non-empty response, this arg is not needed.
            Default value is False.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model. This does not stop the grpc stream itself and is only
            respected by the model that is configured with dynamic batching.
            See here for more details:
            https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher
        parameters : dict
            Optional custom parameters to be included in the inference
            request.
        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """

        if self._stream is None:
            raise_error(
                "stream not available, use start_stream() to make one available."
            )

        if type(model_version) != str:
            raise_error("model version must be a string")

        request = _get_inference_request(
            model_name=model_name,
            inputs=inputs,
            model_version=model_version,
            request_id=request_id,
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout,
            parameters=parameters,
        )

        # Unique to streaming inference as it only pertains to decoupled models
        # Only attach the parameter if True, no need to send/parse when False.
        if enable_empty_final_response:
            request.parameters["triton_enable_empty_final_response"].bool_param = True

        if self._verbose:
            print("async_stream_infer\n{}".format(request))
        # Enqueues the request to the stream
        self._stream._enqueue_request(request)
        if self._verbose:
            print("enqueued request {} to stream...".format(request_id))
