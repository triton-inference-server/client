# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Clients for easy interaction with models deployed on the Triton Inference Server.

Typical usage example:

```python
client = ModelClient("localhost", "MyModel")
result_dict = client.infer_sample(input_a=a, input_b=b)
client.close()
```

Inference inputs can be provided either as positional or keyword arguments:

```python
result_dict = client.infer_sample(input1, input2)
result_dict = client.infer_sample(a=input1, b=input2)
```

Mixing of argument passing conventions is not supported and will raise PyTritonClientValueError.
"""

import asyncio
import contextlib
import itertools
import logging
import socket
import time
import warnings
from concurrent.futures import Future
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional, Tuple, Union

# import gevent
import numpy as np
import tritonclient.hl.lw.grpc
import tritonclient.hl.lw.http
import tritonclient.hl.lw.utils
from tritonclient.hl.exceptions import (
    PyTritonClientClosedError,
    PyTritonClientInferenceServerError,
    PyTritonClientModelDoesntSupportBatchingError,
    PyTritonClientQueueFullError,
    PyTritonClientTimeoutError,
    PyTritonClientValueError,
)
from tritonclient.hl.triton_model_config import TritonModelConfig
from tritonclient.hl.utils import (
    _DEFAULT_NETWORK_TIMEOUT_S,
    _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
    TritonUrl,
    get_model_config,
    wait_for_model_ready,
    wait_for_server_ready,
)
from tritonclient.hl.warnings import NotSupportedTimeoutWarning

# Old client tritonclient imports
# import tritonclient.grpc
# import tritonclient.grpc.aio
# import tritonclient.http
# import tritonclient.http.aio
# import tritonclient.utils


_LOGGER = logging.getLogger(__name__)

_DEFAULT_SYNC_INIT_TIMEOUT_S = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
_DEFAULT_FUTURES_INIT_TIMEOUT_S = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
DEFAULT_INFERENCE_TIMEOUT_S = 60.0


_IOType = Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]


def _verify_inputs_args(inputs, named_inputs):
    if not inputs and not named_inputs:
        raise PyTritonClientValueError("Provide input data")
    if not bool(inputs) ^ bool(named_inputs):
        raise PyTritonClientValueError(
            "Use either positional either keyword method arguments convention"
        )


def _verify_parameters(
    parameters_or_headers: Optional[Dict[str, Union[str, int, bool]]] = None
):
    if parameters_or_headers is None:
        return
    if not isinstance(parameters_or_headers, dict):
        raise PyTritonClientValueError("Parameters and headers must be a dictionary")
    for key, value in parameters_or_headers.items():
        if not isinstance(key, str):
            raise PyTritonClientValueError("Parameter/header key must be a string")
        if not isinstance(value, (str, int, bool)):
            raise PyTritonClientValueError(
                "Parameter/header value must be a string, integer or boolean"
            )


class BaseModelClient:
    """Base client for model deployed on the Triton Inference Server."""

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
        model_config: Optional[TritonModelConfig] = None,
        ensure_model_is_ready: bool = True,
    ):
        """Inits BaseModelClient for given model deployed on the Triton Inference Server.

        Common usage:

        ```python
        client = ModelClient("localhost", "BERT")
        result_dict = client.infer_sample(input1_sample, input2_sample)
        client.close()
        ```

        Args:
            url: The Triton Inference Server url, e.g. `grpc://localhost:8001`.
                In case no scheme is provided http scheme will be used as default.
                In case no port is provided default port for given scheme will be used -
                8001 for grpc scheme, 8000 for http scheme.
            model_name: name of the model to interact with.
            model_version: version of the model to interact with.
                If model_version is None inference on latest model will be performed.
                The latest versions of the model are numerically the greatest version numbers.
            lazy_init: if initialization should be performed just before sending first request to inference server.
            init_timeout_s: timeout in seconds for the server and model to be ready. If not passed, the default timeout of 300 seconds will be used.
            inference_timeout_s: timeout in seconds for a single model inference request. If not passed, the default timeout of 60 seconds will be used.
            model_config: model configuration. If not passed, it will be read from inference server during initialization.
            ensure_model_is_ready: if model should be checked if it is ready before first inference request.

        Raises:
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientTimeoutError:
                if `lazy_init` argument is False and wait time for server and model being ready exceeds `init_timeout_s`.
            PyTritonClientInvalidUrlError: If provided Triton Inference Server url is invalid.
        """
        self._init_timeout_s = (
            _DEFAULT_SYNC_INIT_TIMEOUT_S if init_timeout_s is None else init_timeout_s
        )
        self._inference_timeout_s = (
            DEFAULT_INFERENCE_TIMEOUT_S
            if inference_timeout_s is None
            else inference_timeout_s
        )
        self._network_timeout_s = min(_DEFAULT_NETWORK_TIMEOUT_S, self._init_timeout_s)

        self._general_client = self.create_client_from_url(
            url, network_timeout_s=self._network_timeout_s
        )
        self._infer_client = self.create_client_from_url(
            url, network_timeout_s=self._inference_timeout_s
        )

        self._model_name = model_name
        self._model_version = model_version

        self._request_id_generator = itertools.count(0)

        if model_config is not None:
            self._model_config = model_config
            self._model_ready = None if ensure_model_is_ready else True

        else:
            self._model_config = None
            self._model_ready = None
        self._lazy_init: bool = lazy_init

        self._handle_lazy_init()

    @classmethod
    def from_existing_client(cls, existing_client: "BaseModelClient"):
        """Create a new instance from an existing client using the same class.

        Common usage:
        ```python
        client = BaseModelClient.from_existing_client(existing_client)
        ```

        Args:
            existing_client: An instance of an already initialized subclass.

        Returns:
            A new instance of the same subclass with shared configuration and readiness state.
        """
        kwargs = {}
        # Copy model configuration and readiness state if present
        if hasattr(existing_client, "_model_config"):
            kwargs["model_config"] = existing_client._model_config
            kwargs["ensure_model_is_ready"] = False

        new_client = cls(
            url=existing_client._url,
            model_name=existing_client._model_name,
            model_version=existing_client._model_version,
            init_timeout_s=existing_client._init_timeout_s,
            inference_timeout_s=existing_client._inference_timeout_s,
            **kwargs,
        )

        return new_client

    def create_client_from_url(
        self, url: str, network_timeout_s: Optional[float] = None
    ):
        """Create Triton Inference Server client.

        Args:
            url: url of the server to connect to.
                If url doesn't contain scheme (e.g. "localhost:8001") http scheme is added.
                If url doesn't contain port (e.g. "localhost") default port for given scheme is added.
            network_timeout_s: timeout for client commands. Default value is 60.0 s.

        Returns:
            Triton Inference Server client.

        Raises:
            PyTritonClientInvalidUrlError: If provided Triton Inference Server url is invalid.
        """
        self._triton_url = TritonUrl.from_url(url)
        self._url = self._triton_url.without_scheme
        self._triton_client_lib = self.get_lib()

        if self._triton_url.scheme == "grpc":
            # by default grpc client has very large number of timeout, thus we want to make it equal to http client timeout
            network_timeout_s = (
                _DEFAULT_NETWORK_TIMEOUT_S
                if network_timeout_s is None
                else network_timeout_s
            )
            warnings.warn(
                f"tritonclient.grpc doesn't support timeout for other commands than infer. Ignoring network_timeout: {network_timeout_s}.",
                NotSupportedTimeoutWarning,
                stacklevel=1,
            )

        triton_client_init_kwargs = self._get_init_extra_args()

        _LOGGER.debug(
            f"Creating InferenceServerClient for {self._triton_url.with_scheme} with {triton_client_init_kwargs}"
        )
        return self._triton_client_lib.InferenceServerClient(
            self._url, **triton_client_init_kwargs
        )

    def get_lib(self):
        """Returns tritonclient library for given scheme."""
        raise NotImplementedError

    @property
    def _next_request_id(self) -> str:
        # pytype complained about creating generator in __init__ method
        # so we create it lazily
        if getattr(self, "_request_id_generator", None) is None:
            self._request_id_generator = itertools.count(0)
        return str(next(self._request_id_generator))

    def _get_init_extra_args(self):
        timeout = self._inference_timeout_s  # pytype: disable=attribute-error
        #  The inference timeout is used for both the HTTP and the GRPC protocols. However,
        #  the way the timeout is passed to the client differs depending on the protocol.
        #  For the HTTP protocol, the timeout is set in the ``__init__`` method as ``network_timeout``
        #  and ``connection_timeout``. For the GRPC protocol, the timeout
        #  is passed to the infer method as ``client_timeout``.
        #  Both protocols support timeouts correctly and will raise an exception
        #  if the network request or the inference process takes longer than the timeout.
        #  This is a design choice of the underlying tritonclient library.

        if self._triton_url.scheme != "http":
            return {}

        kwargs = {
            # This value sets the maximum time allowed for each network request in both model loading and inference process
            "network_timeout": timeout,
            # This value sets the maximum time allowed for establishing a connection to the server.
            # We use the inference timeout here instead of the init timeout because the init timeout
            # is meant for waiting for the model to be ready. The connection timeout should be shorter
            # than the init timeout because it only checks if connection is established (e.g. correct port)
            "connection_timeout": timeout,
        }
        return kwargs

    def _monkey_patch_client(self):
        pass

    def _get_model_config_extra_args(self):
        # For the GRPC protocol, the timeout must be passed to the each request as client_timeout
        # model_config doesn't yet support timeout but it is planned for the future
        # grpc_network_timeout_s will be used for model_config
        return {}

    def _handle_lazy_init(self):
        raise NotImplementedError


def _run_once_per_lib(f):
    def wrapper(_self):
        if _self._triton_client_lib not in wrapper.patched:
            wrapper.patched.add(_self._triton_client_lib)
            return f(_self)

    wrapper.patched = set()
    return wrapper


class ModelClient(BaseModelClient):
    """Synchronous client for model deployed on the Triton Inference Server."""

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
        model_config: Optional[TritonModelConfig] = None,
        ensure_model_is_ready: bool = True,
    ):
        """Inits ModelClient for given model deployed on the Triton Inference Server.

        If `lazy_init` argument is False, model configuration will be read
        from inference server during initialization.

        Common usage:

        ```python
        client = ModelClient("localhost", "BERT")
        result_dict = client.infer_sample(input1_sample, input2_sample)
        client.close()
        ```

        Client supports also context manager protocol:

        ```python
        with ModelClient("localhost", "BERT") as client:
            result_dict = client.infer_sample(input1_sample, input2_sample)
        ```

        The creation of client requires connection to the server and downloading model configuration. You can create client from existing client using the same class:

        ```python
        client = ModelClient.from_existing_client(existing_client)
        ```

        Args:
            url: The Triton Inference Server url, e.g. 'grpc://localhost:8001'.
                In case no scheme is provided http scheme will be used as default.
                In case no port is provided default port for given scheme will be used -
                8001 for grpc scheme, 8000 for http scheme.
            model_name: name of the model to interact with.
            model_version: version of the model to interact with.
                If model_version is None inference on latest model will be performed.
                The latest versions of the model are numerically the greatest version numbers.
            lazy_init: if initialization should be performed just before sending first request to inference server.
            init_timeout_s: timeout for maximum waiting time in loop, which sends retry requests ask if model is ready. It is applied at initialization time only when `lazy_init` argument is False. Default is to do retry loop at first inference.
            inference_timeout_s: timeout in seconds for the model inference process.
                If non passed default 60 seconds timeout will be used.
                For HTTP client it is not only inference timeout but any client request timeout
                - get model config, is model loaded. For GRPC client it is only inference timeout.
            model_config: model configuration. If not passed, it will be read from inference server during initialization.
            ensure_model_is_ready: if model should be checked if it is ready before first inference request.

        Raises:
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientTimeoutError:
                if `lazy_init` argument is False and wait time for server and model being ready exceeds `init_timeout_s`.
            PyTritonClientUrlParseError: In case of problems with parsing url.
        """
        super().__init__(
            url=url,
            model_name=model_name,
            model_version=model_version,
            lazy_init=lazy_init,
            init_timeout_s=init_timeout_s,
            inference_timeout_s=inference_timeout_s,
            model_config=model_config,
            ensure_model_is_ready=ensure_model_is_ready,
        )

    def get_lib(self):
        """Returns tritonclient library for given scheme."""
        return {"grpc": tritonclient.hl.lw.grpc, "http": tritonclient.hl.lw.http}[
            self._triton_url.scheme.lower()
        ]

    def __enter__(self):
        """Create context for using ModelClient as a context manager."""
        return self

    def __exit__(self, *_):
        """Close resources used by ModelClient instance when exiting from the context."""
        self.close()

    def load_model(self, config: Optional[str] = None, files: Optional[dict] = None):
        """Load model on the Triton Inference Server.

        Args:
            config: str - Optional JSON representation of a model config provided for
                the load request, if provided, this config will be used for
                loading the model.
            files: dict - Optional dictionary specifying file path (with "file:" prefix) in
                the override model directory to the file content as bytes.
                The files will form the model directory that the model will be
                loaded from. If specified, 'config' must be provided to be
                the model configuration of the override model directory.
        """
        self._general_client.load_model(self._model_name, config=config, files=files)

    def unload_model(self):
        """Unload model from the Triton Inference Server."""
        self._general_client.unload_model(self._model_name)

    def close(self):
        """Close resources used by ModelClient.

        This method closes the resources used by the ModelClient instance,
        including the Triton Inference Server connections.
        Once this method is called, the ModelClient instance should not be used again.
        """
        _LOGGER.debug("Closing ModelClient")
        try:
            if self._general_client is not None:
                self._general_client.close()
            if self._infer_client is not None:
                self._infer_client.close()
            self._general_client = None
            self._infer_client = None
        except Exception as e:
            _LOGGER.error(f"Error while closing ModelClient resources: {e}")
            raise e

    def wait_for_model(self, timeout_s: float):
        """Wait for the Triton Inference Server and the deployed model to be ready.

        Args:
            timeout_s: timeout in seconds to wait for the server and model to be ready.

        Raises:
            PyTritonClientTimeoutError: If the server and model are not ready before the given timeout.
            PyTritonClientModelUnavailableError: If the model with the given name (and version) is unavailable.
            KeyboardInterrupt: If the hosting process receives SIGINT.
            PyTritonClientClosedError: If the ModelClient is closed.
        """
        if self._general_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")
        wait_for_model_ready(
            self._general_client,
            self._model_name,
            self._model_version,
            timeout_s=timeout_s,
        )

    @property
    def is_batching_supported(self):
        """Checks if model supports batching.

        Also waits for server to get into readiness state.
        """
        return self.model_config.max_batch_size > 0

    def wait_for_server(self, timeout_s: float):
        """Wait for Triton Inference Server readiness.

        Args:
            timeout_s: timeout to server get into readiness state.

        Raises:
            PyTritonClientTimeoutError: If server is not in readiness state before given timeout.
            KeyboardInterrupt: If hosting process receives SIGINT
        """
        wait_for_server_ready(self._general_client, timeout_s=timeout_s)

    @property
    def model_config(self) -> TritonModelConfig:
        """Obtain the configuration of the model deployed on the Triton Inference Server.

        This method waits for the server to get into readiness state before obtaining the model configuration.

        Returns:
            TritonModelConfig: configuration of the model deployed on the Triton Inference Server.

        Raises:
            PyTritonClientTimeoutError: If the server and model are not in readiness state before the given timeout.
            PyTritonClientModelUnavailableError: If the model with the given name (and version) is unavailable.
            KeyboardInterrupt: If the hosting process receives SIGINT.
            PyTritonClientClosedError: If the ModelClient is closed.
        """
        if not self._model_config:
            if self._general_client is None:
                raise PyTritonClientClosedError("ModelClient is closed")

            self._model_config = get_model_config(
                self._general_client,
                self._model_name,
                self._model_version,
                timeout_s=self._init_timeout_s,
            )
        return self._model_config

    def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:
        """Run synchronous inference on a single data sample.

        Typical usage:

        ```python
        client = ModelClient("localhost", "MyModel")
        result_dict = client.infer_sample(input1, input2)
        client.close()
        ```

        Inference inputs can be provided either as positional or keyword arguments:

        ```python
        result_dict = client.infer_sample(input1, input2)
        result_dict = client.infer_sample(a=input1, b=input2)
        ```

        Args:
            *inputs: Inference inputs provided as positional arguments.
            parameters: Custom inference parameters.
            headers: Custom inference headers.
            **named_inputs: Inference inputs provided as named arguments.

        Returns:
            Dictionary with inference results, where dictionary keys are output names.

        Raises:
            PyTritonClientValueError: If mixing of positional and named arguments passing detected.
            PyTritonClientTimeoutError: If the wait time for the server and model being ready exceeds `init_timeout_s` or
                inference request time exceeds `inference_timeout_s`.
            PyTritonClientModelUnavailableError: If the model with the given name (and version) is unavailable.
            PyTritonClientInferenceServerError: If an error occurred on the inference callable or Triton Inference Server side.
        """
        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        if self.is_batching_supported:
            if inputs:
                inputs = tuple(data[np.newaxis, ...] for data in inputs)
            elif named_inputs:
                named_inputs = {
                    name: data[np.newaxis, ...] for name, data in named_inputs.items()
                }

        result = self._infer(inputs or named_inputs, parameters, headers)

        return self._debatch_result(result)

    def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:
        """Run synchronous inference on batched data.

        Typical usage:

        ```python
        client = ModelClient("localhost", "MyModel")
        result_dict = client.infer_batch(input1, input2)
        client.close()
        ```

        Inference inputs can be provided either as positional or keyword arguments:

        ```python
        result_dict = client.infer_batch(input1, input2)
        result_dict = client.infer_batch(a=input1, b=input2)
        ```

        Args:
            *inputs: Inference inputs provided as positional arguments.
            parameters: Custom inference parameters.
            headers: Custom inference headers.
            **named_inputs: Inference inputs provided as named arguments.

        Returns:
            Dictionary with inference results, where dictionary keys are output names.

        Raises:
            PyTritonClientValueError: If mixing of positional and named arguments passing detected.
            PyTritonClientTimeoutError: If the wait time for the server and model being ready exceeds `init_timeout_s` or
                inference request time exceeds `inference_timeout_s`.
            PyTritonClientModelUnavailableError: If the model with the given name (and version) is unavailable.
            PyTritonClientInferenceServerError: If an error occurred on the inference callable or Triton Inference Server side.
            PyTritonClientModelDoesntSupportBatchingError: If the model doesn't support batching.
            PyTritonClientValueError: if mixing of positional and named arguments passing detected.
            PyTritonClientTimeoutError:
                in case of first method call, `lazy_init` argument is False
                and wait time for server and model being ready exceeds `init_timeout_s` or
                inference time exceeds `inference_timeout_s` passed to `__init__`.
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientInferenceServerError: If error occurred on inference callable or Triton Inference Server side,
        """
        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        if not self.is_batching_supported:
            raise PyTritonClientModelDoesntSupportBatchingError(
                f"Model {self.model_config.model_name} doesn't support batching - use infer_sample method instead"
            )

        return self._infer(inputs or named_inputs, parameters, headers)

    def infer(self, inputs):
        """Run synchronous batch inference using a single dictionary of inputs."""
        return self.infer_batch(**inputs)

    def _wait_and_init_model_config(self, init_timeout_s: float):
        if self._general_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")

        should_finish_before_s = time.time() + init_timeout_s
        self.wait_for_model(init_timeout_s)
        self._model_ready = True
        timeout_s = max(0.0, should_finish_before_s - time.time())
        self._model_config = get_model_config(
            self._general_client,
            self._model_name,
            self._model_version,
            timeout_s=timeout_s,
        )

    def _create_request(self, inputs: _IOType):
        if self._infer_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")

        if not self._model_ready:
            self._wait_and_init_model_config(self._init_timeout_s)

        if isinstance(inputs, Tuple):
            inputs = {
                input_spec.name: input_data
                for input_spec, input_data in zip(self.model_config.inputs, inputs)
            }

        inputs_wrapped = []

        # to help pytype to obtain variable type
        inputs: Dict[str, np.ndarray]

        for input_name, input_data in inputs.items():
            if input_data.dtype == object and not isinstance(
                input_data.reshape(-1)[0], bytes
            ):
                raise RuntimeError(
                    f"Numpy array for {input_name!r} input with dtype=object should contain encoded strings \
                    \\(e.g. into utf-8\\). Element type: {type(input_data.reshape(-1)[0])}"
                )
            if input_data.dtype.type == np.str_:
                raise RuntimeError(
                    "Unicode inputs are not supported. "
                    f"Encode numpy array for {input_name!r} input (ex. with np.char.encode(array, 'utf-8'))."
                )
            triton_dtype = tritonclient.utils.np_to_triton_dtype(input_data.dtype)
            infer_input = self._triton_client_lib.InferInput(
                input_name, input_data.shape, triton_dtype
            )
            infer_input.set_data_from_numpy(input_data)
            inputs_wrapped.append(infer_input)

        outputs_wrapped = [
            self._triton_client_lib.InferRequestedOutput(output_spec.name)
            for output_spec in self.model_config.outputs
        ]
        return inputs_wrapped, outputs_wrapped

    def _infer(self, inputs: _IOType, parameters, headers) -> Dict[str, np.ndarray]:
        # import tritonclient.http
        # import tritonclient.utils
        if self.model_config.decoupled:
            raise PyTritonClientInferenceServerError(
                "Model config is decoupled. Use DecoupledModelClient instead."
            )

        inputs_wrapped, outputs_wrapped = self._create_request(inputs)

        try:
            _LOGGER.debug("Sending inference request to Triton Inference Server")
            response = self._infer_client.infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                headers=headers,
                outputs=outputs_wrapped,
                request_id=self._next_request_id,
                parameters=parameters,
                **self._get_infer_extra_args(),
            )
        except tritonclient.utils.InferenceServerException as e:
            # tritonclient.grpc raises exception with message containing "Deadline Exceeded" for timeout
            if "Deadline Exceeded" in e.message():
                raise PyTritonClientTimeoutError(
                    f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s. Message: {e.message()}"
                ) from e

            raise PyTritonClientInferenceServerError(
                f"Error occurred during inference request. Message: {e.message()}"
            ) from e
        except (
            socket.timeout
        ) as e:  # tritonclient.http raises socket.timeout for timeout
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        except (
            OSError
        ) as e:  # tritonclient.http raises socket.error for connection error
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e

        ## FIXME: Why it is necessary
        # if isinstance(response, tritonclient.http.InferResult):
        #    outputs = {
        #        output["name"]: response.as_numpy(output["name"]) for output in response.get_response()["outputs"]
        #    }
        # else:
        outputs = {
            output.name: response.as_numpy(output.name)
            for output in response.get_response().outputs
        }

        return outputs

    def _get_numpy_result(self, result):
        # FIXME: Investigate if still it works for coupled model
        # import tritonclient.hl.lw.grpc
        # if isinstance(result, tritonclient.hl.lw.grpc.InferResult):
        result = {
            output.name: result.as_numpy(output.name)
            for output in result.get_response().outputs
        }
        # else:
        #    result = {output["name"]: result.as_numpy(output["name"]) for output in result.get_response()["outputs"]}
        return result

    def _debatch_result(self, result):
        if self.is_batching_supported:
            result = {name: data[0] for name, data in result.items()}
        return result

    def _handle_lazy_init(self):
        if not self._lazy_init:
            self._wait_and_init_model_config(self._init_timeout_s)

    def _get_infer_extra_args(self):
        if self._triton_url.scheme == "http":
            return {}
        # For the GRPC protocol, the timeout is passed to the infer method as client_timeout
        # This timeout applies to the whole inference process and each network request

        # The ``infer`` supports also timeout argument for both GRPC and HTTP.
        # It is applied at server side and supported only for dynamic batching.
        # However, it is not used here yet and planned for future release
        kwargs = {"client_timeout": self._inference_timeout_s}
        return kwargs

    @_run_once_per_lib
    def _monkey_patch_client(self):
        """Monkey patch InferenceServerClient to catch error in __del__."""
        _LOGGER.info(f"Patch ModelClient {self._triton_url.scheme}")
        if not hasattr(self._triton_client_lib.InferenceServerClient, "__del__"):
            return

        old_del = self._triton_client_lib.InferenceServerClient.__del__

        def _monkey_patched_del(self):
            """Monkey patched del."""
            try:
                old_del(self)
            except gevent.exceptions.InvalidThreadUseError:
                _LOGGER.info(
                    "gevent.exceptions.InvalidThreadUseError in __del__ of InferenceServerClient"
                )
            except Exception as e:
                _LOGGER.error("Exception in __del__ of InferenceServerClient: %s", e)

        self._triton_client_lib.InferenceServerClient.__del__ = _monkey_patched_del


class DecoupledModelClient(ModelClient):
    """Synchronous client for decoupled model deployed on the Triton Inference Server."""

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
        model_config: Optional[TritonModelConfig] = None,
        ensure_model_is_ready: bool = True,
    ):
        """Inits DecoupledModelClient for given decoupled model deployed on the Triton Inference Server.

        Common usage:

        ```python
        client = DecoupledModelClient("localhost", "BERT")
        for response in client.infer_sample(input1_sample, input2_sample):
            print(response)
        client.close()
        ```

        Args:
            url: The Triton Inference Server url, e.g. `grpc://localhost:8001`.
                In case no scheme is provided http scheme will be used as default.
                In case no port is provided default port for given scheme will be used -
                8001 for grpc scheme, 8000 for http scheme.
            model_name: name of the model to interact with.
            model_version: version of the model to interact with.
                If model_version is None inference on latest model will be performed.
                The latest versions of the model are numerically the greatest version numbers.
            lazy_init: if initialization should be performed just before sending first request to inference server.
            init_timeout_s: timeout in seconds for the server and model to be ready. If not passed, the default timeout of 300 seconds will be used.
            inference_timeout_s: timeout in seconds for a single model inference request. If not passed, the default timeout of 60 seconds will be used.
            model_config: model configuration. If not passed, it will be read from inference server during initialization.
            ensure_model_is_ready: if model should be checked if it is ready before first inference request.

        Raises:
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientTimeoutError:
                if `lazy_init` argument is False and wait time for server and model being ready exceeds `init_timeout_s`.
            PyTritonClientInvalidUrlError: If provided Triton Inference Server url is invalid.
        """
        super().__init__(
            url,
            model_name,
            model_version,
            lazy_init=lazy_init,
            init_timeout_s=init_timeout_s,
            inference_timeout_s=inference_timeout_s,
            model_config=model_config,
            ensure_model_is_ready=ensure_model_is_ready,
        )
        # Let's use generate endpoints for PoC
        # if self._triton_url.scheme == "http":
        #    raise PyTritonClientValueError("DecoupledModelClient is only supported for grpc protocol")
        self._queue = Queue()
        self._lock = Lock()

    def close(self):
        """Close resources used by DecoupledModelClient."""
        _LOGGER.debug("Closing DecoupledModelClient")
        if self._lock.acquire(blocking=False):
            try:
                super().close()
            finally:
                self._lock.release()
        else:
            _LOGGER.warning("DecoupledModelClient is stil streaming answers")
            self._infer_client.stop_stream(False)
            super().close()

    def _infer(self, inputs: _IOType, parameters, headers):
        if not self._lock.acquire(blocking=False):
            raise PyTritonClientInferenceServerError("Inference is already in progress")
        if not self.model_config.decoupled:
            raise PyTritonClientInferenceServerError(
                "Model config is coupled. Use ModelClient instead."
            )

        inputs_wrapped, outputs_wrapped = self._create_request(inputs)
        if parameters is not None:
            raise PyTritonClientValueError(
                "DecoupledModelClient does not support parameters"
            )
        if headers is not None:
            raise PyTritonClientValueError(
                "DecoupledModelClient does not support headers"
            )
        try:
            _LOGGER.debug("Sending inference request to Triton Inference Server")
            if self._infer_client._stream is None:
                self._infer_client.start_stream(
                    callback=lambda result, error: self._response_callback(
                        result, error
                    )
                )

            self._infer_client.async_stream_infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                outputs=outputs_wrapped,
                request_id=self._next_request_id,
                enable_empty_final_response=True,
                **self._get_infer_extra_args(),
            )
        except tritonclient.utils.InferenceServerException as e:
            # tritonclient.grpc raises exception with message containing "Deadline Exceeded" for timeout
            if "Deadline Exceeded" in e.message():
                raise PyTritonClientTimeoutError(
                    f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s. Message: {e.message()}"
                ) from e

            raise PyTritonClientInferenceServerError(
                f"Error occurred during inference request. Message: {e.message()}"
            ) from e
        except (
            socket.timeout
        ) as e:  # tritonclient.http raises socket.timeout for timeout
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        except (
            OSError
        ) as e:  # tritonclient.http raises socket.error for connection error
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        _LOGGER.debug("Returning response iterator")
        return self._create_response_iterator()

    def _response_callback(self, response, error):
        _LOGGER.debug(f"Received response from Triton Inference Server: {response}")
        if error:
            _LOGGER.error(f"Error occurred during inference request. Message: {error}")
            self._queue.put(error)
        else:
            actual_response = response.get_response()
            # Check if the object is not None
            triton_final_response = actual_response.parameters.get(
                "triton_final_response"
            )
            if triton_final_response and triton_final_response.bool_param:
                self._queue.put(None)
            else:
                result = self._get_numpy_result(response)
                self._queue.put(result)

    def _create_response_iterator(self):
        try:
            while True:
                try:
                    item = self._queue.get(self._inference_timeout_s)
                except Empty as e:
                    message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s"
                    _LOGGER.error(message)
                    raise PyTritonClientTimeoutError(message) from e
                if isinstance(item, Exception):
                    if hasattr(item, "message"):
                        message = f"Error occurred during inference request. Message: {item.message()}"
                    else:
                        message = (
                            f"Error occurred during inference request. Message: {item}"
                        )
                    _LOGGER.error(message)
                    raise PyTritonClientInferenceServerError(message) from item

                if item is None:
                    break
                yield item
        finally:
            self._lock.release()

    def _debatch_result(self, result):
        if self.is_batching_supported:
            result = (
                {name: data[0] for name, data in result_.items()} for result_ in result
            )
        return result

    def _get_infer_extra_args(self):
        # kwargs = super()._get_infer_extra_args()
        kwargs = {}
        # kwargs["enable_empty_final_response"] = True
        return kwargs


class InferenceServerClientBase:
    def __init__(self):
        self._plugin = None

    def _call_plugin(self, request):
        """Called by the subclasses before sending a request to the
        network.
        """
        if self._plugin != None:
            self._plugin(request)

    def register_plugin(self, plugin):
        """Register a Client Plugin.

        Parameters
        ----------
        plugin : InferenceServerClientPlugin
            A client plugin

        Raises
        ------
        InferenceServerException
            If a plugin is already registered.
        """

        if self._plugin is None:
            self._plugin = plugin
        else:
            raise_error(
                "A plugin is already registered. Please "
                "unregister the previous plugin first before"
                " registering a new plugin."
            )

    def plugin(self):
        """Retrieve the registered plugin if any.

        Returns
        ------
        InferenceServerClientPlugin or None
        """
        return self._plugin

    def unregister_plugin(self):
        """Unregister a plugin.

        Raises
        ------
        InferenceServerException
            If no plugin has been registered.
        """
        if self._plugin is None:
            raise_error("No plugin has been registered.")

        self._plugin = None


# # Change url to 'http://localhost:8000' for utilizing HTTP client
# client = Client(url='grpc://loacalhost:8001')
#
# input_tensor_as_numpy = np.array(...)
#
# # Infer should be async similar to the existing Python APIs
# responses = client.model('simple').infer(inputs={'input': input_tensor_as_numpy})
#
# for response in responses:
# 	numpy_array = np.asarray(response.outputs['output'])
#
# client.close()


class Client(InferenceServerClientBase):
    def __init__(self, url: str) -> None:
        self._client_url = url
        super().__init__()

    def model(self, name: str) -> ModelClient:
        return ModelClient(url=self._client_url, model_name=name)
