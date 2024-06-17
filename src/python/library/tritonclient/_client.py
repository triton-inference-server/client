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
from typing import Union

from tritonclient.hl import DecoupledModelClient, ModelClient
from tritonclient.utils import raise_error


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


class Client:
    def __init__(self, url: str) -> None:
        self._client_url = url
        super().__init__()

    def model(self, name: str) -> Union[ModelClient, DecoupledModelClient]:
        client = ModelClient(url=self._client_url, model_name=name)
        if client.model_config.decoupled:
            try:
                decoupled_client = DecoupledModelClient.from_existing_client(client)
            finally:
                client.close()
            return decoupled_client
        else:
            return client
