# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    import aiohttp
    from urllib.parse import quote, quote_plus
    import rapidjson as json
    import gzip, zlib
    import base64
except ModuleNotFoundError as error:
    raise RuntimeError(
        'The installation does not include http support. Specify \'http\' or \'all\' while installing the tritonclient package to include the support'
    ) from error

from tritonclient.utils import *

from tritonclient.http import _get_query_string, _get_inference_request
from tritonclient.http import InferInput, InferRequestedOutput, InferResult


async def _get_error(response):
    """
    Returns the InferenceServerException object if response
    indicates the error. If no error then return None
    """
    if response.status != 200:
        error_response = json.loads(await response.read())
        return InferenceServerException(msg=error_response["error"])
    else:
        return None


async def _raise_if_error(response):
    """
    Raise InferenceServerException if received non-Success
    response from the server
    """
    error = await _get_error(response)
    if error is not None:
        raise error


class InferenceServerClient:
    """An InferenceServerClient object is used to perform any kind of
    communication with the InferenceServer using http protocol. None
    of the methods are thread safe. The object is intended to be used
    by a single thread and simultaneously calling different methods
    with different threads is not supported and will cause undefined
    behavior.

    Parameters
    ----------
    url : str
        The inference server name, port and optional base path 
        in the following format: host:port/<base-path>, e.g.
        'localhost:8000'.

    verbose : bool
        If True generate verbose output. Default value is False.
    conn_limit : int
        Total number of simultaneous connections. Default value is 100.
    conn_timeout : float
        Total number of seconds for an entire request.
        Default value is 60.0 sec.
    ssl : bool
        If True, channels the requests to encrypted https scheme.
        Some improper settings may cause connection to prematurely
        terminate with an unsuccessful handshake. See
        `ssl_context` option for using secure default
        settings. Default value for this option is False.
    ssl_context : None / False / ssl.SSLContext object
        None: Use default SSL check, ssl.create_default_context()
        False: Skip SSL certification validation
        ssl.SSLContext: For custom SSL certification validation
        The argument is ignored if 'ssl' is specified False.

    Raises
        ------
        Exception
            If unable to create a client.

    """

    def __init__(self,
                 url,
                 verbose=False,
                 conn_limit=100,
                 conn_timeout=60.0,
                 ssl=False,
                 ssl_context=None):
        if url.startswith("http://") or url.startswith("https://"):
            raise_error("url should not include the scheme")
        scheme = "https://" if ssl else "http://"
        self._url = scheme + (url if url[-1] != "/" else url[:-1])
        self._conn = aiohttp.TCPConnector(ssl=ssl_context, limit=conn_limit)
        self._stub = aiohttp.ClientSession(
            connector=self._conn, 
            timeout=aiohttp.ClientTimeout(total=conn_timeout), 
            auto_decompress=False)
        self._verbose = verbose

    async def __aenter__(self):
        return self

    async def __aexit__(self, type, value, traceback):
        await self.close()

    async def close(self):
        """Close the client. Any future calls to server
        will result in an Error.

        """
        await self._stub.close()
        await self._conn.close()

    async def _get(self, request_uri, headers, query_params):
        """Issues the GET request to the server

         Parameters
        ----------
        request_uri: str
            The request URI to be used in GET request.
        headers: dict
            Additional HTTP headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        aiohttp.ClientResponse
            The response from server.
        """
        self._validate_headers(headers)
        req_url = self._url + "/" + request_uri
        if query_params is not None:
            req_url = req_url + "?" + _get_query_string(query_params)

        if self._verbose:
            print("GET {}, headers {}".format(req_url, headers))

        res = await self._stub.get(
            url=req_url, headers=self._fix_header(headers))

        if self._verbose:
            print(res)

        return res

    async def _post(self, request_uri, request_body, headers, query_params):
        """Issues the POST request to the server

        Parameters
        ----------
        request_uri: str
            The request URI to be used in POST request.
        request_body: str
            The body of the request
        headers: dict
            Additional HTTP headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        aiohttp.ClientResponse
            The response from server.
        """
        self._validate_headers(headers)
        req_url = self._url + "/" + request_uri
        if query_params is not None:
            req_url = req_url + "?" + _get_query_string(query_params)

        if self._verbose:
            print("POST {}, headers {}\n{}".format(
                req_url, headers, request_body))
        
        if isinstance(request_body, str):
            request_body = request_body.encode()
        res = await self._stub.post(
            url=req_url, data=request_body, headers=self._fix_header(headers))

        if self._verbose:
            print(res)

        return res

    def _validate_headers(self, headers):
        """Checks for any unsupported HTTP headers before processing a request.

        Parameters
        ----------
        headers: dict
            HTTP headers to validate before processing the request.

        Raises
        ------
        InferenceServerException
            If an unsupported HTTP header is included in a request.
        """
        if not headers:
            return

        # HTTP headers are case-insensitive, so force lowercase for comparison
        headers_lowercase = {k.lower(): v for k, v in headers.items()}
        # The python client lirary (and geventhttpclient) do not encode request
        # data based on "Transfer-Encoding" header, so reject this header if
        # included. Other libraries may do this encoding under the hood.
        # The python client library does expose special arguments to support
        # some "Content-Encoding" headers.
        if "transfer-encoding" in headers_lowercase:
            raise_error("Unsupported HTTP header: 'Transfer-Encoding' is not "
                        "supported in the Python client library. Use raw HTTP "
                        "request libraries or the C++ client instead for this "
                        "header.")
    
    def _fix_header(self, headers):
        """Returns a header that is valid for aiohttp.
        
        Parameters
        ----------
        headers: dict (or None)
            HTTP headers to fix before processing the request.
            
        """
        if headers is None:
            return None
        # Convert the headers to only consist of str type
        fix_header = {}
        for key, value in headers.items():
            fix_header[str(key)] = str(value)
        return fix_header

    async def is_server_live(self, headers=None, query_params=None):
        """Contact the inference server and get liveness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        Exception
            If unable to get liveness.

        """

        request_uri = "v2/health/live"
        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)

        return response.status == 200

    async def is_server_ready(self, headers=None, query_params=None):
        """Contact the inference server and get readiness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        Exception
            If unable to get readiness.

        """
        request_uri = "v2/health/ready"
        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)

        return response.status == 200

    async def is_model_ready(self,
                             model_name,
                             model_version="",
                             headers=None,
                             query_params=None):
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
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        bool
            True if the model is ready, False if not ready.

        Raises
        ------
        Exception
            If unable to get model readiness.

        """
        if type(model_version) != str:
            raise_error("model version must be a string")
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}/ready".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}/ready".format(quote(model_name))

        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)

        return response.status == 200

    async def get_server_metadata(self, headers=None, query_params=None):
        """Contact the inference server and get its metadata.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        dict
            The JSON dict holding the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get server metadata.

        """
        request_uri = "v2"
        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def get_model_metadata(self,
                                 model_name,
                                 model_version="",
                                 headers=None,
                                 query_params=None):
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
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get model metadata.

        """
        if type(model_version) != str:
            raise_error("model version must be a string")
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}".format(quote(model_name))

        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def get_model_config(self,
                               model_name,
                               model_version="",
                               headers=None,
                               query_params=None):
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
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the model config.

        Raises
        ------
        InferenceServerException
            If unable to get model configuration.

        """
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}/config".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}/config".format(quote(model_name))

        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def get_model_repository_index(self, headers=None, query_params=None):
        """Get the index of model repository contents

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the model repository index.

        Raises
        ------
        InferenceServerException
            If unable to get the repository index.

        """
        request_uri = "v2/repository/index"
        response = await self._post(request_uri=request_uri,
                                    request_body="",
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def load_model(self,
                         model_name,
                         headers=None,
                         query_params=None,
                         config=None,
                         files=None):
        """Request the inference server to load or reload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.
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

        Raises
        ------
        InferenceServerException
            If unable to load the model.

        """
        request_uri = "v2/repository/models/{}/load".format(quote(model_name))
        load_request = {}
        if config is not None:
            if "parameters" not in load_request:
                load_request["parameters"] = {}
            load_request["parameters"]["config"] = config
        if files is not None:
            for path, content in files.items():
                if "parameters" not in load_request:
                    load_request["parameters"] = {}
                load_request["parameters"][path] = base64.b64encode(content)
        response = await self._post(request_uri=request_uri,
                                    request_body=json.dumps(load_request),
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)
        if self._verbose:
            print("Loaded model '{}'".format(model_name))

    async def unload_model(self,
                           model_name,
                           headers=None,
                           query_params=None,
                           unload_dependents=False):
        """Request the inference server to unload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be unloaded.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction
        unload_dependents : bool
            Whether the dependents of the model should also be unloaded.

        Raises
        ------
        InferenceServerException
            If unable to unload the model.

        """
        request_uri = "v2/repository/models/{}/unload".format(quote(model_name))
        unload_request = {
            "parameters": {
                "unload_dependents": unload_dependents
            }
        }
        response = await self._post(request_uri=request_uri,
                                    request_body=json.dumps(unload_request),
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)
        if self._verbose:
            print("Loaded model '{}'".format(model_name))

    async def get_inference_statistics(self,
                                       model_name="",
                                       model_version="",
                                       headers=None,
                                       query_params=None):
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
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the model inference statistics.

        Raises
        ------
        InferenceServerException
            If unable to get the model inference statistics.

        """

        if model_name != "":
            if type(model_version) != str:
                raise_error("model version must be a string")
            if model_version != "":
                request_uri = "v2/models/{}/versions/{}/stats".format(
                    quote(model_name), model_version)
            else:
                request_uri = "v2/models/{}/stats".format(quote(model_name))
        else:
            request_uri = "v2/models/stats"

        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def update_trace_settings(self,
                                    model_name=None,
                                    settings={},
                                    headers=None,
                                    query_params=None):
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
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the updated trace settings.

        Raises
        ------
        InferenceServerException
            If unable to update the trace settings.

        """

        if (model_name is not None) and (model_name != ""):
            request_uri = "v2/models/{}/trace/setting".format(quote(model_name))
        else:
            request_uri = "v2/trace/setting"

        response = await self._post(request_uri=request_uri,
                                    request_body=json.dumps(settings),
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def get_trace_settings(self,
                                 model_name=None,
                                 headers=None,
                                 query_params=None):
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
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding the trace settings.

        Raises
        ------
        InferenceServerException
            If unable to get the trace settings.

        """

        if (model_name is not None) and (model_name != ""):
            request_uri = "v2/models/{}/trace/setting".format(quote(model_name))
        else:
            request_uri = "v2/trace/setting"

        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def get_system_shared_memory_status(self,
                                              region_name="",
                                              headers=None,
                                              query_params=None):
        """Request system shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active system shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding system shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """
        if region_name != "":
            request_uri = "v2/systemsharedmemory/region/{}/status".format(
                quote(region_name))
        else:
            request_uri = "v2/systemsharedmemory/status"

        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def register_system_shared_memory(self,
                                            name,
                                            key,
                                            byte_size,
                                            offset=0,
                                            headers=None,
                                            query_params=None):
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
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to register the specified system shared memory.

        """
        request_uri = "v2/systemsharedmemory/region/{}/register".format(
            quote(name))

        register_request = {
            'key': key,
            'offset': offset,
            'byte_size': byte_size
        }
        request_body = json.dumps(register_request)

        response = await self._post(request_uri=request_uri,
                                    request_body=request_body,
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)
        if self._verbose:
            print("Registered system shared memory with name '{}'".format(name))

    async def unregister_system_shared_memory(self,
                                              name="",
                                              headers=None,
                                              query_params=None):
        """Request the server to unregister a system shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the system shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified system shared memory region.

        """
        if name != "":
            request_uri = "v2/systemsharedmemory/region/{}/unregister".format(
                quote(name))
        else:
            request_uri = "v2/systemsharedmemory/unregister"

        response = await self._post(request_uri=request_uri,
                                    request_body="",
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)
        if self._verbose:
            if name != "":
                print("Unregistered system shared memory with name '{}'".format(
                    name))
            else:
                print("Unregistered all system shared memory regions")

    async def get_cuda_shared_memory_status(self,
                                            region_name="",
                                            headers=None,
                                            query_params=None):
        """Request cuda shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active cuda shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Returns
        -------
        dict
            The JSON dict holding cuda shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """
        if region_name != "":
            request_uri = "v2/cudasharedmemory/region/{}/status".format(
                quote(region_name))
        else:
            request_uri = "v2/cudasharedmemory/status"

        response = await self._get(request_uri=request_uri,
                                   headers=headers,
                                   query_params=query_params)
        await _raise_if_error(response)

        content = await response.read()
        if self._verbose:
            print(content)

        return json.loads(content)

    async def register_cuda_shared_memory(self,
                                          name,
                                          raw_handle,
                                          device_id,
                                          byte_size,
                                          headers=None,
                                          query_params=None):
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
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to register the specified cuda shared memory.

        """
        request_uri = "v2/cudasharedmemory/region/{}/register".format(
            quote(name))

        register_request = {
            'raw_handle': {
                'b64': raw_handle
            },
            'device_id': device_id,
            'byte_size': byte_size
        }
        request_body = json.dumps(register_request)

        response = await self._post(request_uri=request_uri,
                                    request_body=request_body,
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)
        if self._verbose:
            print("Registered cuda shared memory with name '{}'".format(name))

    async def unregister_cuda_shared_memory(self,
                                            name="",
                                            headers=None,
                                            query_params=None):
        """Request the server to unregister a cuda shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the cuda shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional
            HTTP headers to include in the request
        query_params: dict
            Optional url query parameters to use in network
            transaction

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified cuda shared memory region.

        """
        if name != "":
            request_uri = "v2/cudasharedmemory/region/{}/unregister".format(
                quote(name))
        else:
            request_uri = "v2/cudasharedmemory/unregister"

        response = await self._post(request_uri=request_uri,
                                    request_body="",
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)
        if self._verbose:
            if name != "":
                print("Unregistered cuda shared memory with name '{}'".format(
                    name))
            else:
                print("Unregistered all cuda shared memory regions")

    @staticmethod
    def generate_request_body(inputs,
                              outputs=None,
                              request_id="",
                              sequence_id=0,
                              sequence_start=False,
                              sequence_end=False,
                              priority=0,
                              timeout=None):
        """Generate a request body for inference using the supplied 'inputs'
        requesting the outputs specified by 'outputs'.

        Parameters
        ----------
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id: str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int or str
            The unique identifier for the sequence being represented by the
            object. A value of 0 or "" means that the request does not
            belong to a sequence. Default is 0.
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
            for the model.

        Returns
        -------
        Bytes
            The request body of the inference.
        Int
            The byte size of the inference request header in the request body.
            Returns None if the whole request body constitutes the request header.
            

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """
        return _get_inference_request(inputs=inputs,
                                      request_id=request_id,
                                      outputs=outputs,
                                      sequence_id=sequence_id,
                                      sequence_start=sequence_start,
                                      sequence_end=sequence_end,
                                      priority=priority,
                                      timeout=timeout)

    @staticmethod
    def parse_response_body(response_body,
                            verbose=False,
                            header_length=None,
                            content_encoding=None):
        """Generate a InferResult object from the given 'response_body'

        Parameters
        ----------
        response_body : bytes
            The inference response from the server
        verbose : bool
            If True generate verbose output. Default value is False.
        header_length : int
            The length of the inference header if the header does not occupy
            the whole response body. Default value is None.
        content_encoding : string
            The encoding of the response body if it is compressed.
            Default value is None.
        
        Returns
        -------
        InferResult
            The InferResult object generated from the response body
        """
        return InferResult.from_response_body(response_body, verbose,
                                              header_length, content_encoding)

    async def infer(self,
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
                    headers=None,
                    query_params=None,
                    request_compression_algorithm=None,
                    response_compression_algorithm=None):
        """Run synchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

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
        request_id: str
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
            for the model.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.
        request_compression_algorithm : str
            Optional HTTP compression algorithm to use for the request body on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.
        response_compression_algorithm : str
            Optional HTTP compression algorithm to request for the response body.
            Note that the response may not be compressed if the server does not
            support the specified algorithm. Currently supports "deflate",
            "gzip" and None. By default, no compression is requested.

        Returns
        -------
        InferResult
            The object holding the result of the inference.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """

        request_body, json_size = _get_inference_request(
            inputs=inputs,
            request_id=request_id,
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout)

        if request_compression_algorithm == "gzip":
            if headers is None:
                headers = {}
            headers["Content-Encoding"] = "gzip"
            request_body = gzip.compress(request_body)
        elif request_compression_algorithm == 'deflate':
            if headers is None:
                headers = {}
            headers["Content-Encoding"] = "deflate"
            # "Content-Encoding: deflate" actually means compressing in zlib structure
            # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Encoding
            request_body = zlib.compress(request_body)

        if response_compression_algorithm == "gzip":
            if headers is None:
                headers = {}
            headers["Accept-Encoding"] = "gzip"
        elif response_compression_algorithm == 'deflate':
            if headers is None:
                headers = {}
            headers["Accept-Encoding"] = "deflate"

        if json_size is not None:
            if headers is None:
                headers = {}
            headers["Inference-Header-Content-Length"] = json_size

        if type(model_version) != str:
            raise_error("model version must be a string")
        if model_version != "":
            request_uri = "v2/models/{}/versions/{}/infer".format(
                quote(model_name), model_version)
        else:
            request_uri = "v2/models/{}/infer".format(quote(model_name))

        response = await self._post(request_uri=request_uri,
                                    request_body=request_body,
                                    headers=headers,
                                    query_params=query_params)
        await _raise_if_error(response)

        content_encoding = response.headers.get("Content-Encoding", None)
        header_length = response.headers.get(
            "Inference-Header-Content-Length", None)
        response_body = await response.read()

        return InferResult.from_response_body(
            response_body, self._verbose, header_length, content_encoding)
