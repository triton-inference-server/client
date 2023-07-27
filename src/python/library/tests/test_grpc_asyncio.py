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
import unittest

from tritonclient.grpc.aio import InferenceServerClient


class GRPCAsyncIOTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self._client = InferenceServerClient(url="localhost:8001")
        self._model_name = "resnet50"

    async def test_server_live(self):
        self.assertTrue(await self._client.is_server_live())

    async def test_server_ready(self):
        self.assertTrue(await self._client.is_server_ready())

    async def test_is_model_ready(self):
        self.assertTrue(await self._client.is_model_ready(self._model_name))

    async def test_get_server_metadata(self):
        server_metadata = await self._client.get_server_metadata()
        self.assertIn("trace", server_metadata.extensions)

        server_metadata = await self._client.get_server_metadata(as_json=True)
        self.assertIn("trace", server_metadata["extensions"])

    async def test_get_model_metadata(self):
        model_metadata = await self._client.get_model_metadata(self._model_name)
        self.assertEqual(model_metadata.name, self._model_name)

        model_metadata = await self._client.get_model_metadata(
            self._model_name, as_json=True
        )
        self.assertEqual(model_metadata["name"], self._model_name)

    async def test_get_model_config(self):
        model_config = await self._client.get_model_config(self._model_name)
        self.assertEqual(model_config.config.name, self._model_name)

    async def test_get_model_repository_index(self):
        models = await self._client.get_model_repository_index()
        for model in models.models:
            if model.name == self._model_name:
                break
        else:
            self.assertFalse(
                f"Failed to find model ({self._model_name}) in the list of models."
            )

    async def test_model_load_unload(self):
        self.assertTrue(await self._client.is_model_ready(self._model_name))
        await self._client.unload_model(self._model_name)
        self.assertFalse(await self._client.is_model_ready(self._model_name))
        await self._client.load_model(self._model_name)
        self.assertTrue(await self._client.is_model_ready(self._model_name))

    async def test_get_inference_statistics(self):
        statistics = await self._client.get_inference_statistics(self._model_name)
        self.assertEqual(statistics.model_stats[0].name, self._model_name)


if __name__ == "__main__":
    unittest.main()
