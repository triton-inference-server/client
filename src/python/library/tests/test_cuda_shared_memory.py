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

import numpy

# Torch support read / write DLPack object on GPU
import torch
import tritonclient.utils as utils
import tritonclient.utils.cuda_shared_memory as cudashm


class DLPackTest(unittest.TestCase):
    """
    Testing DLPack implementation in CUDA shared memory utilities
    """

    def test_from_gpu(self):
        # Create GPU tensor via PyTorch and CUDA shared memory region with
        # enough space
        gpu_tensor = torch.ones(4, 4).cuda(0)
        byte_size = 64
        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        # Set data from DLPack specification of PyTorch tensor
        cudashm.set_shared_memory_region_from_dlpack(shm_handle, [gpu_tensor])

        # Make sure the DLPack specification of the shared memory region can
        # be consumed by PyTorch
        smt = cudashm.as_shared_memory_tensor(shm_handle, "FP32", [4, 4])
        generated_torch_tensor = torch.from_dlpack(smt)
        self.assertTrue(torch.allclose(gpu_tensor, generated_torch_tensor))

        cudashm.destroy_shared_memory_region(shm_handle)

    def test_from_cpu(self):
        # Create CPU tensor via numpy and CUDA shared memory region with
        # enough space
        cpu_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        byte_size = 64
        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        # Set data from DLPack specification of Numpy array
        cudashm.set_shared_memory_region_from_dlpack(shm_handle, [cpu_tensor])

        # Make sure the DLPack specification of the shared memory region can
        # be consumed by PyTorch.
        # Need to pass to PyTorch first as numpy doesn't consume GPU DLPack
        smt = cudashm.as_shared_memory_tensor(shm_handle, "FP32", [4, 4])
        generated_torch_tensor = torch.from_dlpack(smt)

        self.assertTrue(
            numpy.allclose(cpu_tensor, numpy.from_dlpack(generated_torch_tensor.cpu()))
        )

        cudashm.destroy_shared_memory_region(shm_handle)


class NumpyTest(unittest.TestCase):
    """
    Testing Numpy implementation in CUDA shared memory utilities
    """

    def test_from_numpy(self):
        # Retrieve data from shared memory region via DLPack as it has been
        # verified in 'DLPackTest'

        # Create CPU tensor via numpy and CUDA shared memory region with
        # enough space
        cpu_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        byte_size = 64
        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        # Set data from Numpy array
        cudashm.set_shared_memory_region(shm_handle, [cpu_tensor])

        # Make sure the DLPack specification of the shared memory region can
        # be consumed by PyTorch.
        # Need to pass to PyTorch first as numpy doesn't consume GPU DLPack
        smt = cudashm.as_shared_memory_tensor(shm_handle, "FP32", [4, 4])
        generated_torch_tensor = torch.from_dlpack(smt)

        self.assertTrue(
            numpy.allclose(cpu_tensor, numpy.from_dlpack(generated_torch_tensor.cpu()))
        )

        cudashm.destroy_shared_memory_region(shm_handle)

    def test_to_numpy(self):
        # Retrieve data from shared memory region via DLPack as it has been
        # verified in 'DLPackTest'

        # Create CPU tensor via numpy and CUDA shared memory region with
        # enough space
        cpu_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        byte_size = 64
        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        # Set data from Numpy array
        cudashm.set_shared_memory_region(shm_handle, [cpu_tensor])

        # Make sure the DLPack specification of the shared memory region can
        # be consumed by PyTorch.
        # Need to pass to PyTorch first as numpy doesn't consume GPU DLPack
        generated_tensor = cudashm.get_contents_as_numpy(
            shm_handle, numpy.float32, [4, 4]
        )

        self.assertTrue(numpy.allclose(cpu_tensor, generated_tensor))

        cudashm.destroy_shared_memory_region(shm_handle)

    def test_numpy_bytes(self):
        int_tensor = numpy.arange(start=0, stop=16, dtype=numpy.int32)
        bytes_tensor = numpy.array(
            [str(x).encode("utf-8") for x in int_tensor.flatten()], dtype=object
        )
        bytes_tensor = bytes_tensor.reshape(int_tensor.shape)
        bytes_tensor_serialized = utils.serialize_byte_tensor(bytes_tensor)
        byte_size = utils.serialized_byte_size(bytes_tensor_serialized)

        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        # Set data from Numpy array
        cudashm.set_shared_memory_region(shm_handle, [bytes_tensor_serialized])

        # Make sure the DLPack specification of the shared memory region can
        # be consumed by PyTorch.
        # Need to pass to PyTorch first as numpy doesn't consume GPU DLPack
        generated_tensor = cudashm.get_contents_as_numpy(
            shm_handle,
            numpy.object_,
            [
                16,
            ],
        )

        self.assertTrue(numpy.array_equal(bytes_tensor, generated_tensor))

        cudashm.destroy_shared_memory_region(shm_handle)


if __name__ == "__main__":
    unittest.main()
