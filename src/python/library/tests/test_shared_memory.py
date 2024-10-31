# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tritonclient.utils as utils
import tritonclient.utils.shared_memory as shm


class SharedMemoryTest(unittest.TestCase):
    """
    Testing shared memory utilities
    """

    def setUp(self):
        self.shm_handles = []

    def tearDown(self):
        for shm_handle in self.shm_handles:
            shm.destroy_shared_memory_region(shm_handle)

    def test_lifecycle(self):
        cpu_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        byte_size = 64
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", byte_size)
        )

        self.assertEqual(len(shm.mapped_shared_memory_regions()), 1)

        # Set data from Numpy array
        shm.set_shared_memory_region(self.shm_handles[0], [cpu_tensor])
        shm_tensor = shm.get_contents_as_numpy(
            self.shm_handles[0], numpy.float32, [4, 4]
        )

        self.assertTrue(numpy.allclose(cpu_tensor, shm_tensor))

        shm.destroy_shared_memory_region(self.shm_handles.pop(0))

    def test_invalid_create_shm(self):
        # Raises error since tried to create invalid system shared memory region
        with self.assertRaisesRegex(
            shm.SharedMemoryException, "unable to create the shared memory region"
        ):
            self.shm_handles.append(
                shm.create_shared_memory_region("dummy_data", "/dummy_data", -1)
            )

    def test_set_region_offset(self):
        large_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        large_size = 64
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", large_size)
        )
        shm.set_shared_memory_region(self.shm_handles[0], [large_tensor])
        small_tensor = numpy.zeros([2, 4], dtype=numpy.float32)
        small_size = 32
        shm.set_shared_memory_region(
            self.shm_handles[0], [small_tensor], offset=large_size - small_size
        )
        shm_tensor = shm.get_contents_as_numpy(
            self.shm_handles[0], numpy.float32, [2, 4], offset=large_size - small_size
        )

        self.assertTrue(numpy.allclose(small_tensor, shm_tensor))

    def test_set_region_oversize(self):
        large_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        small_size = 32
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", small_size)
        )
        with self.assertRaisesRegex(
            shm.SharedMemoryException, "unable to set the shared memory region"
        ):
            shm.set_shared_memory_region(self.shm_handles[0], [large_tensor])

    def test_duplicate_key(self):
        # by default, return the same handle if existed, warning will be print
        # if size is different
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 32)
        )
        with self.assertRaisesRegex(
            shm.SharedMemoryException,
            "unable to create the shared memory region",
        ):
            self.shm_handles.append(
                shm.create_shared_memory_region(
                    "shm_name", "shm_key", 32, create_only=True
                )
            )

        # Get handle to the same shared memory region but with larger size requested,
        # check if actual size is checked
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 64)
        )

        self.assertEqual(len(shm.mapped_shared_memory_regions()), 1)

        large_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        with self.assertRaisesRegex(
            shm.SharedMemoryException, "unable to set the shared memory region"
        ):
            shm.set_shared_memory_region(self.shm_handles[-1], [large_tensor])

    def test_destroy_duplicate(self):
        # destruction of duplicate shared memory region will occur when the last
        # managed handle is destroyed
        self.assertEqual(len(shm.mapped_shared_memory_regions()), 0)
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 64)
        )
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 32)
        )
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 32)
        )
        self.assertEqual(len(shm.mapped_shared_memory_regions()), 1)

        shm.destroy_shared_memory_region(self.shm_handles.pop(0))
        shm.destroy_shared_memory_region(self.shm_handles.pop(0))
        self.assertEqual(len(shm.mapped_shared_memory_regions()), 1)

        shm.destroy_shared_memory_region(self.shm_handles.pop(0))
        self.assertEqual(len(shm.mapped_shared_memory_regions()), 0)

    def test_numpy_bytes(self):
        int_tensor = numpy.arange(start=0, stop=16, dtype=numpy.int32)
        bytes_tensor = numpy.array(
            [str(x).encode("utf-8") for x in int_tensor.flatten()], dtype=object
        )
        bytes_tensor = bytes_tensor.reshape(int_tensor.shape)
        bytes_tensor_serialized = utils.serialize_byte_tensor(bytes_tensor)
        byte_size = utils.serialized_byte_size(bytes_tensor_serialized)

        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", byte_size)
        )

        # Set data from Numpy array
        shm.set_shared_memory_region(self.shm_handles[0], [bytes_tensor_serialized])

        shm_tensor = shm.get_contents_as_numpy(
            self.shm_handles[0],
            numpy.object_,
            [
                16,
            ],
        )

        self.assertTrue(numpy.array_equal(bytes_tensor, shm_tensor))


if __name__ == "__main__":
    unittest.main()
