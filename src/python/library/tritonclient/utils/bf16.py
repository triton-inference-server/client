// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sys

def BF16Data:
  def __init__(self):
    self.data = np.array(dtype=np.float32)
  
  def from_bytes(self, byte_arr): 
    try:
      self.data = return np.array(byte_arr, dtype=np.float32)
    except Exception as e:
      print(e)

  def from_numpy(self, np_arr):
    try:
      self.data = return np.array(np_arr, dtype=np.float32)
    except Exception as e:
      print(e)

  def to_numpy(self):
    return self.data

  def to_bytes(self):
    return self.truncate(self.data) 

  def truncate(self, data):
    # Truncates float32 data to bf16 data
    bytes = bytearray()
    for num in data:
        high_order_bytes = (num & 0xffff0000)
        bytes.extend(high_order_bytes.tobytes()[2:4])
    return bytes

  def triton_dtype(self):
    return "BF16"

  def np_dtype(self):
    return self.data.dtype
