// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package triton.client;

import triton.client.pojo.Parameters;
import triton.client.pojo.IOTensor;

/**
 * @author xiafei.qiuxf
 * @date 2021/4/13
 */
public class InferRequestedOutput {
    /**
     * The name of output tensor to associate with this object.
     */
    private final String name;
    /**
     * Indicates whether to return result data for the output in
     * binary format or explicit tensor within JSON. The default
     * value is True, which means the data will be delivered as
     * binary data in the HTTP body after JSON object. This field
     * will be unset if shared memory is set for the output.
     */
    private final boolean isBinary;

    /**
     * The number of classifications to be requested. The default
     * value is 0 which means the classification results are not
     * requested.
     */
    private final int classCount;

    private final Parameters parameters = new Parameters();

    public InferRequestedOutput(String name, boolean isBinary, int classCount) {
        this.name = name;
        this.isBinary = isBinary;
        this.classCount = classCount;
        if (this.classCount != 0) {
            this.parameters.put("classification", classCount);
        }
        this.parameters.put("binary_data", this.isBinary);

    }

    public InferRequestedOutput(String name) {
        this(name, true, 0);
    }

    public InferRequestedOutput(String name, boolean isBinary) {
        this(name, isBinary, 0);
    }

    public IOTensor getTensor() {
        IOTensor tensor = new IOTensor();
        tensor.setName(this.name);
        tensor.setParameters(this.parameters);
        return tensor;
    }
}
