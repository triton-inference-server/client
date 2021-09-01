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

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import triton.client.pojo.DataType;
import triton.client.pojo.IOTensor;
import triton.client.pojo.InferenceResponse;
import triton.client.pojo.Parameters;
import triton.client.pojo.ResponseError;
import org.apache.commons.io.IOUtils;
import org.apache.http.Header;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;

/**
 * An object of InferResult class holds the response of an inference request and provide methods to retrieve inference
 * results.
 */
public class InferResult {

    static class Index {
        int start;
        int length;

        public Index(int start, int length) {
            this.start = start;
            this.length = length;
        }
    }

    private final InferenceResponse response;
    private final Map<String, Index> nameToBinaryIdx;
    private final byte[] binaryData;

    public InferResult(HttpResponse resp) throws IOException, InferenceException {
        HttpEntity entity = resp.getEntity();
        Preconditions.checkState(entity != null, "Get null entity from HTTP response.");
        InputStream stream = entity.getContent();

        int httpCode = resp.getStatusLine().getStatusCode();
        if (httpCode != HttpStatus.SC_OK) {
            byte[] bodyBytes = IOUtils.toByteArray(stream);
            if (bodyBytes.length > 0) {
                String bodyJson = new String(bodyBytes, Charsets.UTF_8);
                try {
                    ResponseError err = Util.fromJson(bodyJson, ResponseError.class);
                    throw new InferenceException(err.getError());
                } catch (JsonProcessingException e) {
                    throw new InferenceException("Malformed error response: " + bodyJson);
                }
            } else {
                throw new InferenceException("Inference return status: " + httpCode);
            }
        }

        this.nameToBinaryIdx = new HashMap<>();
        Header contentLenHeader = resp.getFirstHeader("Inference-Header-Content-Length");
        if (contentLenHeader != null) {
            // Construct response object from json part.
            int jsonLen = Integer.parseInt(contentLenHeader.getValue());
            byte[] bodyBytes = new byte[jsonLen];
            int readLen = stream.read(bodyBytes);
            Preconditions.checkState(readLen == jsonLen,
                "Expect content length: %d, but got %d.", jsonLen, readLen);
            String bodyJson = new String(bodyBytes, Charsets.UTF_8);
            this.response = Util.fromJson(bodyJson, InferenceResponse.class);

            // Construct name to binary index mapping.
            int startPos = 0;
            for (IOTensor output : this.response.getOutputs()) {
                Parameters param = output.getParameters();
                if (param == null) { continue; }
                Integer size = param.getInt(Parameters.KEY_BINARY_DATA_SIZE);
                if (size == null) { continue; }
                this.nameToBinaryIdx.put(output.getName(), new Index(startPos, size));
                startPos += size;
            }

            // Read and check binary data.
            this.binaryData = IOUtils.toByteArray(stream);
            Preconditions.checkState(this.binaryData.length == startPos);
        } else {
            String bodyJson = new String(IOUtils.toByteArray(stream), Charsets.UTF_8);
            this.response = Util.fromJson(bodyJson, InferenceResponse.class);
            this.binaryData = null;
        }
    }

    @VisibleForTesting
    InferResult(InferenceResponse response, Map<String, Index> nameToBinaryIdx, byte[] binaryData) {
        this.response = response;
        this.nameToBinaryIdx = nameToBinaryIdx;
        this.binaryData = binaryData;
    }

    @VisibleForTesting
    public InferenceResponse getResponse() {
        return response;
    }

    @VisibleForTesting
    public Map<String, Index> getNameToBinaryIdx() {
        return nameToBinaryIdx;
    }

    @VisibleForTesting
    public byte[] getBinaryData() {
        return binaryData;
    }

    public List<String> getOutputs() {
        List<IOTensor> outputs = this.response.getOutputs();
        List<String> ret = new ArrayList<>(outputs.size());
        for (IOTensor out : outputs) {
            ret.add(out.getName());
        }
        return ret;
    }

    /**
     * Get boolean tensor named as by parameter output. The tensor must be of DataType.BOOL.
     *
     * @param output name of output tensor.
     * @return null if output not found or the tensor in boolean array.
     */
    public boolean[] getOutputAsBool(String output) {
        IOTensor out = this.response.getOutputByName(output);
        if (out == null) {
            return null;
        }
        Preconditions.checkArgument(out.getDatatype() == DataType.BOOL,
            "Could not get boolean[] from data of type %s on output %s.", out.getDatatype(), out.getName());
        return (boolean[])getOutputImpl(out, boolean.class, buf -> buf.get() != 0);
    }

    /**
     * Get boolean tensor named as by parameter output. The tensor must be of DataType.INT8 or DataType.UINT8.
     *
     * @param output name of output tensor.
     * @return null if output not found or the tensor in byte array.
     */
    public byte[] getOutputAsByte(String output) {
        IOTensor out = this.response.getOutputByName(output);
        if (out == null) {
            return null;
        }
        Preconditions.checkArgument(out.getDatatype() == DataType.INT8 || out.getDatatype() == DataType.UINT8,
            "Could not get byte[] from data of type %s on output %s.", out.getDatatype(), out.getName());
        return (byte[])getOutputImpl(out, byte.class, ByteBuffer::get);
    }

    /**
     * Get boolean tensor named as by parameter output. The tensor must be of DataType.INT16 or DataType.UINT16.
     *
     * @param output name of output tensor.
     * @return null if output not found or the tensor in short array.
     */
    public short[] getOutputAsShort(String output) {
        IOTensor out = this.response.getOutputByName(output);
        if (out == null) {
            return null;
        }
        Preconditions.checkArgument(out.getDatatype() == DataType.INT16 || out.getDatatype() == DataType.UINT16,
            "Could not get short[] from data of type %s on output %s.", out.getDatatype(), out.getName());
        return (short[])getOutputImpl(out, short.class, ByteBuffer::getShort);
    }

    /**
     * Get boolean tensor named as by parameter output. The tensor must be of DataType.INT32 or DataType.UINT32.
     *
     * @param output name of output tensor.
     * @return null if output not found or the tensor in int array.
     */
    public int[] getOutputAsInt(String output) {
        IOTensor out = this.response.getOutputByName(output);
        if (out == null) {
            return null;
        }
        Preconditions.checkArgument(out.getDatatype() == DataType.INT32 || out.getDatatype() == DataType.UINT32,
            "Could not get int[] from data of type %s on output %s.", out.getDatatype(), out.getName());
        return (int[])getOutputImpl(out, int.class, ByteBuffer::getInt);
    }

    /**
     * Get boolean tensor named as by parameter output. The tensor must be of DataType.INT64 or DataType.UINT64.
     *
     * @param output name of output tensor.
     * @return null if output not found or the tensor in long array.
     */
    public long[] getOutputAsLong(String output) {
        IOTensor out = this.response.getOutputByName(output);
        if (out == null) {
            return null;
        }
        Preconditions.checkArgument(out.getDatatype() == DataType.INT64 || out.getDatatype() == DataType.UINT64,
            "Could not get long[] from data of type %s on output %s.", out.getDatatype(), out.getName());
        return (long[])getOutputImpl(out, long.class, ByteBuffer::getLong);
    }

    /**
     * Get boolean tensor named as by parameter output. The tensor must be of DataType.FP32.
     *
     * @param output name of output tensor.
     * @return null if output not found or the tensor in float array.
     */
    public float[] getOutputAsFloat(String output) {
        IOTensor out = this.response.getOutputByName(output);
        if (out == null) {
            return null;
        }
        Preconditions.checkArgument(out.getDatatype() == DataType.FP32,
            "Could not get float[] from data of type %s on output %s.", out.getDatatype(), out.getName());
        return (float[])getOutputImpl(out, float.class, ByteBuffer::getFloat);
    }

    /**
     * Get boolean tensor named as by parameter output. The tensor must be of DataType.FP64.
     *
     * @param output name of output tensor.
     * @return null if output not found or the tensor in double array.
     */
    public double[] getOutputAsDouble(String output) {
        IOTensor out = this.response.getOutputByName(output);
        if (out == null) {
            return null;
        }
        Preconditions.checkArgument(out.getDatatype() == DataType.FP64,
            "Could not get double[] from data of type %s on output %s.", out.getDatatype(), out.getName());
        return (double[])getOutputImpl(out, double.class, ByteBuffer::getDouble);
    }

    private <T> Object getOutputImpl(IOTensor out, Class<T> clazz, Function<ByteBuffer, T> getter) {
        Index idx = this.nameToBinaryIdx.get(out.getName());
        if (idx != null) { // Output in binary format.
            Preconditions.checkState(this.binaryData != null);
            long numElem = Util.elemNumFromShape(out.getShape());
            Object array = Array.newInstance(clazz, (int)numElem);
            ByteBuffer buf = ByteBuffer.wrap(this.binaryData, idx.start, idx.length);
            buf.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < numElem; i++) {
                Array.set(array, i, getter.apply(buf));
            }
            return array;
        } else { // Output in json format.
            Object[] data = out.getData();
            Object array = Array.newInstance(clazz, data.length);
            for (int i = 0; i < data.length; i++) {
                Array.set(array, i, Util.numericCast(data[i], clazz));
            }
            return array;
        }
    }
}
