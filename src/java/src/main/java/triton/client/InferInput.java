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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Function;

import triton.client.pojo.DataType;
import triton.client.pojo.IOTensor;
import triton.client.pojo.Parameters;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Booleans;
import com.google.common.primitives.Bytes;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.common.primitives.Shorts;
import com.google.common.primitives.UnsignedInteger;
import com.google.common.primitives.UnsignedLong;

/**
 * This class describes an input tensor feeding to inference server, including it's name, shape, datatype and the actual
 * tensor data.
 */
public class InferInput {
    /**
     * Name of inference input.
     */
    private final String name;
    /**
     * Shape of inference input.
     */
    private final long[] shape;
    /**
     * Data type of inference input.
     */
    private final DataType dataType;

    /**
     * Other parameters on this inference input.
     */
    private final Parameters parameters;
    /**
     * Binary representation of tensor data of this input tensor if it's in binary format.
     */
    private byte[] binaryData;
    /**
     * Tensor data of this input tensor if it's in JSON format.
     */
    private Object[] data;

    /**
     * Number of elements in this inference input.
     */
    private final long numElement;

    /**
     * Create an inference input.
     *
     * @param name     name of input.
     * @param shape    shape of input.
     * @param dataType data type of input
     */
    public InferInput(String name, long[] shape, DataType dataType) {
        this.name = name;
        this.shape = shape;
        this.dataType = dataType;
        this.numElement = Util.elemNumFromShape(shape);
        this.parameters = new Parameters();
    }

    private void checkShape(int len) {
        Preconditions.checkArgument(len == this.numElement,
            "Data tensor's size [%s not consist with shape [%s].", len, this.numElement);
    }

    private <T> void setBinaryDataImpl(List<T> data, BiConsumer<ByteBuffer, T> consumer) {
        this.binaryData = new byte[data.size() * dataType.numByte];
        ByteBuffer buf = ByteBuffer.wrap(this.binaryData);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        for (T datum : data) {
            consumer.accept(buf, datum);
        }
        Preconditions.checkState(buf.position() == this.binaryData.length);
        this.parameters.put(Parameters.KEY_BINARY_DATA_SIZE, this.binaryData.length);
    }

    private <T> void setJSONDataImpl(List<T> data) {
        this.parameters.remove(Parameters.KEY_BINARY_DATA_SIZE);
        //this.data = new JSONArray(data.size());
        //this.data.addAll(data);
        this.data = data.toArray(new Object[0]);
    }

    private <T, U> void setJSONDataImpl(List<T> data, Function<T, U> mapper) {
        this.parameters.remove(Parameters.KEY_BINARY_DATA_SIZE);
        //this.data = new JSONArray(data.size());
        //for (T datum : data) {
        //    this.data.add(mapper.apply(datum));
        //}
        this.data = new Object[data.size()];
        int i = 0;
        for (T datum : data) {
            this.data[i++] = mapper.apply(datum);
        }
    }

    /**
     * Set boolean tensor data.
     *
     * @param data         tensor data in java array. It's length must match input shape given in constructor.
     * @param isBinaryData whether it's in binary format.
     */
    public void setData(boolean[] data, boolean isBinaryData) {
        Preconditions.checkArgument(this.dataType == DataType.BOOL,
            "Could not set boolean[] as data of type: %s", this.dataType);
        if (isBinaryData) {
            //setBinaryDataImpl(Booleans.asList(data), (buf, b) -> buf.put(b ? (byte)1 : (byte)0));
            this.binaryData = BinaryProtocol.toBytes(this.dataType, data);
            this.updateBinaryDataSize();
        } else {
            setJSONDataImpl(Booleans.asList(data));
        }
    }

    /**
     * Set INT8/UINT8 tensor data. For unsigned type, top bit should be stored in the sign bit. Note: guava currently
     * had nothing like UnsignedShort.
     *
     * @param data         tensor data in java array. It's length must match input shape given in constructor.
     * @param isBinaryData whether it's in binary format.
     */
    public void setData(byte[] data, boolean isBinaryData) {
        Preconditions.checkArgument(this.dataType == DataType.INT8 || this.dataType == DataType.UINT8,
            "Could not set boolean[] as data of type: %s", this.dataType);
        if (isBinaryData) {
            //setBinaryDataImpl(Bytes.asList(data), ByteBuffer::put);
            this.binaryData = BinaryProtocol.toBytes(this.dataType, data);
            this.updateBinaryDataSize();
        } else if (this.dataType.singed) {
            setJSONDataImpl(Bytes.asList(data));
        } else {
            setJSONDataImpl(Bytes.asList(data), bits -> UnsignedInteger.fromIntBits(bits).longValue());
        }
    }

    /**
     * Set INT16/UINT16 tensor data. For unsigned type, top bit should be stored in the sign bit. You may turn to
     * {@link com.google.common.primitives.Shorts} for help.
     * <p>
     *
     * @param data
     * @param isBinaryData
     */
    public void setData(short[] data, boolean isBinaryData) {
        Preconditions.checkArgument(this.dataType == DataType.INT16 || this.dataType == DataType.UINT16,
            "Could not set boolean[] as data of type: %s", this.dataType);
        if (isBinaryData) {
            //setBinaryDataImpl(Shorts.asList(data), ByteBuffer::putShort);
            this.binaryData = BinaryProtocol.toBytes(this.dataType, data);
            this.updateBinaryDataSize();
        } else if (this.dataType.singed) {
            setJSONDataImpl(Shorts.asList(data));
        } else {
            setJSONDataImpl(Shorts.asList(data), bits -> UnsignedInteger.fromIntBits(bits).longValue());
        }
    }

    /**
     * Set INT32/UINT32 tensor data. For unsigned type, top bit should be stored in the sign bit. You may turn to
     * {@link com.google.common.primitives.UnsignedInteger} and {@link com.google.common.primitives.UnsignedInts} for
     * help.
     *
     * @param data         tensor data in java array. It's length must match input shape given in constructor.
     * @param isBinaryData whether it's in binary format.
     */
    public void setData(int[] data, boolean isBinaryData) {
        Preconditions.checkArgument(this.dataType == DataType.INT32 || this.dataType == DataType.UINT32,
            "Could not set boolean[] as data of type: %s", this.dataType);
        if (isBinaryData) {
            //setBinaryDataImpl(Ints.asList(data), ByteBuffer::putInt);
            this.binaryData = BinaryProtocol.toBytes(this.dataType, data);
            this.updateBinaryDataSize();
        } else {
            setJSONDataImpl(Ints.asList(data));
            if (this.dataType.singed) {
                setJSONDataImpl(Ints.asList(data));
            } else { // uint32
                setJSONDataImpl(Ints.asList(data), bits -> UnsignedInteger.fromIntBits(bits).longValue());
            }
        }
    }

    /**
     * Set INT64/UINT64 tensor data. For UINT64, top bit should be stored in the sign bit. You may turn
     * to {@link com.google.common.primitives.UnsignedLong} and {@link com.google.common.primitives.UnsignedLongs} for
     * help.
     *
     * @param data         tensor data in java array. It's length must match input shape given in constructor.
     * @param isBinaryData whether it's in binary format.
     */
    public void setData(long[] data, boolean isBinaryData) {
        this.checkShape(data.length);
        Preconditions.checkArgument(this.dataType == DataType.INT64 || this.dataType == DataType.UINT64,
            "Could not set long[] as data of type: %s", this.dataType);
        if (isBinaryData) {
            this.binaryData = BinaryProtocol.toBytes(this.dataType, data);
            this.updateBinaryDataSize();
        } else if (this.dataType.singed) {
            setJSONDataImpl(Longs.asList(data));
        } else {
            // TODO(xiafei.qiuxf): This is a little bit inefficient, maybe we can write a custom json
            //  serializer/deserializer for unsigned types.
            setJSONDataImpl(Longs.asList(data), bits -> UnsignedLong.fromLongBits(bits).bigIntegerValue());
        }
    }

    /**
     * Set FP32 tensor data.
     *
     * @param data         tensor data in java array. It's length must match input shape given in constructor.
     * @param isBinaryData whether it's in binary format.
     */
    public void setData(float[] data, boolean isBinaryData) {
        this.checkShape(data.length);
        Preconditions.checkArgument(this.dataType == DataType.FP32,
            "Could not set float[] as data of type: %s", this.dataType);
        if (isBinaryData) {
            this.binaryData = BinaryProtocol.toBytes(this.dataType, data);
            this.updateBinaryDataSize();
        } else {
            setJSONDataImpl(Floats.asList(data));
        }
    }

    /**
     * Set FP64 tensor data.
     *
     * @param data         tensor data in java array. It's length must match input shape given in constructor.
     * @param isBinaryData whether it's in binary format.
     */
    public void setData(double[] data, boolean isBinaryData) {
        this.checkShape(data.length);
        Preconditions.checkArgument(this.dataType == DataType.FP64,
            "Could not set double[] as data of type: %s", this.dataType);
        if (isBinaryData) {
            this.binaryData = BinaryProtocol.toBytes(this.dataType, data);
            this.updateBinaryDataSize();
        } else {
            setJSONDataImpl(Doubles.asList(data));
        }
    }

    /**
     * Set boolean tensor data.
     *
     * @param data         tensor data in java array. It's length must match input shape given in constructor.
     * @param isBinaryData whether it's in binary format.
     */
    public void setData(String[] data, boolean isBinaryData) {
        this.checkShape(data.length);
        Preconditions.checkArgument(this.dataType == DataType.BYTES,
            "Could not set String[] as data of type: %s", this.dataType);
        if (isBinaryData) {
            this.binaryData = BinaryProtocol.toBytes(this.dataType, data);
            this.updateBinaryDataSize();
        } else {
            //this.data = new JSONArray(data.length);
            //this.data.addAll(Arrays.asList(data));
            this.data = new Object[data.length];
            System.arraycopy(data, 0, this.data, 0, data.length);
        }
    }

    private void updateBinaryDataSize() {
        this.parameters.put(Parameters.KEY_BINARY_DATA_SIZE, this.binaryData.length);
    }

    public String getName() {
        return name;
    }

    IOTensor getTensor() {
        Preconditions.checkArgument(this.binaryData != null || this.data != null,
            ".setData method not call on InferInput %s", this.name);
        IOTensor tensor = new IOTensor();
        tensor.setName(this.name);
        tensor.setDatatype(this.dataType);
        tensor.setShape(this.shape);
        tensor.setParameters(this.parameters);
        tensor.setData(this.data);
        return tensor;
    }

    /**
     * Get binary representation of tensor data of this inference input.
     *
     * @return null if this inference input is in JSON format.
     */
    byte[] getBinaryData() {
        return binaryData;
    }

    Object[] getJSONData() {
        return this.data;
    }
}
