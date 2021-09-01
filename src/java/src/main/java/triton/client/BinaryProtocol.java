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

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.function.BiConsumer;

import triton.client.pojo.DataType;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Booleans;
import com.google.common.primitives.Bytes;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.common.primitives.Shorts;

/**
 * @author xiafei.qiuxf
 * @date 2021/4/20
 */
public class BinaryProtocol {

    private static <T> byte[] setBinaryDataImpl(DataType dataType, List<T> data, BiConsumer<ByteBuffer, T> consumer) {
        byte[] binaryData = new byte[data.size() * dataType.numByte];
        ByteBuffer buf = ByteBuffer.wrap(binaryData);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        for (T datum : data) {
            consumer.accept(buf, datum);
        }
        Preconditions.checkState(buf.position() == binaryData.length);
        return binaryData;
    }

    public static byte[] toBytes(DataType dataType, boolean[] data) {
        return setBinaryDataImpl(dataType, Booleans.asList(data), (buf, b) -> buf.put(b ? (byte)1 : (byte)0));
    }

    public static byte[] toBytes(DataType dataType, byte[] data) {
        return setBinaryDataImpl(dataType, Bytes.asList(data), ByteBuffer::put);
    }

    public static byte[] toBytes(DataType dataType, short[] data) {
        return setBinaryDataImpl(dataType, Shorts.asList(data), ByteBuffer::putShort);
    }

    public static byte[] toBytes(DataType dataType, int[] data) {
        return setBinaryDataImpl(dataType, Ints.asList(data), ByteBuffer::putInt);
    }

    public static byte[] toBytes(DataType dataType, long[] data) {
        return setBinaryDataImpl(dataType, Longs.asList(data), ByteBuffer::putLong);
    }

    public static byte[] toBytes(DataType dataType, float[] data) {
        return setBinaryDataImpl(dataType, Floats.asList(data), ByteBuffer::putFloat);
    }


    public static byte[] toBytes(DataType dataType, double[] data) {
        return setBinaryDataImpl(dataType, Doubles.asList(data), ByteBuffer::putDouble);
    }

    public static byte[] toBytes(DataType dataType, String[] data) {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        for (String datum : data) {
            byte[] bytes = datum.getBytes(StandardCharsets.UTF_8);
            try {
                o.write(Util.intToBytes(bytes.length));
                o.write(bytes);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        return o.toByteArray();
    }
}
