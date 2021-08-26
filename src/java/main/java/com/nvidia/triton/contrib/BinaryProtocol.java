package com.nvidia.triton.contrib;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.function.BiConsumer;

import com.nvidia.triton.contrib.pojo.DataType;
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
