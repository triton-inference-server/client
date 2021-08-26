package com.nvidia.triton.contrib.pojo;

/**
 * Data types from
 */
public enum DataType {

    BOOL(1, false),
    INT8(1, true),
    INT16(2, true),
    INT32(4, true),
    INT64(8, true),
    UINT8(1, false),
    UINT16(2, false),
    UINT32(4, false),
    UINT64(8, false),
    FP16(2, false),
    FP32(4, false),
    FP64(8, false),
    BYTES(-1, false);

    public final int numByte;
    public final boolean singed;

    DataType(int numByte, boolean singed) {
        this.numByte = numByte;
        this.singed = singed;
    }
}












