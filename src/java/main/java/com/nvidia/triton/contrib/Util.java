package com.nvidia.triton.contrib;

import java.util.Collection;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Util {

    /**
     * Check whether a string object is null or empty.
     *
     * @param s the string object.
     * @return true if it's null or empty
     */
    public static boolean isEmpty(String s) {
        return s == null || s.isEmpty();
    }

    /**
     * Check whether a collection object is null or empty.
     *
     * @param c the collection object.
     * @return true if it's null or empty
     */
    public static boolean isEmpty(Collection<?> c) {
        return c == null || c.isEmpty();
    }

    /**
     * Calculate element number from tensor shape.
     *
     * @param shape tensor shape
     * @return element number
     */
    public static long elemNumFromShape(long[] shape) {
        long ret = 1;
        for (long n : shape) {
            ret *= n;
        }
        return ret;
    }

    /**
     * Convert int to bytes in little endian.
     *
     * @param a
     * @return
     */
    public static byte[] intToBytes(int a) {
        byte[] ret = new byte[4];
        ret[0] = (byte)(a & 0xFF);
        ret[1] = (byte)((a >> 8) & 0xFF);
        ret[2] = (byte)((a >> 16) & 0xFF);
        ret[3] = (byte)((a >> 24) & 0xFF);
        return ret;
    }

    private static final ObjectMapper jsonMapper = new ObjectMapper();

    /**
     * Serialize object to JSON string.
     *
     * @param obj
     * @return JSON string.
     */
    public static String toJson(Object obj) throws JsonProcessingException {
        return jsonMapper.writeValueAsString(obj);
    }

    /**
     * Parse JSON string to object.
     *
     * @param text  JSON string.
     * @param clazz Class of target object.
     * @param <T>
     * @return Parsed object.
     */
    public static <T> T fromJson(String text, Class<T> clazz) throws JsonProcessingException {
        return jsonMapper.readValue(text, clazz);
    }

    public static Object numericCast(Object input, Class<?> clazz) {
        if (clazz == boolean.class || clazz == Boolean.class) {
            if (input.getClass() != Boolean.class) {
                throw new UnsupportedOperationException(
                    String.format("Casting %s to %s.", input.getClass().getCanonicalName(), clazz.getCanonicalName()));
            }
            return input;
        }
        if (!Number.class.isAssignableFrom(input.getClass())) {
            throw new UnsupportedOperationException(
                String.format("Input should be boolean or numeric types, %s is not supported",
                    input.getClass().getCanonicalName()));
        }
        Number num = (Number)input;
        if (clazz == byte.class || clazz == Byte.class) {
            return num.byteValue();
        }
        if (clazz == short.class || clazz == Short.class) {
            return num.shortValue();
        }
        if (clazz == int.class || clazz == Integer.class) {
            return num.intValue();
        }
        if (clazz == long.class || clazz == Long.class) {
            return num.longValue();
        }
        if (clazz == float.class || clazz == Float.class) {
            return num.floatValue();
        }
        if (clazz == double.class || clazz == Double.class) {
            return num.doubleValue();
        }
        throw new UnsupportedOperationException(
            String.format("Unsupported target type: %s.", clazz.getCanonicalName()));
    }
}
