package com.nvidia.triton.contrib;

import com.nvidia.triton.contrib.pojo.ResponseError;

/**
 * Universal exceptions of triton client.
 */
public class InferenceException extends Exception {
    public InferenceException(ResponseError err) {
        super(err.getError());
    }

    public InferenceException(String message) {
        super(message);
    }

    public InferenceException(Throwable cause) {
        super(cause);
    }
}
