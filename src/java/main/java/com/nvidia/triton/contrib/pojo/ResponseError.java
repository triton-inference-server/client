package com.nvidia.triton.contrib.pojo;

/**
 * This is a JSON object representing error response body of triton server.
 */
public class ResponseError {
    private String error;

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }
}
