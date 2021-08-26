package com.nvidia.triton.contrib.endpoint;

import java.util.Objects;

import com.nvidia.triton.contrib.InferenceServerClient;
import com.nvidia.triton.contrib.Util;
import com.google.common.base.Preconditions;

/**
 * Endpoint is an abstraction that allow different kinds of strategy to provide ip and port for
 * {@link InferenceServerClient} to send requests.
 */
public abstract class AbstractEndpoint {

    private static final int RETRY_COUNT = 10;
    private String lastResult = "";

    abstract String getEndpointImpl() throws Exception;

    abstract int getEndpointNum() throws Exception;

    /**
     * Get string in ip:port[/path] format.
     *
     * @return
     * @throws Exception
     */
    public String getEndpoint() throws Exception {
        for (int i = 0; i < RETRY_COUNT; i++) {
            String url = this.getEndpointImpl();
            Preconditions.checkState(!Util.isEmpty(url), "getEndpointImpl should not return null or empty string!");
            if (!Objects.equals(this.lastResult, url) || this.getEndpointNum() < 2) {
                this.lastResult = url;
                return url;
            }
        }
        throw new RuntimeException(String.format("Failed to get endpoint address after trying %d times.", RETRY_COUNT));
    }
}
