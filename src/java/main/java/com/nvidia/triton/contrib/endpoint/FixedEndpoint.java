package com.nvidia.triton.contrib.endpoint;

import com.nvidia.triton.contrib.Util;
import com.google.common.base.Preconditions;

/**
 * Endpoint that connect to single address.
 */
public class FixedEndpoint extends AbstractEndpoint {

    private final String addr;

    /**
     * Create a endpoint connecting to a fixed address.
     *
     * @param endpoint Endpoint in host:port[/path] format without schema part.
     */
    public FixedEndpoint(String endpoint) {
        Preconditions.checkArgument(!Util.isEmpty(endpoint), "endpoint should not be null or empty.");
        Preconditions.checkArgument(!endpoint.contains("://"),
            "endpoint should be in host:port[/path] format without scheme.");
        this.addr = endpoint;
    }

    @Override
    String getEndpointImpl() {
        return this.addr;
    }

    @Override
    int getEndpointNum() {
        return 1;
    }
}
