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

/**
 * @author xiafei.qiuxf
 * @date 2021/4/13
 */

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import triton.client.endpoint.AbstractEndpoint;
import triton.client.endpoint.FixedEndpoint;
import triton.client.pojo.IOTensor;
import triton.client.pojo.Parameters;
import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.http.HttpResponse;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.impl.nio.client.CloseableHttpAsyncClient;
import org.apache.http.impl.nio.client.HttpAsyncClients;
import org.apache.http.impl.nio.conn.PoolingNHttpClientConnectionManager;
import org.apache.http.impl.nio.reactor.DefaultConnectingIOReactor;
import org.apache.http.impl.nio.reactor.IOReactorConfig;
import org.apache.http.nio.entity.NByteArrayEntity;
import org.apache.http.nio.reactor.ConnectingIOReactor;

/**
 * An InferenceServerClient object is used to perform any kind of communication with the InferenceServer using http
 * protocol. None of the methods are thread safe. The object is intended to be used by a single thread and
 * simultaneously calling different methods with different threads is not supported and will cause undefined behavior.
 */
public class InferenceServerClient implements AutoCloseable {

    private static final Log LOG = LogFactory.getLog(InferenceServerClient.class);

    private static class HttpConfig {
        private int ioThreadNum;
        private int readTimeout;
        private int connectTimeout;
        private int requestTimeout;
        private int maxConnectionCount;
        private int maxConnectionPerRoute;
        private boolean keepAlive;

        public HttpConfig() {
            this.ioThreadNum = 10;
            this.readTimeout = 5000;
            this.connectTimeout = 5000;
            this.maxConnectionCount = 1000;
            this.maxConnectionPerRoute = 1000;
            this.requestTimeout = 0;
            this.keepAlive = true;
        }

        public HttpConfig(int ioThreadNum, int readTimeout, int connectTimeout,
                          int maxConnectionCount, int maxConnectionPerRoute) {
            super();
            this.ioThreadNum = ioThreadNum;
            this.readTimeout = readTimeout;
            this.connectTimeout = connectTimeout;
            this.maxConnectionCount = maxConnectionCount;
            this.maxConnectionPerRoute = maxConnectionPerRoute;
            this.keepAlive = true;
        }

        public HttpConfig(int ioThreadNum, int readTimeout, int connectTimeout,
                          int maxConnectionCount, int maxConnectionPerRoute, int requestTimeout) {
            this(ioThreadNum, readTimeout, connectTimeout, maxConnectionCount, maxConnectionPerRoute);
            this.requestTimeout = requestTimeout;
        }

        public int getIoThreadNum() {
            return ioThreadNum;
        }

        public void setIoThreadNum(int ioThreadNum) {
            this.ioThreadNum = ioThreadNum;
        }

        public int getReadTimeout() {
            return readTimeout;
        }

        public void setReadTimeout(int readTimeout) {
            this.readTimeout = readTimeout;
        }

        public int getConnectTimeout() {
            return connectTimeout;
        }

        public void setConnectTimeout(int connectTimeout) {
            this.connectTimeout = connectTimeout;
        }

        public int getMaxConnectionCount() {
            return maxConnectionCount;
        }

        public void setMaxConnectionCount(int maxConnectionCount) {
            this.maxConnectionCount = maxConnectionCount;
        }

        public int getMaxConnectionPerRoute() {
            return maxConnectionPerRoute;
        }

        public void setMaxConnectionPerRoute(int maxConnectionPerRoute) {
            this.maxConnectionPerRoute = maxConnectionPerRoute;
        }

        public int getRequestTimeout() { return requestTimeout; }

        public void setRequestTimeout(int requestTimeout) {
            this.requestTimeout = requestTimeout;
        }

        public boolean isKeepAlive() {
            return keepAlive;
        }

        public void setKeepAlive(boolean keepAlive) {
            this.keepAlive = keepAlive;
        }
    }

    /**
     * Inference server url.
     */
    private final AbstractEndpoint endpoint;

    private final int requestTimeout;

    private final CloseableHttpAsyncClient httpClient;

    private int retryCnt = 3;

    /**
     * Create an InferenceServerClient object.
     *
     * @param endpoint          The endpoint to connect with.
     * @param connectionTimeout The timeout value for the connection. Default value is 5 sec.
     * @param networkTimeout    The timeout value for the network. Default value is 5 sec.
     */
    public InferenceServerClient(AbstractEndpoint endpoint, int connectionTimeout, int networkTimeout)
        throws IOException {
        this.endpoint = endpoint;
        this.requestTimeout = networkTimeout;
        HttpConfig httpConfig = new HttpConfig();
        httpConfig.setConnectTimeout(connectionTimeout);
        httpConfig.setReadTimeout(networkTimeout);
        this.httpClient = createHttpClient(httpConfig);
    }

    public InferenceServerClient(String ipPort, int connectionTimeout, int networkTimeout) throws IOException {
        this(new FixedEndpoint(ipPort), connectionTimeout, networkTimeout);
    }

    public InferenceServerClient(AbstractEndpoint endpoint, HttpConfig httpConfig) throws IOException {
        this.endpoint = endpoint;
        this.requestTimeout = httpConfig.getRequestTimeout();
        this.httpClient = createHttpClient(httpConfig);
    }

    private static CloseableHttpAsyncClient createHttpClient(HttpConfig httpConfig) throws IOException {
        ConnectingIOReactor ioReactor = new DefaultConnectingIOReactor();
        PoolingNHttpClientConnectionManager cm = new PoolingNHttpClientConnectionManager(
            ioReactor);
        cm.setMaxTotal(httpConfig.getMaxConnectionCount());
        cm.setDefaultMaxPerRoute(httpConfig.getMaxConnectionPerRoute());
        IOReactorConfig config = IOReactorConfig.custom()
            .setTcpNoDelay(true)
            .setSoTimeout(httpConfig.getReadTimeout())
            .setSoReuseAddress(true)
            .setConnectTimeout(httpConfig.getConnectTimeout())
            .setIoThreadCount(httpConfig.getIoThreadNum())
            .setSoKeepAlive(httpConfig.isKeepAlive()).build();
        final RequestConfig requestConfig = RequestConfig.custom()
            .setConnectTimeout(httpConfig.getConnectTimeout())
            .setSocketTimeout(httpConfig.getReadTimeout()).build();
        CloseableHttpAsyncClient httpClient = HttpAsyncClients.custom().setConnectionManager(cm)
            .setDefaultIOReactorConfig(config)
            .setDefaultRequestConfig(requestConfig).build();
        httpClient.start();
        return httpClient;
    }

    public void setRetryCnt(int retryCnt) {
        Preconditions.checkArgument(retryCnt > 0, "Invalid retryCount: %s", retryCnt);
        this.retryCnt = retryCnt;
    }

    public InferResult infer(InferArguments arg)
        throws InferenceException {
        Preconditions.checkArgument(arg != null, "arguments should not be null.");
        Map<String, Object> inferReq = new HashMap<>();
        Parameters param = new Parameters();
        if (!Util.isEmpty(arg.requestId)) {
            inferReq.put("id", arg.requestId);
        }
        if (arg.sequenceId != 0) {
            param.put("sequence_id", arg.sequenceId);
            param.put("sequence_start", arg.sequenceStart);
            param.put("sequence_end", arg.sequenceEnd);
        }
        if (arg.priority != 0) {
            param.put("priority", arg.priority);
        }
        if (arg.timeout > 0) {
            param.put("timeout", arg.timeout);
        }
        List<IOTensor> inputs = Lists.transform(arg.inputs, InferInput::getTensor);
        inferReq.put("inputs", inputs);
        if (!Util.isEmpty(arg.outputs)) {
            List<IOTensor> outputs = Lists.transform(arg.outputs, InferRequestedOutput::getTensor);
            inferReq.put("outputs", outputs);
        } else {
            // Looks strange, just do as in Triton's python SDK.
            param.put("binary_data_output", true);
        }
        if (!param.isEmpty()) {
            inferReq.put("parameters", param);
        }

        HttpPost post;
        try {
            post = createHttpPost(arg, inferReq);
        } catch (Exception e) {
            throw new InferenceException(e);
        }

        for (int i = 0; i <= this.retryCnt; i++) {
            try {
                Future<HttpResponse> future = this.httpClient.execute(post, null);
                HttpResponse resp = this.requestTimeout > 0
                    ? future.get(this.requestTimeout, TimeUnit.MILLISECONDS) : future.get();
                return new InferResult(resp);
            } catch (Exception e) {
                if (i == this.retryCnt) {
                    LOG.error(String.format("Inference failed on %dth retry, url: %s", i, post.getURI().toString()), e);
                    throw new InferenceException(e);
                } else if (LOG.isDebugEnabled()) {
                    LOG.debug(String.format("Inference failed on %dth retry, url: %s", i, post.getURI().toString()), e);
                }
            }
        }
        // Should never reach here.
        return null;
    }

    private HttpPost createHttpPost(InferArguments arg, Map<String, Object> inferReq)
        throws Exception {
        // Create post body in binary format:
        //    <json body><optional_binary_tensor>...
        ByteArrayOutputStream bodyBytes = new ByteArrayOutputStream();
        byte[] jsonBytes = Util.toJson(inferReq).getBytes(StandardCharsets.UTF_8);
        boolean hasBinaryInput = false;
        bodyBytes.write(jsonBytes);
        for (InferInput input : arg.inputs) {
            byte[] binInput = input.getBinaryData();
            if (binInput != null) {
                bodyBytes.write(binInput);
                hasBinaryInput = true;
            }
        }
        if (hasBinaryInput) {
            arg.headers.put("Inference-Header-Content-Length", String.valueOf(jsonBytes.length));
        }

        // Create target URI.
        URIBuilder ub = new URIBuilder(this.getUrl());
        String safeModelName = URLEncoder.encode(arg.modelName, Charsets.UTF_8.toString());
        String requestUri = Util.isEmpty(arg.modelVersion) ?
            String.format("/v2/models/%s/infer", safeModelName) :
            String.format("/v2/models/%s/versions/%s/infer", safeModelName, arg.modelVersion);
        ub.setPath(ub.getPath() + requestUri);
        arg.queryParams.forEach(ub::addParameter);

        // Crete HttpPost, uri, body and headers.
        HttpPost post = new HttpPost(ub.build());
        arg.headers.forEach(post::setHeader);
        post.setEntity(new NByteArrayEntity(bodyBytes.toByteArray()));
        return post;
    }

    private String getUrl() throws Exception {
        return "http://" + this.endpoint.getEndpoint();
    }

    public InferResult infer(String modelName, List<InferInput> inputs, List<InferRequestedOutput> outputs)
        throws InferenceException {
        return this.infer(new InferArguments(modelName, inputs, outputs));
    }

    @Override
    public void close() throws Exception {
        this.httpClient.close();
    }

    public static class InferArguments {
        final String modelName;
        final List<InferInput> inputs;
        final List<InferRequestedOutput> outputs;
        String modelVersion = "";
        String requestId = "";
        int sequenceId = 0;
        boolean sequenceStart = false;
        boolean sequenceEnd = false;
        int priority = 0;
        int timeout = -1;
        Map<String, String> headers = new HashMap<>();
        Map<String, String> queryParams = new HashMap<>();

        public InferArguments(String modelName, List<InferInput> inputs, List<InferRequestedOutput> outputs) {
            this.modelName = modelName;
            this.inputs = inputs;
            this.outputs = outputs;
        }

        public InferArguments(String modelName, List<InferInput> inputs) {
            this.modelName = modelName;
            this.inputs = inputs;
            this.outputs = null;
        }

        public InferArguments(String modelName, InferInput... inputs) {
            this.modelName = modelName;
            this.inputs = new ArrayList<>(Arrays.asList(inputs));
            this.outputs = null;
        }

        public InferArguments setModelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
            return this;
        }

        public InferArguments setRequestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public InferArguments setSequenceId(int sequenceId) {
            this.sequenceId = sequenceId;
            return this;
        }

        public InferArguments setSequenceStart(boolean sequenceStart) {
            this.sequenceStart = sequenceStart;
            return this;
        }

        public InferArguments setSequenceEnd(boolean sequenceEnd) {
            this.sequenceEnd = sequenceEnd;
            return this;
        }

        public InferArguments setPriority(int priority) {
            this.priority = priority;
            return this;
        }

        public InferArguments setTimeout(int timeout) {
            this.timeout = timeout;
            return this;
        }

        public InferArguments setHeader(String key, String value) {
            this.headers.put(key, value);
            return this;
        }

        public InferArguments addQueryParam(String key, String value) {
            this.queryParams.put(key, value);
            return this;
        }
    }
}
