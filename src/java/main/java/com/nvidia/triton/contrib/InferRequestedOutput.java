package com.nvidia.triton.contrib;

import com.nvidia.triton.contrib.pojo.Parameters;
import com.nvidia.triton.contrib.pojo.IOTensor;

/**
 * @author xiafei.qiuxf
 * @date 2021/4/13
 */
public class InferRequestedOutput {
    /**
     * The name of output tensor to associate with this object.
     */
    private final String name;
    /**
     * Indicates whether to return result data for the output in
     * binary format or explicit tensor within JSON. The default
     * value is True, which means the data will be delivered as
     * binary data in the HTTP body after JSON object. This field
     * will be unset if shared memory is set for the output.
     */
    private final boolean isBinary;

    /**
     * The number of classifications to be requested. The default
     * value is 0 which means the classification results are not
     * requested.
     */
    private final int classCount;

    private final Parameters parameters = new Parameters();

    public InferRequestedOutput(String name, boolean isBinary, int classCount) {
        this.name = name;
        this.isBinary = isBinary;
        this.classCount = classCount;
        if (this.classCount != 0) {
            this.parameters.put("classification", classCount);
        }
        this.parameters.put("binary_data", this.isBinary);

    }

    public InferRequestedOutput(String name) {
        this(name, true, 0);
    }

    public InferRequestedOutput(String name, boolean isBinary) {
        this(name, isBinary, 0);
    }

    public IOTensor getTensor() {
        IOTensor tensor = new IOTensor();
        tensor.setName(this.name);
        tensor.setParameters(this.parameters);
        return tensor;
    }
}
