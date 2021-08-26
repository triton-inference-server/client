package com.nvidia.triton.contrib.pojo;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;

/**
 * This class represent
 * <a href="https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/rest_predict_v2.yaml#L258">request_input</a>
 * and
 * <a href="https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/rest_predict_v2.yaml#L299">response_output</a>
 * in kfserving's v2 rest schema. The two objects share the same definitions.
 */

@JsonInclude(Include.NON_NULL)
public class IOTensor {
    private String name;
    private long[] shape;
    private DataType datatype;
    private Parameters parameters;
    private Object[] data;

    public IOTensor() {
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public long[] getShape() {
        return shape;
    }

    public void setShape(long[] shape) {
        this.shape = shape;
    }

    public DataType getDatatype() {
        return datatype;
    }

    public void setDatatype(DataType datatype) {
        this.datatype = datatype;
    }

    public Parameters getParameters() {
        return parameters;
    }

    public void setParameters(Parameters parameters) {
        this.parameters = parameters;
    }

    public Object[] getData() {
        return data;
    }

    public void setData(Object[] data) {
        this.data = data;
    }
}
