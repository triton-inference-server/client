package com.nvidia.triton.contrib.pojo;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * JSON object for
 * <a href="https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/rest_predict_v2.yaml#L320">inference_response</a>
 * object in kfserving's v2 rest schema.
 */
@JsonInclude(Include.NON_NULL)
public class InferenceResponse {

    @JsonProperty("model_name")
    private String modelName;
    @JsonProperty("model_version")
    private String modelVersion;
    private String id;
    private Parameters parameters;
    private List<IOTensor> outputs;

    public InferenceResponse() {
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public void setId(String id) {
        this.id = id;
    }

    public void setParameters(Parameters parameters) {
        this.parameters = parameters;
    }

    public void setOutputs(List<IOTensor> outputs) {
        this.outputs = outputs;
    }

    public String getModelName() {
        return modelName;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public String getId() {
        return id;
    }

    public Parameters getParameters() {
        return parameters;
    }

    public List<IOTensor> getOutputs() {
        return outputs;
    }

    public IOTensor getOutputByName(String name) {
        for (IOTensor output : this.outputs) {
            if (output.getName().equals(name)) {
                return output;
            }
        }
        return null;
    }
}
