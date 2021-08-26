package com.nvidia.triton.contrib.pojo;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.nvidia.triton.contrib.pojo.Parameters.ParamDeserializer;
import com.nvidia.triton.contrib.pojo.Parameters.ParamSerializer;

/**
 * This class represent
 * <a href="https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/rest_predict_v2.yaml#L254">parameters</a>
 * object in kfserving's v2 rest schema is just a JSON string, and offer some util methods.
 * When serializing/deserializing, Parameters should act just like a map.
 */
@JsonSerialize(using = ParamSerializer.class)
@JsonDeserialize(using = ParamDeserializer.class)
public class Parameters {

    public final static String KEY_BINARY_DATA_SIZE = "binary_data_size";

    private Map<String, Object> params;

    public Parameters() {
        this.params = new HashMap<>();
    }

    public Parameters(Map<String, Object> params) {
        this.params = params;
    }

    /**
     * Add or over-write parameter key-values.
     *
     * @param key   name of new parameter.
     * @param value value of new parameter.
     * @return The original value if key exists.
     */
    public Object put(String key, Object value) {
        return this.params.put(key, value);
    }

    /**
     * Remove key from parameters.
     *
     * @param key name of new parameter.
     * @return The original value  if key exists.
     */
    public Object remove(String key) {
        return this.params.remove(key);
    }

    /**
     * Check if parameters are empty.
     *
     * @return true if empty.
     */
    public boolean isEmpty() {
        return this.params.isEmpty();
    }

    /**
     * Get an parameter value as bool. Some conversions are done under the hood.
     *
     * @param name name of the parameter.
     * @return null if named parameter not exists, otherwise the parameter value in bool format.
     * @throws IllegalArgumentException if conversions failed.
     */
    public Boolean getBool(final String name) {
        Object o = this.params.get(name);
        if (o == null) {
            return null;
        } else if (o instanceof Boolean) {
            return (Boolean)o;
        } else if (o instanceof String) {
            return Boolean.valueOf(((String)o));
        } else {
            throw new IllegalArgumentException(
                String.format("Could not convert type %s to Boolean", o.getClass().getCanonicalName()));
        }
    }

    /**
     * Get an parameter value as integer. Some conversions are done under the hood.
     *
     * @param name name of the parameter.
     * @return null if named parameter not exists, otherwise the parameter value in integer format.
     * @throws IllegalArgumentException if conversions failed.
     */
    public Integer getInt(final String name) {
        Object o = this.params.get(name);
        if (o == null) {
            return null;
        } else if (o instanceof Number) {
            return ((Number)o).intValue();
        } else if (o instanceof String) {
            return Integer.valueOf(((String)o));
        } else {
            throw new IllegalArgumentException(
                String.format("Could not convert type %s to Integer", o.getClass().getCanonicalName()));
        }
    }

    /**
     * Get an parameter value as float. Some conversions are done under the hood.
     *
     * @param name name of the parameter.
     * @return null if named parameter not exists, otherwise the parameter value in float format.
     * @throws IllegalArgumentException if conversions failed.
     */
    public Float getFloat(final String name) {
        Object o = this.params.get(name);
        if (o == null) {
            return null;
        } else if (o instanceof Number) {
            return ((Number)o).floatValue();
        } else if (o instanceof String) {
            return Float.valueOf(((String)o));
        } else {
            throw new IllegalArgumentException(
                String.format("Could not convert type %s to Float", o.getClass().getCanonicalName()));
        }
    }

    /**
     * Get an parameter value as double. Some conversions are done under the hood.
     *
     * @param name name of the parameter.
     * @return null if named parameter not exists, otherwise the parameter value in double format.
     * @throws IllegalArgumentException if conversions failed.
     */
    public Double getDouble(final String name) {
        Object o = this.params.get(name);
        if (o == null) {
            return null;
        } else if (o instanceof Number) {
            return ((Number)o).doubleValue();
        } else if (o instanceof String) {
            return Double.valueOf(((String)o));
        } else {
            throw new IllegalArgumentException(
                String.format("Could not convert type %s to Double", o.getClass().getCanonicalName()));
        }
    }

    /**
     * Get an parameter value as string.
     *
     * @param name name of the parameter.
     * @return null if named parameter not exists, otherwise the parameter value in string format.
     */
    public String getString(final String name) {
        Object o = this.params.get(name);
        if (o == null) {
            return null;
        } else {
            return String.valueOf(o);
        }
    }

    static class ParamSerializer extends JsonSerializer<Parameters> {

        @Override
        public void serialize(Parameters parameters, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
            throws IOException {
            jsonGenerator.writeObject(parameters.params);
        }
    }

    static class ParamDeserializer extends JsonDeserializer<Parameters> {

        @Override
        public Parameters deserialize(JsonParser p, DeserializationContext ctx)
            throws IOException {
            ObjectMapper mapper = new ObjectMapper();
            final Map<String, Object> obj = mapper.readValue(p,
                new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {});
            return new Parameters(obj);
        }
    }

}
