<!--
Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Input Data

Use the [`--help`](cli.md#--help) option to see complete documentation for all
input data options. By default Perf Analyzer sends random data to all the inputs
of your model. You can select a different input data mode with the
[`--input-data`](cli.md#--input-datazerorandompath) option:

- _random_: (default) Send random data for each input. Note: Perf Analyzer only
  generates random data once per input and reuses that for all inferences
- _zero_: Send zeros for each input.
- directory path: A path to a directory containing a binary file for each input,
  named the same as the input. Each binary file must contain the data required
  for that input for a batch-1 request. Each file should contain the raw binary
  representation of the input in row-major order.
- file path: A path to a JSON file containing data to be used with every
  inference request. See the "Real Input Data" section for further details.
  [`--input-data`](cli.md#--input-datazerorandompath) can be provided multiple
  times with different file paths to specific multiple JSON files.

For tensors with `STRING`/`BYTES` datatype, the
[`--string-length`](cli.md#--string-lengthn) and
[`--string-data`](cli.md#--string-datastring) options may be used in some cases
(see [`--help`](cli.md#--help) for full documentation).

For models that support batching you can use the [`-b`](cli.md#-b-n) option to
indicate the batch size of the requests that Perf Analyzer should send. For
models with variable-sized inputs you must provide the
[`--shape`](cli.md#--shapestring) argument so that Perf Analyzer knows what
shape tensors to use. For example, for a model that has an input called
`IMAGE` that has shape `[3, N, M]`, where `N` and `M` are variable-size
dimensions, to tell Perf Analyzer to send batch size 4 requests of shape
`[3, 224, 224]`:

```
$ perf_analyzer -m mymodel -b 4 --shape IMAGE:3,224,224
```

## Real Input Data

The performance of some models is highly dependent on the data used. For such
cases you can provide data to be used with every inference request made by Perf
Analyzer in a JSON file. Perf Analyzer will use the provided data in a
round-robin order when sending inference requests. For sequence models, if a
sequence length is specified via
[`--sequence-length`](cli.md#--sequence-lengthn), Perf Analyzer will also loop
through the provided data in a round-robin order up to the specified sequence
length (with a percentage variation customizable via
[`--sequence-length-variation`](cli.md#--sequence-length-variationn)).
Otherwise, the sequence length will be the number of inputs specified in
user-provided input data.

Each entry in the `"data"` array must specify all input tensors with the exact
size expected by the model for a single batch. The following example describes
data for a model with inputs named, `INPUT0` and `INPUT1`, shape `[4, 4]` and
data type `INT32`:

```json
{
  "data":
    [
      {
        "INPUT0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "INPUT1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      },
      {
        "INPUT0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "INPUT1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      },
      {
        "INPUT0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "INPUT1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      },
      {
        "INPUT0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "INPUT1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      }
    ]
}
```

Note that the `[4, 4]` tensor has been flattened in a row-major format for the
inputs. In addition to specifying explicit tensors, you can also provide Base64
encoded binary data for the tensors. Each data object must list its data in a
row-major order. Binary data must be in little-endian byte order. The following
example highlights how this can be achieved:

```json
{
  "data":
    [
      {
        "INPUT0": {"b64": "YmFzZTY0IGRlY29kZXI="},
        "INPUT1": {"b64": "YmFzZTY0IGRlY29kZXI="}
      },
      {
        "INPUT0": {"b64": "YmFzZTY0IGRlY29kZXI="},
        "INPUT1": {"b64": "YmFzZTY0IGRlY29kZXI="}
      },
      {
        "INPUT0": {"b64": "YmFzZTY0IGRlY29kZXI="},
        "INPUT1": {"b64": "YmFzZTY0IGRlY29kZXI="}
      }
    ]
}
```

In case of sequence models, multiple data streams can be specified in the JSON
file. Each sequence will get a data stream of its own and Perf Analyzer will
ensure the data from each stream is played back to the same correlation ID. The
below example highlights how to specify data for multiple streams for a sequence
model with a single input named `INPUT`, shape `[1]` and data type `STRING`:

```json
{
  "data":
    [
      [
        {
          "INPUT": ["1"]
        },
        {
          "INPUT": ["2"]
        },
        {
          "INPUT": ["3"]
        },
        {
          "INPUT": ["4"]
        }
      ],
      [
        {
          "INPUT": ["1"]
        },
        {
          "INPUT": ["1"]
        },
        {
          "INPUT": ["1"]
        }
      ],
      [
        {
          "INPUT": ["1"]
        },
        {
          "INPUT": ["1"]
        }
      ]
    ]
}
```

The above example describes three data streams with lengths 4, 3 and 2
respectively. Perf Analyzer will hence produce sequences of length 4, 3 and 2 in
this case.

You can also provide an optional `"shape"` field to the tensors. This is
especially useful while profiling the models with variable-sized tensors as
input. Additionally note that when providing the `"shape"` field, tensor
contents must be provided separately in a "content" field in row-major order.
The specified shape values will override default input shapes provided as a
command line option (see [`--shape`](cli.md#--shapestring)) for variable-sized
inputs. In the absence of a `"shape"` field, the provided defaults will be used.
There is no need to specify shape as a command line option if all the input data
provide shape values for variable tensors. Below is an example JSON file for a
model with a single input `INPUT`, shape `[-1, -1]` and data type `INT32`:

```json
{
  "data":
    [
      {
        "INPUT":
          {
              "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "shape": [2,8]
          }
      },
      {
        "INPUT":
          {
              "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "shape": [8,2]
          }
      },
      {
        "INPUT":
          {
              "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          }
      },
      {
        "INPUT":
          {
              "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "shape": [4,4]
          }
      }
    ]
}
```

The following is the example to provide contents as base64 string with explicit
shapes:

```json
{
  "data":
    [
      {
        "INPUT":
          {
            "content": {"b64": "/9j/4AAQSkZ(...)"},
            "shape": [7964]
          }
      },
      {
        "INPUT":
          {
            "content": {"b64": "/9j/4AAQSkZ(...)"},
            "shape": [7964]
          }
      }
    ]
}
```

Note that for `STRING` type, an element is represented by a 4-byte unsigned
integer giving the length followed by the actual bytes. The byte array to be
encoded using base64 must include the 4-byte unsigned integers.

### Output Validation

When real input data is provided, it is optional to request Perf Analyzer to
validate the inference output for the input data.

Validation output can be specified in the `"validation_data"` field have the
same format as the `"data"` field for real input. Note that the entries in
`"validation_data"` must align with `"data"` for proper mapping. The following
example describes validation data for a model with inputs named `INPUT0` and
`INPUT1`, outputs named `OUTPUT0` and `OUTPUT1`, all tensors have shape `[4, 4]`
and data type `INT32`:

```json
{
  "data":
    [
      {
        "INPUT0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "INPUT1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      }
    ],
  "validation_data":
    [
      {
        "OUTPUT0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "OUTPUT1": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
      }
    ]
}
```

Besides the above example, the validation outputs can be specified in the same
variations described in the real input data section.

# Shared Memory

By default Perf Analyzer sends input tensor data and receives output tensor data
over the network. You can instead instruct Perf Analyzer to use system shared
memory or CUDA shared memory to communicate tensor data. By using these options
you can model the performance that you can achieve by using shared memory in
your application. Use
[`--shared-memory=system`](cli.md#--shared-memorynonesystemcuda) to use system
(CPU) shared memory or
[`--shared-memory=cuda`](cli.md#--shared-memorynonesystemcuda) to use CUDA
shared memory.
