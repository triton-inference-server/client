
## Dependencies

Just for test install PyTriton client:

```bash
pip install nvidia-pytriton
```

## Non-decoupled PyTriton client

It is possible to test new client using PyTriton server:

```python
import time
import numpy as np
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from pytriton.decorators import batch

@batch
def identity(input):
    return {"output": input}


triton = Triton()
triton.bind(
    model_name="identity",
    infer_func=identity,
    inputs=[Tensor(name="input", dtype=np.bytes_, shape=(1,))],
    outputs=[Tensor(name="output", dtype=np.bytes_, shape=(1,))],
    strict=False,
)
triton.run()
```


You can test new client with simple request:

<!--pytest-codeblocks:cont-->
```python
import numpy as np
from tritonclient._client import Client

client = Client("localhost:8000").model("identity")

result = client.infer(inputs={"input": np.char.encode([["a"]], "utf-8")})
```

<!--pytest-codeblocks:cont-->
<!--
```python
client.close()
triton.stop()

# Sleep for a while to let the server run
import time
time.sleep(40)

assert "output" in result
```
-->


Expected output:

<!--pytest.mark.skip-->
```python
{'output': array(['a'], dtype='<U1')}
```

## Decoupled PyTriton client

```python
from pytriton.decorators import batch
import time
import numpy as np

# Decorate your model function with `@batch`. This allows Triton to batch multiple requests together.
@batch
def _infer_fn(input):
    for _ in range(3):
        time.sleep(2.0)
        yield {"output": input}

# Create a Triton model configuration and bind it to the model function `_infer_fn`.
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
```


Bind Triton:

<!--pytest-codeblocks:cont-->
```python
triton = Triton()
triton.bind(
    model_name="decoupled_identity",
    infer_func=_infer_fn,
    inputs=[
        Tensor(name="input", dtype=np.int32, shape=(-1,)),
        # Shape with a batch dimension (-1) to support variable-sized batches.
    ],
    outputs=[
        Tensor(name="output", dtype=np.int32, shape=(-1,)),
        # Output shape with a batch dimension (-1).
    ],
    config=ModelConfig(decoupled=True),
)
```

Start Triton:

<!--pytest-codeblocks:cont-->
```python
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Let the server run for a while
import time
time.sleep(40)
```
-->

User client for itegration over decoupled results:

<!--pytest-codeblocks:cont-->
```python
import numpy as np
from tritonclient._client import Client

client = Client("localhost:8000").model("decoupled_identity")

results = []

# Test fails with 500 error

for result in client.infer(inputs={"input": np.array([1], dtype=np.int32)}):
    print(result)
    results.append(result)
```

<!--pytest-codeblocks:cont-->
<!--
```python
client.close()
triton.stop()

assert "output" in results[0]
```
-->

Expected output:

<!--pytest.mark.skip-->
```python
{'output': array([0.001])}
{'output': array([0.001])}
{'output': array([0.001])}
```