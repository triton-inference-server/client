
Just for test install PyTriton client:

```bash
pip install nvidia-pytriton
```

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

```python
import numpy as np
from tritonclient._client import Client

Client("localhost:8000").model("identity").infer(inputs={"input": np.char.encode([["a"]], "utf-8")}
```

Expected output:

```python
{'output': array(['a'], dtype='<U1')}
```

