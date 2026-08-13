"""Small stateful model used to observe worker reuse and reloads."""

import os
import uuid


class PersistentFake:
    """Echo requests while exposing state that changes only on model load."""

    def load_model(self, model=None, **kwargs):
        self.load_id = uuid.uuid4().hex
        self.embed_count = 0
        self.model_name = model
        self.load_kwargs = kwargs

    def embed(self, input, **kwargs):
        self.embed_count += 1
        return {
            "output": {
                "load_id": self.load_id,
                "pid": os.getpid(),
                "embed_count": self.embed_count,
                "model": self.model_name,
                "load_kwargs": self.load_kwargs,
                "input": input,
                "kwargs": kwargs,
            }
        }
