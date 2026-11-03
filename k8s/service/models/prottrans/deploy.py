"""Ray job: start ProtT5-XL-UniRef50 as a named, detached actor on the ProtTrans
worker, next to H-Optimus-0 (../hoptimus0/deploy.py).

Same checkpoint and preprocessing as vhmodels/models/ProtTrans/model.py
(variant prot_t5_xl_uniref50). Only the encoder is loaded, and each
protein gets one mean-pooled 1024-dim embedding instead of per-residue ones, to
keep the API responses small.
"""

import re

import ray

MODEL = "virtual-human-chc/prot_t5_xl_uniref50"
REVISION = "167966d0626d11110e00a3e26c1cd6c171b42e12"
# Per request, so one large batch cannot push the actor out of its memory.
MAX_RESIDUES = 4000

RUNTIME_ENV = {
    "pip": {
        "packages": [
            "torch==2.9.1+cpu",
            "transformers==4.57.1",
            "sentencepiece==0.2.1",
            "protobuf==6.33.0",
            "hf_xet==1.2.0",
        ],
        "pip_install_options": ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
    }
}


@ray.remote
class ProtT5:
    def __init__(self):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import T5EncoderModel, T5Tokenizer

        self.torch = torch
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL, revision=REVISION, do_lower_case=False)
        # The checkpoint is the full fp32 T5; T5EncoderModel skips the decoder.
        self.model = T5EncoderModel.from_pretrained(MODEL, revision=REVISION).eval()
        # The safetensors file stores the tied token embedding only as
        # decoder.embed_tokens.weight ("shared.weight" is just an alias in its
        # metadata), so from_pretrained leaves the encoder's embedding randomly
        # initialized. Copy it in; encoder.embed_tokens is the same module.
        checkpoint = hf_hub_download(MODEL, "model.safetensors", revision=REVISION)
        with safe_open(checkpoint, framework="pt") as f:
            shared = f.get_tensor("decoder.embed_tokens.weight")
        with torch.no_grad():
            self.model.shared.weight.copy_(shared)
        assert self.model.encoder.embed_tokens.weight is self.model.shared.weight

    def warm_up(self):
        """Embed one short sequence and return the embedding size."""
        return self.embed(["MKTAYIAKQR"])["dim"]

    def embed(self, sequences):
        """Embed protein sequences given as one-letter amino acid codes."""
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("expected a non-empty list of protein sequences")
        if not all(isinstance(s, str) and s for s in sequences):
            raise ValueError("every sequence must be a non-empty string")
        residues = sum(map(len, sequences))
        if residues > MAX_RESIDUES:
            raise ValueError(f"{residues} residues in one request, at most {MAX_RESIDUES}")

        spaced = [" ".join(re.sub(r"[UZOB]", "X", s)) for s in sequences]
        batch = self.tokenizer(spaced, add_special_tokens=True, padding="longest", return_tensors="pt")
        with self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        embeddings = []
        for states, mask in zip(hidden, batch["attention_mask"]):
            length = int(mask.sum()) - 1  # drop the trailing </s>, as vhmodels does
            embeddings.append(states[:length].mean(dim=0).tolist())
        return {"dim": len(embeddings[0]), "embeddings": embeddings}


ray.init(namespace="vhmodels")
actor = ProtT5.options(
    name="prottrans",
    lifetime="detached",
    get_if_exists=True,
    num_cpus=2,
    memory=5 * 1024**3,
    resources={"prottrans": 1},  # only the ProtTrans worker has it
    runtime_env=RUNTIME_ENV,
).remote()

# The first call waits for pip install, the 11.3GB checkpoint download and
# model loading.
dim = ray.get(actor.warm_up.remote())
print(f"actor 'prottrans' ready, embedding dim {dim}")
