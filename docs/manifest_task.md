Yes. After looking through the current `apptainer-dev` branch, I would **not** try to make the manifest describe models as “PyTorch + weights file + architecture file.” Your four existing models are already too heterogeneous for that.

The right abstraction for `vhmodels` is:

> **Model identity → implementation/interface → external sources → runtimes/environments.**

Your current registry already separates runtime execution from model behavior reasonably well: `RuntimeBackend` handles Conda/Apptainer, while each model implements `BaseModel`. 

## Proposed manifest structure

I would evolve `config.json` toward something like this:

```json
{
  "schema_version": "1.0",

  "model": {
    "id": "dinobloom",
    "version": "1.0.0",
    "description": "...",
    "homepage": "https://huggingface.co/virtual-human-chc/DinoBloom"
  },

  "implementation": {
    "class_path": "DinoBloom.model.DinoBloom",
    "capabilities": ["embed"]
  },

  "variants": {
    "s": {},
    "b": {},
    "l": {},
    "g": {}
  },

  "sources": {
    "architecture": {
      "type": "torch_hub",
      "repo": "facebookresearch/dinov2",
      "revision": "<tag-or-commit>"
    },

    "weights": {
      "type": "huggingface",
      "repo_id": "virtual-human-chc/DinoBloom",
      "revision": "<commit>",
      "files": {
        "checkpoint": "pytorch_model_{variant}.bin"
      }
    }
  },

  "runtimes": {
    "conda": {
      "env_name": "vhmodels-dinobloom",
      "platforms": {
        "linux-x86_64": {
          "environment": "environment.linux-x86_64.yml"
        },
        "macos-arm64": {
          "environment": "environment.macos-arm64.yml"
        }
      }
    },

    "apptainer": {
      "platform": "linux-x86_64",
      "python": "3.10",
      "requirements": "requirements.linux-x86_64.txt",
      "torch_backend": "cu126"
    }
  }
}
```

This is mostly a restructuring of information you already have, rather than a completely new concept. Your existing configs currently mix `supported_platforms`, `environment_files`, `apptainer`, `conda_env`, `class_path`, and metadata at the same level. ([GitHub][1])

### The most important part: `sources`

I would make `sources` a **flexible dictionary of typed resources**, not a fixed schema saying every model has `architecture` and `weights`.

For example, support source types such as:

```text
huggingface
torch_hub
url
git
local
python_package
```

Each type has its own schema.

For Hugging Face:

```json
{
  "type": "huggingface",
  "repo_id": "virtual-human-chc/MolE",
  "revision": "abc123...",
  "files": {
    "config": "config.yaml",
    "checkpoint": "model.pth"
  }
}
```

For Torch Hub:

```json
{
  "type": "torch_hub",
  "repo": "facebookresearch/dinov2",
  "revision": "abc123..."
}
```

For a raw file:

```json
{
  "type": "url",
  "url": "https://...",
  "sha256": "..."
}
```

And importantly, `files` should be optional. A library may manage the whole repository itself.

### This fits all four current models well

**DinoBloom** has two fundamentally different sources:

```text
architecture → torch.hub
weights      → Hugging Face .bin
```

That is exactly what its current `load_model()` does. ([GitHub][2])

**Hyformer** is:

```text
source → Hugging Face

files:
  vocab
  tokenizer_config
  model_config
  checkpoint
```

Its loader explicitly downloads those separate files. 

**MolE** is:

```text
source → Hugging Face
  config.yaml
  model.pth

architecture → installed `mole_package`
```

The actual Python architecture comes from a package installed by the environment, rather than from HF. 

**ProtTrans** is even simpler from the registry's perspective:

```json
{
  "type": "huggingface",
  "repo_id": "virtual-human-chc/{variant}",
  "revision": "..."
}
```

You should **not enumerate tokenizer files, config files, weights, etc.** because `transformers.from_pretrained()` already understands how to resolve that repository. Your loader should let Transformers do its job. 

## I would keep `model.py`

This is important. I would **not try to declaratively encode all loading behavior into the JSON**.

Keep:

```text
BaseModel
├── load_model()
├── embed()
├── predict()
└── generate()
```

because the four models already have very different construction logic. 

The manifest answers:

```text
What is this model?
Where are its resources?
Which variants exist?
Which environments can run it?
Which vhmodels interface does it implement?
```

`model.py` answers:

```text
How exactly are these things assembled
into a working Python model?
```

That separation is important.

## One architectural improvement I would make

Eventually I would move **downloading** out of each `model.py`.

Right now DinoBloom, Hyformer and MolE independently call `hf_hub_download()`. ([GitHub][2])

A more principled architecture would be:

```text
config.json
    ↓
SourceResolver
    │
    ├── HuggingFaceResolver
    ├── TorchHubResolver
    ├── URLResolver
    └── ...
    ↓
resolved/cached resources
    ↓
model.py : load_model(...)
    ↓
actual model
```

That gives you centralized **downloading, caching, revisions, hashes and logging**, while still allowing arbitrary model-specific loading logic.

And I would validate `config.json` with **Pydantic discriminated unions**: `HuggingFaceSource`, `TorchHubSource`, `URLSource`, `CondaRuntime`, `ApptainerRuntime`, etc. That gives you BioImage.IO-like strict manifest validation without forcing all models into the same physical artifact layout.

For models with multiple variants (DinoBloom, ProtTrans, Hyformer) you can create a model-level common config:

```text
DinoBloom/
├── model.json           ← common metadata
├── model.py
└── manifests/
    ├── s.json
    ├── b.json
    ├── l.json
    └── g.json
```text

for example `model.json`  could contain description, class_path, homepage, runtime. While `s.json` contains only variant-specific information. Your registry resolves them into one fully resolved manifest before execution.

The internal flow should finally look like this:

```text
load_model(project="dinobloom", model="s", runtime="apptainer")
        ↓
Registry.resolve("dinobloom", "s")
        ↓
variant manifest
        ↓
validate manifest
        ↓
select runtime
        ↓
ModelProxy / RuntimeBackend
        ↓
worker starts in Conda / Apptainer environment
        ↓
SourceResolver.resolve(manifest.sources)
        │
        ├── HuggingFaceResolver
        ├── TorchHubResolver
        ├── URLResolver
        └── LocalResolver
        ↓
resolved local resources / cache paths
        ↓
import implementation.class_path
        ↓
model.load_model(
    manifest,
    resolved_resources
)
        ↓
actual loaded model
        ↓
embed() / predict() / ...
```

so e.g. for DinoBloom specifically:

```text
Registry
→ DinoBloom / s manifest

Runtime
→ Apptainer

SourceResolver
├── torch_hub source → resolves DINOv2 architecture
└── huggingface source → downloads/caches DinoBloom weights

model.py
→ combines architecture + weights
→ returns loaded DinoBloom model
```

