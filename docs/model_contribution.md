# Adding models

Thank you for taking the time to add your model to `vhmodels`! The process consists of the following steps:

1. upload your weights to Hugging Face
2. add a description of your model
3. create a new model entry in `vhmodels`
4. add example data
5. add your model to the quick start instructions

## Step 1: Hugging Face upload

Upload your model weights to Hugging Face. Use a format that can be easily
loaded when your model is used. You can find an overview of different formats
on the [Hugging Face Blog](https://huggingface.co/blog/ngxson/common-ai-model-formats).

## Step 2: Model description

In order for others to easily understand how your model can be used, you need a good description. Please include the following sections:

- Short description: overview about the model
- Model versions (if applicable): different available versions
- Long description: in-depth description of your model
- Metadata
    - Input
        - Description: short description of the input data
        - Input format:
            - Shape: shape of the data
            - Data type (for non-tabular data): type of the data
            - Columns (for tabular data): description of column names with data types
        - Example input file: path to an example input file
    - Model
        - Modality: on which modality does the model work on, e.g. chemical compounds or images
        - Scale: on which scale does the model work on, e.g. one image, one chemical compound, combination of chemical compound and bacterial strain
        - Description: concise description what the model does with the input data
        - Training data (if applicable): data used for training
        - Publication (if applicable): link to publication
    - Output
        - Description: short description of the output data
        - Output format: e.g. table or tensor
            - Shape: shape of the data
            - Data type (for non-tabular data): type of the data
            - Columns (for tabular data): description of column names with data types
- Installation: list the model's runtime requirements. In the repository, you
  will provide both a Conda environment file and a uv-compatible Linux
  requirements file as described below.
- Example: example code how to perform inference with your model
- References: references cited throughout the description/model card
- Copyright: information about the copyright

## Step 3: add the model to `vhmodels`

Create a new folder for your model in `vhmodels/models`. The folder needs to contain the following files:

- `model.json`
```json
{
    "schema_version": "1.0",
    "model": {
        "id": "<your_model>",
        "version": "1.0.0",
        "description": "short description of <your_model>",
        "homepage": "link to Hugging Face repository"
    },
    "implementation": {
        "class_path": "<Your_model>.model.<Your_model>"
    },
    "supported_platforms": ["linux-x86_64"],
    "runtimes": {
        "conda": {
            "env_name": "vhmodels-<your_model>",
            "platforms": {
                "linux-x86_64": { "environment": "environment.linux-x86_64.yml" }
            }
        },
        "apptainer": {
            "platform": "linux-x86_64",
            "python": "3.10",
            "requirements": "requirements.linux-x86_64.txt",
            "torch_backend": "cu126"
        }
    },
    "sources": {
        "weights": {
            "type": "huggingface",
            "repo_id": "virtual-human-chc/<your_model>"
        }
    }
}
```

`supported_platforms` must include `linux-x86_64` for Apptainer. Add
`macos-arm64` and a matching platform entry under `runtimes.conda.platforms`
only when you also provide `environment.macos-arm64.yml`. In
`runtimes.apptainer`, `python` and `requirements` are required; `torch_backend`
is optional and selects the uv PyTorch backend (for example, `cu126`). If a
transitive package must be omitted, set `exclude` to a uv excludes file in the
same model directory.

`sources` describes where your model's resources come from, as one or more
independently typed entries (`huggingface`, `torch_hub`, `url`, `git`, `local`,
`python_package` -- see [docs/manifest.md](manifest.md) for the full
field reference). It is validated by Pydantic and resolved for you at load
time through `vhmodels.models.registry.REGISTRY` and
`vhmodels.models.source_resolver.SourceResolver`; `model.py` should read
already-resolved paths/repo ids from there rather than calling
`hf_hub_download` or similar directly.

- `manifests/<variant>.json` (one per variant your model exposes, e.g. the
  value passed as `vhmodels.load_model(project=..., model=<variant>)`)

```json
{
    "variant": "<variant_id>"
}
```

A variant manifest only needs a `sources` block when that variant's resources
differ from `model.json`'s defaults -- for example when a filename or repo id
depends on the variant. String fields in `model.json`'s `sources` may contain
a `{variant}` placeholder, substituted with the requested variant id when the
manifest is resolved; see `ProtTrans/model.json` and its
`manifests/prot_electra_bfd.json` override for a worked example. A model with
only one variant still needs exactly one file here, conventionally
`manifests/default.json`.

- `environment.linux-x86_64.yml`

```yaml
name: vhmodels-<your_model>
channels:
  - conda-forge
  - nodefaults
dependencies:
  - # your dependencies here, you can also add pip dependencies
```

You can only use conda-forge channels, as official anaconda channels are blocked on the HPC.
When adding `torch` via pip, you can specify the CUDA versions.

Additionally, you can also provide an `environment.macos-arm64.yml` file.

- `requirements.linux-x86_64.txt`

List the Python packages for the Apptainer image in standard pip requirements
syntax. Keep its Python and package versions consistent with the Conda file.
The shared definition starts from Ubuntu 24.04, installs these requirements with
uv, and stores the finished environment at `/opt/venv`. uv is supplied by the
image build; contributors and users do not need it on the host. If a model needs
additional Ubuntu packages, add them to `vhmodels/envs/Apptainer`. List only
what the model itself needs. Manifest validation and downloads run on the host,
so do not add `pydantic` or other `vhmodels` host dependencies.

- `requirements-exclude.linux-x86_64.txt` (optional)

Use an exclusion file only when a transitive dependency conflicts with the
package selected by the model, as with Hyformer's legacy `rdkit-pypi`
dependency. List one excluded requirement per line, explain why it is excluded
in a comment, and reference the file through `runtimes.apptainer.exclude` in
`model.json`.

- `model.py`

Here, the model loading and calling is implemented. Please have a look at the already implemented
models as examples. It might be necessary to adapt your existing model code so that it can be
easily used in `vhmodels`.

```python
from vhmodels.vh_checker.base import BaseModel


class YourModel(BaseModel):
    PROJECT = "your_model"

    def __init__(self):
        self.model = None

    def load_model(self, model, resources, **kwargs):
        """Load the variant ``model`` from the host-resolved ``resources``."""
        # resources["weights"] is e.g. {"repo_id": ..., "files": {"checkpoint": "/path"}};
        # the host resolves and downloads them (see docs/manifest.md). Never
        # import the registry or SourceResolver here.

    def embed(self, input, **kwargs):
        """
        Creates embeddings for the provided input data.

        The model is expected to be already loaded before calling this function.

        Parameters
        ----------
        inputs :
            Either the data itself or a path to the data file.

        Returns
        -------
        dict
            A dictionary containing:
            - 'output': list of lists
        """

        # perform the necessary preprocessing and call the model with
        # the input data to generate embeddings
        # make sure that the input data is on the correct device and
        # the output on the CPU again

        return {"output": embeddings}

    # predict and generate functionality is not implemented yet
    def predict(self, input, **kwargs):
        pass

    def generate(self, input, **kwargs):
        pass

```

## Step 4: add example data

If needed, add a small example data file to `example_data/<your_model>` that can
be used in the inference examples.

## Step 5: add quick start instructions

In the README, add your model to the quick start instructions (add all required arguments for the model loading):

```python
import vhmodels

model = vhmodels.load_model(project='<your_model>')
results = model.embed(input='example_data/<your_model>/example.data')
print(results)
```

Also add an inference script `examples/<your_model>.py`, following the existing
scripts, so the model can be run locally and through the
[HPC job submission](../hpc-job-submit/README.md) workflow.
