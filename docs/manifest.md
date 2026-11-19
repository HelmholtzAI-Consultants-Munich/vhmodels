# Model manifests

Each model is described by `model.json` and one `manifests/<variant>.json` per
variant. Both are checked when the model loads; unknown fields are errors.

## File layout

```text
vhmodels/models/DinoBloom/
├── model.json        identity, implementation, runtimes, default sources
├── manifests/
│   ├── s.json        one file per variant; usually just {"variant": "s"}
│   ├── b.json
│   ├── l.json
│   └── g.json
└── model.py
```

- **`model.json`** holds what all variants share: `model` (id, version, description, homepage), `implementation` (`class_path`, `capabilities`, `input`), `runtimes` (conda, uv, apptainer), `supported_platforms`, the default `sources` and the `card`.
- **`manifests/<variant>.json`** holds the variant id, its `params` and, only where it differs, a `description`, `sources` and `card`. A model with one variant has `manifests/default.json`.
- **The variant id** is the `model` argument of `vhmodels.load_model(project=..., model=<variant>)`, e.g. DinoBloom's `"s"`, `"b"`, `"l"`, `"g"`. It can be left out for a model with one variant.
- **`runtimes.uv`** is the dependency spec of the Apptainer image, the OCI image and the Ray venv; `runtimes.apptainer` requires it.

## Input

`implementation.input` declares the shape of `input` for `embed` and
`predict`. `load_model` (every runtime) and Ray Serve reject other shapes
before the model runs, with `ValueError` (HTTP 422). The model still checks
the values (valid SMILES, image size, TSV columns).

| `type` | Accepts | Models |
| --- | --- | --- |
| `sequences` | `list[str]` | Hyformer, ProtTrans |
| `file` | `str` path | MolE |
| `images` | `str` image or folder path, or `list[str]` of paths | DinoBloom, HOptimus0 |
| `named_files` | `dict` of `str` paths with exactly the names in `keys` | Nicheformer (`["technology_mean", "data"]`) |

Without `input`, nothing is checked.

## Sources

`sources` is a dict of named resources, each with a `type`:

| type | fields | resolves to |
|---|---|---|
| `huggingface` | `repo_id`, `revision?`, `files?` | repo id/revision, plus one downloaded local path per `files` entry |
| `torch_hub` | `repo`, `revision?`, `entrypoint?` | passed through: `torch.hub.load` manages its own cache |
| `url` | `url`, `sha256?`, `filename?` | one downloaded local path, checked against `sha256` if given |
| `git` | `url`, `revision` | a local clone path |
| `local` | `path` | a path relative to the model directory |
| `python_package` | `name`, `version?` | passed through; the package must be in the model environment |

- **Hugging Face.** Without `files` (ProtTrans), nothing is downloaded: `from_pretrained()` fetches the repo itself. Each entry in `files` is downloaded.
- **Pinning.** Name something that never changes: a commit in `revision` for `huggingface` and `git`, a `sha256` for `url`. To upgrade, edit the manifest; the new version is downloaded. A branch name, or a `url` without `sha256`, stays at its first download.
- **Cache.** `git` and `url` sources are cached under `<system temp dir>/vhmodels-sources/`; on a cluster, set `TMPDIR` to shared storage.
- **Variants.** Strings in a source may contain `{variant}`, replaced by the variant id, e.g. DinoBloom's `"entrypoint": "dinov2_vit{variant}14"` or ProtTrans's `"repo_id": "virtual-human-chc/{variant}"`. A variant with different resources overrides the source by name in its manifest, e.g. ProtTrans's `manifests/prot_electra_bfd.json` replaces `tokenizer` and `weights`.

## Model card

`card` names the Hugging Face model card that describes the model.
`vh-checker catalog` builds the [model catalog](../vhmodels/models/README.md) from the manifests and
these cards. As in `sources`, `repo_id` may contain `{variant}`, and a variant
manifest can replace the card, e.g. ProtTrans's `manifests/prot_electra_bfd.json`,
which has no repo of its own.

```json
"card": { "repo_id": "virtual-human-chc/{variant}" }
```

The catalog reads four keys from the card's metadata (its YAML header):

| Card key | Catalog column | Example |
| --- | --- | --- |
| `license` | License of the weights | `apache-2.0` |
| `vh_modality` | Modality | `proteins` |
| `vh_citation` | Citation, as a DOI or URL | `https://doi.org/10.1038/s41592-025-02814-z` |
| `vh_developer` | Developer | `Koch et al. (Marr Lab, Helmholtz Munich)` |

`vh_modality` is one of `microscopy images`, `histology`, `molecules`,
`peptides`, `proteins` or `spatial transcriptomics`; add a value only for a new
modality.

A missing key is an error for a `virtual-human-chc` card. An external card
cannot be edited, so `card.overrides` (`license`, `modality`, `citation`,
`developer`) fills its gaps; overrides take precedence over the card. See
`HOptimus0/model.json`:

```json
"card": {
  "repo_id": "bioptimus/H-optimus-0",
  "overrides": { "modality": "histology", "developer": "Bioptimus", "citation": "https://..." }
}
```

`params` in a variant manifest is the parameter count shown in the catalog,
written by hand, e.g. `"86M"` or `"1.1B"`.

## What model.py receives

vhmodels downloads the sources before the model loads and passes them to
`load_model(model, resources, **kwargs)` as plain data, one dict per source
name:

| Source type | Resolved dict |
| --- | --- |
| `huggingface` | `{"type", "repo_id", "revision", "files": {name: path}}`; `files` is empty when the source declares none |
| `torch_hub` | `{"type", "repo", "revision", "entrypoint"}` |
| `url` | `{"type", "path", "sha256"}` |
| `git` | `{"type", "path", "revision"}` |
| `local` | `{"type", "path"}` |
| `python_package` | `{"type", "name", "version"}` |

```python
class DinoBloom(BaseModel):
    PROJECT = "dinobloom"

    def load_model(self, model, resources, **kwargs):
        architecture = resources["architecture"]
        self.model = torch.hub.load(architecture["repo"], architecture["entrypoint"], pretrained=False)
        ckpt_path = resources["weights"]["files"]["checkpoint"]
        ...
```

The manifest says what the model is and where its resources are; `model.py`
builds a working model from them. `model.py` never downloads sources itself.
