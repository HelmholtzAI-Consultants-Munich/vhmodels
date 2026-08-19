# Model manifests

This describes the manifest system introduced from [manifest_task.md](manifest_task.md):
versioned, schema-validated model metadata, with downloading separated from
model assembly. It's a reference for how the pieces fit together, not a
restatement of that design doc.

## File layout

Each model directory carries two kinds of manifest, plus its `model.py`:

```
vhmodels/models/DinoBloom/
├── model.json              # identity, implementation, runtimes, default sources
├── manifests/
│   ├── s.json               # one file per variant; usually just {"variant": "s"}
│   ├── b.json
│   ├── l.json
│   └── g.json
└── model.py
```

`model.json` holds everything shared across variants. `manifests/<variant>.json`
holds only what differs — often nothing beyond the variant id itself, since
`model.json`'s sources can reference `{variant}` (see below). A model with a
single variant (MolE) still gets one file, `manifests/default.json`, so the
registry never has to special-case "no variants."

`vhmodels.load_model(project=..., model=<variant>)`'s `model` argument *is*
the variant id looked up here — DinoBloom's `"s"/"b"/"l"/"g"`, ProtTrans's
`"prot_bert_bfd"` and friends, Hyformer's `"hyformer_molecules_50M"` etc.

## Schema

Defined with Pydantic in `vhmodels/models/schema.py`:

- `ModelManifest` — validates `model.json`: `schema_version`, `model` (id,
  version, description, homepage), `implementation.class_path`, `runtimes`
  (conda/apptainer), `supported_platforms`, and default `sources`.
- `VariantManifest` — validates `manifests/<variant>.json`: `variant`, an
  optional `description` override, and `sources` overrides.
- `ResolvedManifest` — the merged, fully validated result of resolving one
  `(project, variant)` pair; this is what `model.py` and `SourceResolver`
  actually consume.

All three reject unknown fields (`extra="forbid"`), so a typo or a leftover
field from the old flat `config.json` fails loudly at load time instead of
being silently ignored.

### Sources

`sources` is a dict of independently-typed, named resources, validated as a
Pydantic discriminated union on `type`:

| type | fields | resolves to |
|---|---|---|
| `huggingface` | `repo_id`, `revision?`, `files?` | repo id/revision, plus one downloaded local path per `files` entry |
| `torch_hub` | `repo`, `revision?`, `entrypoint?` | passed through — `torch.hub.load` manages its own cache |
| `url` | `url`, `sha256?`, `filename?` | one downloaded local path, checksummed if `sha256` is given |
| `git` | `url`, `revision?` | a local clone path |
| `local` | `path` | a path resolved relative to the model directory |
| `python_package` | `name`, `version?` | confirms the package is importable in the current environment |

A `huggingface` source with no `files` (ProtTrans) resolves to just a repo
id/revision — nothing is downloaded, because `transformers.from_pretrained()`
already knows how to fetch and cache that repo itself. Declaring `files`
(DinoBloom's weights, Hyformer's four files, MolE's `config.yaml`/`model.pth`)
is what tells `SourceResolver` to actually call `hf_hub_download` for each one.

String fields anywhere inside a source may contain a `{variant}` placeholder.
It's substituted with the requested variant id when the manifest is resolved
— see DinoBloom's `"entrypoint": "dinov2_vit{variant}14"` and
`"checkpoint": "pytorch_model_{variant}.bin"`, or ProtTrans's
`"repo_id": "virtual-human-chc/{variant}"`. This is why most variant manifests
are just `{"variant": "s"}`: the templated model-level source already says
where everything is.

When a variant needs something genuinely different, its manifest overrides
that named source outright. ProtTrans's `prot_electra_bfd` is the real
example: its tokenizer and weights come from two *different* upstream repos
(`prot_electra_generator_bfd` / `prot_electra_discriminator_bfd`), so
`manifests/prot_electra_bfd.json` replaces both `tokenizer` and `weights`
wholesale instead of relying on the `{variant}` template. This also let
`ProtTrans/model.py` drop the `if model == "prot_electra_bfd": ...`
special-case it used to have — the routing is now data, not code.

## Resolving a manifest

`vhmodels.models.registry.Registry` (a singleton instance, `REGISTRY`, is
constructed once per process) does the merge:

```
Registry.resolve(project, variant)
    -> load model.json                     (ModelManifest)
    -> load manifests/<variant>.json       (VariantManifest)
    -> merge sources: variant keys override model-level keys of the same name
    -> substitute {variant} in every string field
    -> re-validate the substituted sources against the Source union
    -> ResolvedManifest
```

`variant` can be omitted when a model has exactly one — `Registry.resolve("mole")`
resolves to `"default"` automatically; models with several variants must name
one, or `resolve()` raises with the list of valid ids.

`Registry` also backs the CLI: `vh-checker list`, `create-env`, and
`create-apptainer-image` all read `ModelManifest` fields (`supported_platforms`,
`runtimes.conda`, `runtimes.apptainer`) instead of the old flat dict.

## Downloading: SourceResolver

`vhmodels/models/source_resolver.py`'s `SourceResolver.resolve(sources, model_dir)`
turns a `ResolvedManifest.sources` dict into a dict of small `Resolved*`
objects (`ResolvedHuggingFace`, `ResolvedTorchHub`, ...) with plain attributes
— local `Path`s where something was downloaded, repo ids/revisions where
nothing needed to be. This is the one place that calls `hf_hub_download`;
model code no longer does it independently per model.

## Why the worker stays dependency-free

`vhmodels.vh_checker.worker` (the process that actually loads a model, whether
in a Conda env or an Apptainer container) and `vhmodels.vh_checker.base`
(`BaseModel.get_class`) are imported by **every** isolated model process,
including the dependency-free fixtures used by `test_apptainer_integration.py`
and `test_conda_integration.py` — those deliberately run in an environment
with no third-party packages installed at all, to test the worker/socket
plumbing in isolation from any model's dependencies.

That means `base.py` cannot depend on Pydantic just to look up a
`class_path`. It uses `vhmodels/models/discovery.py` instead — a
standard-library-only module that reads `model.json` with plain `json.load`
for exactly that one field. The full, Pydantic-validated `Registry` (and
`SourceResolver`, and the schema) is only imported:

- host-side, by the CLI and by `vh_checker.factory.load_model()` (lazily, so
  merely `import vhmodels` — which every worker does — never pulls in
  Pydantic), and
- by each real model's own `model.py`, which already depends on `torch` /
  `transformers` / `huggingface_hub` and now also lists `pydantic` in its
  `requirements.<platform>.txt` (Apptainer) — the Conda path gets it for free
  since it's a `vhmodels` package dependency, installed via `pip install -e`.

`tests/test_manifest.py::test_discovery_has_no_third_party_imports` and
`::test_base_and_worker_stay_dependency_free` assert this boundary directly by
importing those modules in a subprocess and checking `pydantic` never lands in
`sys.modules`.

## How model.py uses it

The wire protocol between the host and a worker is unchanged: `load_model`
still just takes a variant string and keyword arguments. Each model's
`load_model` resolves its own manifest and sources at the top:

```python
class DinoBloom(BaseModel):
    PROJECT = "dinobloom"

    def load_model(self, model=None, **kwargs):
        manifest = REGISTRY.resolve(self.PROJECT, model)
        resources = SourceResolver().resolve(manifest.sources, manifest.model_dir)

        architecture = resources["architecture"]
        self.model = torch.hub.load(architecture.repo, architecture.entrypoint, pretrained=False)
        ckpt_path = resources["weights"].files["checkpoint"]
        ...
```

Everything downstream of that — device placement, checkpoint surgery,
tokenizer wiring, output post-processing — stays in `model.py`, per
manifest_task.md's original point: the manifest answers *what this model is
and where its resources are*, `model.py` answers *how they're assembled into
a working model*.

## Tests

`tests/test_manifest.py` covers the schema (discriminated union dispatch and
`extra="forbid"` rejection), `discovery` (dependency-free scanning, warns and
skips an unparsable `model.json` instead of failing the whole registry),
`Registry` (variant listing, `{variant}` substitution, the `prot_electra_bfd`
override, an isolated `models_dir` for fixture-only registries), and
`SourceResolver` (each source type, using `sys.modules` injection to fake
`huggingface_hub` without requiring it installed on the host).
