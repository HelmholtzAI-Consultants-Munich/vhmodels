# Model manifests

This describes the manifest system: versioned, schema-validated model metadata,
with downloading separated from model assembly. It's a reference for how the
pieces fit together.

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
| `url` | `url`, `sha256?`, `filename?` | one downloaded local path, cached per `url`+`sha256` and checksummed if `sha256` is given |
| `git` | `url`, `revision` | a local clone path, cached per `url`+`revision` |
| `local` | `path` | a path resolved relative to the model directory |
| `python_package` | `name`, `version?` | confirms the package is importable in the current environment |

A `huggingface` source with no `files` (ProtTrans) resolves to just a repo
id/revision — nothing is downloaded, because `transformers.from_pretrained()`
already knows how to fetch and cache that repo itself. Declaring `files`
(DinoBloom's weights, Hyformer's four files, MolE's `config.yaml`/`model.pth`)
is what tells `SourceResolver` to actually call `hf_hub_download` for each one.

#### Pinning

Downloaded sources are cached by what they identify, so a source should name
something that never changes: a commit sha (or tag) for `git` — which is why its
`revision` is required — a commit sha for `huggingface`, and a `sha256` for
`url`. The manifest is then the lockfile: to upgrade, edit it. A new revision or
hash is a different cache key, so the new version is fetched and nothing has to
be deleted by hand. Everything below lives under `<system temp dir>/vhmodels-sources/`.

- `git` — cached in `git/<repo>-<key>`, keyed by `url` *and* `revision`, and the
  directory only appears once the clone *and* the checkout succeeded. A branch
  name works, but it stays pinned to whatever it was at first clone.
- `url` — cached in `url/<key>/<filename>`, keyed by `url` *and* `sha256`. A
  cached file that no longer matches its `sha256` is downloaded again; a fresh
  download that doesn't match raises. Without a `sha256` the file stays pinned to
  its first download.

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
special-case it used to have — the routing is now data, not code. The
`prot_t5_xxl_*` manifests override `weights` the same way, only to
pin a Hugging Face commit in `revision`, which `model.py` passes on to
`from_pretrained()`.

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
`runtimes.conda`, `runtimes.uv`, `runtimes.apptainer`) instead of the old flat
dict. `runtimes.uv` is the dependency spec shared by the Apptainer image and the
Ray venv; `runtimes.apptainer` only marks the image platform and requires it.

## Downloading: SourceResolver

`vhmodels/models/source_resolver.py`'s `SourceResolver.resolve(sources, model_dir)`
turns a `ResolvedManifest.sources` dict into plain, JSON-safe dicts, one per
source name:

| Source type | Resolved dict |
| --- | --- |
| `huggingface` | `{"type", "repo_id", "revision", "files": {name: path}}`; `files` is empty when the source declares none |
| `torch_hub` | `{"type", "repo", "revision", "entrypoint"}` |
| `url` | `{"type", "path", "sha256"}` |
| `git` | `{"type", "path", "revision"}` |
| `local` | `{"type", "path"}` |
| `python_package` | `{"type", "name", "version"}` |

Paths are strings. `url` and `git` sources are cached under the host's temp
directory, so on a cluster set `TMPDIR` to shared storage.

`hf_hub_download` is called in exactly one place, here. `python_package`
sources are passed through without importing them, because the package
exists only in the model environment; the environment build's smoke import
checks it.

`Registry.resolve_resources(project, variant)` combines both steps and returns
`(variant, resources)`. It always runs on the host.

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
for exactly that one field.

The full, Pydantic-validated `Registry`, `SourceResolver` and the schema are
imported only on the host: by the CLI and, lazily, by
`vh_checker.factory.load_model()`. Merely `import vhmodels`, which every worker
does, never pulls in Pydantic. Model code never imports them either, so model
environments contain neither Pydantic nor the download logic:

- the requirements files do not list `pydantic`;
- Conda environments and Ray venvs install `vhmodels` with `--no-deps`;
- Apptainer images only put the source on `PYTHONPATH`.

`huggingface_hub` is a host dependency of `vhmodels`, because the host
performs the downloads.

`tests/test_manifest.py` enforces this boundary in three ways:

- `::test_discovery_has_no_third_party_imports` and
  `::test_base_and_worker_stay_dependency_free` import those modules in a
  subprocess and check that `pydantic` never lands in `sys.modules`;
- `::test_model_code_leaves_resolution_to_the_host` checks that no `model.py`
  imports the registry, the resolver or Pydantic.

## How model.py uses it

The host resolves the manifest and sources, then hands the result to the worker:

- **Conda and Apptainer:** the process manager resolves when it starts the worker, and the `load` message carries `"model": <variant>` and `"resources": {...}`.
- **Ray:** `load_model(runtime="ray")` resolves on the driver and passes both to the actor.

Each model's `load_model(model, resources, **kwargs)` receives them as plain data:

```python
class DinoBloom(BaseModel):
    PROJECT = "dinobloom"

    def load_model(self, model, resources, **kwargs):
        architecture = resources["architecture"]
        self.model = torch.hub.load(architecture["repo"], architecture["entrypoint"], pretrained=False)
        ckpt_path = resources["weights"]["files"]["checkpoint"]
        ...
```

Everything downstream of that — device placement, checkpoint surgery,
tokenizer wiring, output post-processing — stays in `model.py`: the manifest answers *what this model is
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
