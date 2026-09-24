# Environment setup (Conda / Apptainer)

Embed needs a **model runtime** — isolated Conda env or Apptainer image per project. Host `pip install -e "<repo_root>.[cli]"` only installs the thin `vhmodels` proxy + `vh-checker` CLI.

**Platforms:** Linux and macOS only. No Windows support.

## When to use this reference

- `conda --version` and the platform runtime (`apptainer` or `limactl`) both fail
- `vh-checker create-env <project>` or `create-apptainer-image <project>` fails
- User on Helmholtz / corp network (Anaconda Miniconda often blocked)

## Diagnose (then stop probing)

```bash
uname -s && uname -m
command -v conda apptainer limactl
```

Host Python tooling (`uv`, `pip`, `poetry`) does **not** replace the model
runtime. The uv executable used to build an Apptainer image is supplied inside
the image build, so it is not a host requirement.

## Install Miniforge (conda-forge) — default path

Use **Miniforge**, not Anaconda Miniconda (`repo.anaconda.com` / `defaults` channel). Miniforge ships `conda` + `mamba` with conda-forge as default — avoids Anaconda licensing blocks at many orgs.

**Script:** `scripts/install-miniforge.sh` (next to this skill's `SKILL.md`).

With user consent:

```bash
bash scripts/install-miniforge.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda --version
```

| Detail | Value |
|--------|--------|
| Install prefix | `~/miniforge3` (override: `MINIFORGE_PREFIX`) |
| Platforms | Darwin arm64/x86_64, Linux aarch64/x86_64/ppc64le |

Installer URLs (script picks automatically):

- `Miniforge3-Darwin-arm64.sh`
- `Miniforge3-Darwin-x86_64.sh`
- `Miniforge3-Linux-aarch64.sh`
- `Miniforge3-Linux-x86_64.sh`
- `Miniforge3-Linux-ppc64le.sh`

Base: `https://github.com/conda-forge/miniforge/releases/latest/download/`

## Apptainer runtime

**Docker not supported.** Use Apptainer on HPC or when Conda is unavailable.
Linux/x86_64 hosts, including HPC systems, require Apptainer. macOS hosts
require Lima instead; `vhmodels` automatically creates and reuses its Linux VM.
Apple Silicon uses VZ and Rosetta's fast translation mode.

Linux install: [apptainer.org/docs/admin/main/installation.html](https://apptainer.org/docs/admin/main/installation.html)

macOS install:

```bash
brew install lima
```

```bash
vh-checker create-apptainer-image <project>   # image: vhmodels-<project>.sif
```

On macOS, keep the checkout, SIF, and input data under your home directory. The
generated Linux/AMD64 SIF can be copied to an x86_64 HPC system and run there
directly.

Each model's `model.json` supplies the Python version, Linux requirements file,
and optional PyTorch backend. The Ubuntu 24.04 image uses uv during the build and
installs the finished environment at `/opt/venv`. Completed uv downloads and
managed Python archives are kept in the per-user build cache at
`${XDG_CACHE_HOME:-~/.cache}/vhmodels/apptainer/uv`; the cache is not included in
the SIF. Override it with `VHMODELS_APPTAINER_CACHE_DIR`, for example to use a
fast per-user scratch filesystem on HPC. On macOS, the override must be under
the user's home directory so Lima can access it. This cache reduces repeated
downloads; rebuilding still creates and compresses a complete SIF, so it does
not reduce the image size.

Then `load_model(..., runtime="apptainer")`.

Definition template: `vhmodels/envs/Apptainer` in the repository checkout.

## Create model env (after Conda or Apptainer works)

**Conda** (default):

```bash
vh-checker create-env dinobloom    # → vhmodels-dinobloom
vh-checker create-env hyformer
vh-checker create-env prottrans
vh-checker create-env mole
```

The first environment/image build and model run can download large dependencies
and Hugging Face weights — warn the user about time, disk, and GPU requirements.

Some `environment.yml` files still list `defaults` or vendor channels; if `create-env` fails on channel policy, Jonas's team may need a conda-forge-only rewrite.

## Environment errors

| Error | Fix |
|-------|-----|
| `conda` not found | Run `scripts/install-miniforge.sh`, `source ~/miniforge3/etc/profile.d/conda.sh` |
| SSL / cert error to `anaconda.com` | Use Miniforge script (GitHub), not Miniconda |
| Environment does not exist | `vh-checker create-env <project>` |
| SIF not found | `vh-checker create-apptainer-image <project>` |
| Import errors in host Python | Expected — inference runs in model sub-env |
