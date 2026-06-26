# Environment setup (Conda / Docker)

Embed needs a **model runtime** — isolated Conda env or Docker image per project. Host `pip install -e .` only installs the thin `vhmodels` proxy.

## When to use this reference

- `conda --version` and `docker --version` both fail
- `vh-checker create-env <project>` or `create-docker-image <project>` fails
- User on Helmholtz / corp network (Anaconda Miniconda often blocked)

## Diagnose (then stop probing)

```bash
uname -s && uname -m
command -v conda docker
```

Host Python (`uv`, `pip`, `poetry`) does **not** replace the model runtime.

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
| Linux without curl | Script runs `apt-get install -y curl ca-certificates` when root |
| Windows | Script prints URL for `Miniforge3-Windows-x86_64.exe`; user runs installer manually |

Installer URLs (script picks automatically):

- `Miniforge3-Darwin-arm64.sh`
- `Miniforge3-Darwin-x86_64.sh`
- `Miniforge3-Linux-aarch64.sh`
- `Miniforge3-Linux-x86_64.sh`
- `Miniforge3-Linux-ppc64le.sh`
- `Miniforge3-Windows-x86_64.exe`

Base: `https://github.com/conda-forge/miniforge/releases/latest/download/`

## Docker alternative

If user prefers containers: [docs.docker.com/get-docker](https://docs.docker.com/get-docker/).

```bash
vh-checker create-docker-image <project>   # image: vhmodels-<project>
```

Then `load_model(..., runtime="docker")`.

Apptainer/Singularity: planned, not in v0.1.

## Create model env (after Conda or Docker works)

**Conda** (default):

```bash
vh-checker create-env dinobloom    # → vhmodels-dinobloom
vh-checker create-env hyformer
vh-checker create-env prottrans
vh-checker create-env mole
```

First run downloads large conda/pip deps and HuggingFace weights — warn user about time, disk, GPU.

Some `environment.yml` files still list `defaults` or vendor channels; if `create-env` fails on channel policy, Jonas's team may need a conda-forge-only rewrite.

## Environment errors

| Error | Fix |
|-------|-----|
| `conda` not found | Run `scripts/install-miniforge.sh`, `source ~/miniforge3/etc/profile.d/conda.sh` |
| SSL / cert error to `anaconda.com` | Use Miniforge script (GitHub), not Miniconda |
| Environment does not exist | `vh-checker create-env <project>` |
| Import errors in host Python | Expected — inference runs in model sub-env |
