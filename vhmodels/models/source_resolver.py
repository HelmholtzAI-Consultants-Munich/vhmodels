"""Resolves manifest ``sources`` into local paths, centralizing downloads.

Each entry in ``ResolvedManifest.sources`` describes *where* a resource comes
from; ``SourceResolver`` turns that description into something ``model.py``
can hand straight to a library call -- a local path, a repo id, a package
name. This keeps ``hf_hub_download`` / cache-directory bookkeeping in one
place instead of duplicated across every model's ``load_model``.

Downloads only happen for sources that declare explicit ``files`` -- a
``huggingface`` source without ``files`` (see ``ProtTrans/model.json``)
resolves to just a repo id/revision, letting the model's own loader
(``transformers.from_pretrained``) do the fetching. ``torch_hub`` is passed
through unchanged for the same reason: ``torch.hub.load`` manages its own
cache.
"""

import importlib.util
from pathlib import Path

from vhmodels.models.schema import (
    GitSource,
    HuggingFaceSource,
    LocalSource,
    PythonPackageSource,
    TorchHubSource,
    URLSource,
)


class ResolvedHuggingFace:
    def __init__(self, repo_id, revision, files):
        self.repo_id = repo_id
        self.revision = revision
        self.files = files  # {name: Path}, empty when the source has no files


class ResolvedTorchHub:
    def __init__(self, repo, revision, entrypoint):
        self.repo = repo
        self.revision = revision
        self.entrypoint = entrypoint


class ResolvedURL:
    def __init__(self, path, sha256):
        self.path = path
        self.sha256 = sha256


class ResolvedGit:
    def __init__(self, path, revision):
        self.path = path
        self.revision = revision


class ResolvedLocal:
    def __init__(self, path):
        self.path = path


class ResolvedPythonPackage:
    def __init__(self, name, version):
        self.name = name
        self.version = version


class SourceResolver:
    """Turns typed manifest sources into locally usable resources."""

    def resolve(self, sources, model_dir=None):
        """Resolve every named source. Returns ``{name: Resolved*}``."""
        model_dir = Path(model_dir) if model_dir is not None else None
        return {
            name: self._resolve_one(source, model_dir)
            for name, source in sources.items()
        }

    def _resolve_one(self, source, model_dir):
        if isinstance(source, HuggingFaceSource):
            return self._resolve_huggingface(source)
        if isinstance(source, TorchHubSource):
            return self._resolve_torch_hub(source)
        if isinstance(source, URLSource):
            return self._resolve_url(source)
        if isinstance(source, GitSource):
            return self._resolve_git(source)
        if isinstance(source, LocalSource):
            return self._resolve_local(source, model_dir)
        if isinstance(source, PythonPackageSource):
            return self._resolve_python_package(source)
        raise NotImplementedError(f"Unsupported source type: {type(source)!r}")

    def _resolve_huggingface(self, source):
        files = {}
        if source.files:
            from huggingface_hub import hf_hub_download

            for name, filename in source.files.items():
                files[name] = Path(
                    hf_hub_download(
                        repo_id=source.repo_id,
                        filename=filename,
                        revision=source.revision,
                    )
                )
        return ResolvedHuggingFace(source.repo_id, source.revision, files)

    def _resolve_torch_hub(self, source):
        # torch.hub.load manages its own on-disk cache; the model loader
        # calls it directly with these fields.
        return ResolvedTorchHub(source.repo, source.revision, source.entrypoint)

    def _resolve_url(self, source):
        import hashlib
        import tempfile
        import urllib.request

        filename = source.filename or source.url.rsplit("/", 1)[-1]
        destination = Path(tempfile.gettempdir()) / "vhmodels-sources" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            urllib.request.urlretrieve(source.url, destination)
        if source.sha256:
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest != source.sha256:
                raise ValueError(
                    f"Checksum mismatch for '{source.url}': "
                    f"expected {source.sha256}, got {digest}."
                )
        return ResolvedURL(destination, source.sha256)

    def _resolve_git(self, source):
        import subprocess
        import tempfile

        destination = (
            Path(tempfile.gettempdir())
            / "vhmodels-sources"
            / (source.url.rsplit("/", 1)[-1].removesuffix(".git"))
        )
        if not destination.exists():
            command = ["git", "clone", source.url, str(destination)]
            subprocess.run(command, check=True)
            if source.revision:
                subprocess.run(
                    ["git", "-C", str(destination), "checkout", source.revision],
                    check=True,
                )
        return ResolvedGit(destination, source.revision)

    def _resolve_local(self, source, model_dir):
        path = Path(source.path)
        if not path.is_absolute() and model_dir is not None:
            path = model_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Local source not found: {path}")
        return ResolvedLocal(path)

    def _resolve_python_package(self, source):
        if importlib.util.find_spec(source.name) is None:
            raise ModuleNotFoundError(
                f"Python package source '{source.name}' is not importable. "
                "It must be installed by the model's runtime environment."
            )
        return ResolvedPythonPackage(source.name, source.version)
