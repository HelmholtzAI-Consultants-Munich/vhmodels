"""Resolves manifest ``sources`` into local paths, centralizing downloads.

Each entry in ``ResolvedManifest.sources`` describes *where* a resource comes
from; ``SourceResolver`` runs on the host and turns that description into a
plain, JSON-safe dict that the worker passes to ``model.load_model`` -- a local
path, a repo id, a package name. This keeps ``hf_hub_download`` /
cache-directory bookkeeping (and Pydantic) out of every model environment.

Downloads only happen for sources that declare explicit ``files`` -- a
``huggingface`` source without ``files`` (see ``ProtTrans/model.json``)
resolves to just a repo id/revision, letting the model's own loader
(``transformers.from_pretrained``) do the fetching. ``torch_hub`` is passed
through unchanged for the same reason: ``torch.hub.load`` manages its own
cache.
"""

import hashlib
from pathlib import Path

from vhmodels.models.schema import (
    GitSource,
    HuggingFaceSource,
    LocalSource,
    PythonPackageSource,
    TorchHubSource,
    URLSource,
)


def _cache_key(*parts):
    """Short, stable name for what a cache entry holds: a hash of its identity."""
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


class SourceResolver:
    """Turns typed manifest sources into locally usable resources."""

    def resolve(self, sources, model_dir=None):
        """Resolve every named source into ``{name: dict}`` with string paths."""
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
                files[name] = hf_hub_download(
                    repo_id=source.repo_id,
                    filename=filename,
                    revision=source.revision,
                )
        return {
            "type": "huggingface",
            "repo_id": source.repo_id,
            "revision": source.revision,
            "files": files,
        }

    def _resolve_torch_hub(self, source):
        # torch.hub.load manages its own on-disk cache; the model loader
        # calls it directly with these fields.
        return {
            "type": "torch_hub",
            "repo": source.repo,
            "revision": source.revision,
            "entrypoint": source.entrypoint,
        }

    def _resolve_url(self, source):
        import os
        import tempfile
        import urllib.request

        filename = source.filename or source.url.rsplit("/", 1)[-1]
        # Keyed on url and sha256 so distinct sources never share a file; the
        # leaf keeps the original filename for consumers that read its extension.
        destination = (
            Path(tempfile.gettempdir())
            / "vhmodels-sources"
            / "url"
            / _cache_key(source.url, source.sha256 or "")
            / filename
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        def sha256_of(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        # A published file was verified when it was downloaded, so a mismatch
        # here means it was damaged since: fetch it again instead of failing.
        if destination.exists() and (
            not source.sha256 or sha256_of(destination) == source.sha256
        ):
            return {"type": "url", "path": str(destination), "sha256": source.sha256}

        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            urllib.request.urlretrieve(source.url, temporary_path)
            if source.sha256:
                digest = sha256_of(temporary_path)
                if digest != source.sha256:
                    raise ValueError(
                        f"Checksum mismatch for '{source.url}': "
                        f"expected {source.sha256}, got {digest}."
                    )
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return {"type": "url", "path": str(destination), "sha256": source.sha256}

    def _resolve_git(self, source):
        import os
        import shutil
        import subprocess
        import tempfile

        name = source.url.rsplit("/", 1)[-1].removesuffix(".git")
        # Keyed on url and revision, so an existing clone is always the one the
        # manifest asks for; changing the revision simply clones afresh.
        destination = (
            Path(tempfile.gettempdir())
            / "vhmodels-sources"
            / "git"
            / f"{name}-{_cache_key(source.url, source.revision)}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary_path = Path(
                tempfile.mkdtemp(dir=destination.parent, prefix=f".{name}.")
            )
            try:
                subprocess.run(
                    ["git", "clone", source.url, str(temporary_path)], check=True
                )
                subprocess.run(
                    ["git", "-C", str(temporary_path), "checkout", source.revision],
                    check=True,
                )
                try:
                    os.replace(temporary_path, destination)
                except OSError:
                    # A concurrent resolve published the same key first. Its
                    # clone is equivalent to ours, so keep it and drop ours.
                    if not destination.is_dir():
                        raise
            finally:
                shutil.rmtree(temporary_path, ignore_errors=True)
        return {"type": "git", "path": str(destination), "revision": source.revision}

    def _resolve_local(self, source, model_dir):
        path = Path(source.path)
        if not path.is_absolute() and model_dir is not None:
            path = model_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Local source not found: {path}")
        return {"type": "local", "path": str(path)}

    def _resolve_python_package(self, source):
        # Installed in the model environment, not on the host; env builds check it.
        return {
            "type": "python_package",
            "name": source.name,
            "version": source.version,
        }
