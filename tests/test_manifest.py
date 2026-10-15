"""Tests for the manifest schema, Registry, and SourceResolver.

See docs/manifest.md for the design this exercises: model.json +
manifests/<variant>.json -> Registry.resolve() -> ResolvedManifest ->
SourceResolver -> local resources.
"""

import ast
import hashlib
import subprocess
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from vhmodels.models import discovery
from vhmodels.models.registry import REGISTRY, Registry
from vhmodels.models.schema import (
    HuggingFaceSource,
    ModelManifest,
    Runtimes,
    Source,
    TorchHubSource,
    VariantManifest,
)
from vhmodels.models.source_resolver import SourceResolver
from vhmodels.utils.uv_utils import MODEL_PYTHON, model_uv_spec

_SOURCE_ADAPTER = TypeAdapter(Source)

_REAL_MODELS = [
    "dinobloom",
    "hoptimus0",
    "hyformer",
    "mole",
    "nicheformer",
    "prottrans",
]


# --- schema: discriminated union + strictness -------------------------------


def test_source_union_dispatches_on_type():
    hf = _SOURCE_ADAPTER.validate_python({"type": "huggingface", "repo_id": "org/repo"})
    assert isinstance(hf, HuggingFaceSource)
    assert hf.revision is None
    assert hf.files == {}

    hub = _SOURCE_ADAPTER.validate_python(
        {"type": "torch_hub", "repo": "org/repo", "entrypoint": "fn"}
    )
    assert isinstance(hub, TorchHubSource)


def test_source_union_rejects_unknown_type():
    with pytest.raises(ValidationError):
        _SOURCE_ADAPTER.validate_python({"type": "ftp", "url": "ftp://x"})


def test_source_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _SOURCE_ADAPTER.validate_python(
            {"type": "huggingface", "repo_id": "org/repo", "typo_field": 1}
        )


def test_model_manifest_requires_declared_sections():
    with pytest.raises(ValidationError):
        ModelManifest.model_validate({"schema_version": "1.0"})


def test_apptainer_runtime_requires_uv_spec():
    with pytest.raises(ValidationError, match="requires runtimes.uv"):
        Runtimes.model_validate({"apptainer": {"platform": "linux-x86_64"}})


def test_uv_runtime_has_no_per_model_python():
    with pytest.raises(ValidationError):
        Runtimes.model_validate({"uv": {"requirements": "r.txt", "python": "3.10"}})


def test_model_python_matches_repository_python_version():
    python_version = Path(__file__).resolve().parents[1] / ".python-version"
    assert python_version.read_text(encoding="utf-8").strip() == MODEL_PYTHON


def test_variant_manifest_defaults_to_no_sources():
    manifest = VariantManifest.model_validate({"variant": "s"})
    assert manifest.sources == {}
    assert manifest.description is None


# --- discovery: dependency-free, used inside every worker -------------------


def test_discovery_has_no_third_party_imports():
    # base.py/worker.py run inside dependency-free test fixtures (see
    # tests/fixtures/persistent_worker), so importing discovery must never
    # pull in pydantic or any other third-party package.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vhmodels.models.discovery; "
            "assert 'pydantic' not in sys.modules, sorted(sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_base_and_worker_stay_dependency_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vhmodels.vh_checker.base; "
            "import vhmodels.vh_checker.worker; "
            "assert 'pydantic' not in sys.modules, sorted(sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "model_file", sorted(discovery.models_dir().glob("*/model.py"))
)
def test_model_code_leaves_resolution_to_the_host(model_file):
    # Manifests and sources are resolved on the host; model envs need no Pydantic.
    tree = ast.parse(model_file.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {
        "pydantic",
        "vhmodels.models.registry",
        "vhmodels.models.source_resolver",
    }
    assert not imported & forbidden


def test_ray_actor_and_envs_stay_dependency_free():
    # The actor module is imported inside every model venv, and Ray is an
    # optional extra: neither it nor `import vhmodels` may load ray/pydantic.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vhmodels, vhmodels.vh_ray.actor, "
            "vhmodels.vh_ray.envs; "
            "loaded = {'pydantic', 'ray'} & set(sys.modules); "
            "assert not loaded, loaded",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_discover_finds_real_models():
    discovered = discovery.discover()
    assert set(_REAL_MODELS) <= set(discovered)
    directory, raw = discovered["dinobloom"]
    assert directory.name == "DinoBloom"
    assert raw["implementation"]["class_path"] == "DinoBloom.model.DinoBloom"


def test_find_class_path_for_real_models():
    assert discovery.find_class_path("mole") == "MolE.model.MolE"
    assert (
        discovery.find_class_path("nicheformer")
        == "Nicheformer.model.Nicheformer"
    )


def test_find_class_path_unknown_project_raises():
    with pytest.raises(KeyError):
        discovery.find_class_path("does-not-exist")


def test_discover_skips_unparsable_manifest_and_keeps_others(tmp_path):
    good = tmp_path / "Good"
    good.mkdir()
    (good / "model.json").write_text(
        '{"model": {"id": "good"}, "implementation": {"class_path": "Good.model.Good"}}'
    )
    bad = tmp_path / "Bad"
    bad.mkdir()
    (bad / "model.json").write_text("{not valid json")

    discovered = discovery.discover(tmp_path)

    assert set(discovered) == {"good"}


# --- Registry: merge model.json + manifests/<variant>.json ------------------


def test_registry_singleton_discovers_all_real_models():
    assert set(_REAL_MODELS) <= set(REGISTRY.list_models())


@pytest.mark.parametrize("project", _REAL_MODELS)
def test_registry_get_model_sets_model_dir(project):
    manifest = REGISTRY.get_model(project)
    assert manifest.model_dir
    assert manifest.model.id == project


@pytest.mark.parametrize("project", _REAL_MODELS)
def test_real_models_declare_existing_uv_files(project):
    spec = model_uv_spec(REGISTRY.get_model(project))
    assert spec.requirements_path.is_file()
    assert spec.exclude_path is None or spec.exclude_path.is_file()


def test_has_model_and_unknown_project_raises():
    assert REGISTRY.has_model("dinobloom") is True
    assert REGISTRY.has_model("does-not-exist") is False
    with pytest.raises(ValueError, match="not registered"):
        REGISTRY.get_model("does-not-exist")


def test_list_variants_matches_manifests_directory():
    assert REGISTRY.list_variants("dinobloom") == ["b", "g", "l", "s"]
    assert REGISTRY.list_variants("mole") == ["default"]
    assert REGISTRY.list_variants("nicheformer") == ["default"]
    assert len(REGISTRY.list_variants("prottrans")) == 12


def test_resolve_requires_variant_when_ambiguous():
    with pytest.raises(ValueError, match="requires a variant"):
        REGISTRY.resolve("dinobloom")


def test_resolve_auto_selects_sole_variant():
    resolved = REGISTRY.resolve("mole")
    assert resolved.variant == "default"


def test_resolve_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        REGISTRY.resolve("dinobloom", "xl")


def test_resolve_substitutes_variant_placeholder_in_nested_fields():
    resolved = REGISTRY.resolve("dinobloom", "g")
    assert resolved.sources["architecture"].entrypoint == "dinov2_vitg14"
    assert resolved.sources["weights"].files["checkpoint"] == "pytorch_model_g.bin"

    resolved = REGISTRY.resolve("prottrans", "prot_bert")
    assert resolved.sources["tokenizer"].repo_id == "virtual-human-chc/prot_bert"
    assert resolved.sources["weights"].repo_id == "virtual-human-chc/prot_bert"

    for variant in ("prot_bert_bfd_ss3", "prot_t5_xxl_bfd"):
        resolved = REGISTRY.resolve("prottrans", variant)
        assert resolved.sources["tokenizer"].repo_id == (
            f"virtual-human-chc/{variant}"
        )
        assert resolved.sources["weights"].repo_id == (
            f"virtual-human-chc/{variant}"
        )


def test_resolve_variant_manifest_overrides_model_level_source():
    # prot_electra_bfd's tokenizer/weights point at two different upstream
    # repos instead of the templated "virtual-human-chc/{variant}" default.
    resolved = REGISTRY.resolve("prottrans", "prot_electra_bfd")
    assert (
        resolved.sources["tokenizer"].repo_id
        == "virtual-human-chc/prot_electra_generator_bfd"
    )
    assert (
        resolved.sources["weights"].repo_id
        == "virtual-human-chc/prot_electra_discriminator_bfd"
    )


def test_prottrans_xxl_variants_pin_their_weights_revision():
    pins = {
        "prot_t5_xxl_uniref50": "2bd8ac66252842975aead5859c9f8e5d1e706ed2",
        "prot_t5_xxl_bfd": "e13d3644de647fb94c6afd12fe09d8c3667e97ee",
    }
    for variant, revision in pins.items():
        resolved = REGISTRY.resolve("prottrans", variant)
        assert resolved.sources["weights"].repo_id == f"virtual-human-chc/{variant}"
        assert resolved.sources["weights"].revision == revision
        # The tokenizer keeps following the repo's default revision.
        assert resolved.sources["tokenizer"].revision is None


def test_resolve_description_falls_back_to_model_level():
    resolved = REGISTRY.resolve("dinobloom", "s")
    assert resolved.description == REGISTRY.get_model("dinobloom").model.description


def test_registry_isolated_by_models_dir(tmp_path):
    project = tmp_path / "Solo"
    project.mkdir()
    (project / "model.json").write_text(
        '{"schema_version": "1.0", '
        '"model": {"id": "solo", "version": "0.1.0", "description": "d"}, '
        '"implementation": {"class_path": "Solo.model.Solo"}, '
        '"supported_platforms": ["linux-x86_64"], "runtimes": {}}'
    )
    (project / "manifests").mkdir()
    (project / "manifests" / "only.json").write_text('{"variant": "only"}')

    registry = Registry(models_dir=tmp_path)

    assert registry.has_model("solo")
    assert not registry.has_model("dinobloom")
    resolved = registry.resolve("solo")
    assert resolved.variant == "only"


# --- SourceResolver -----------------------------------------------------------


@pytest.fixture
def fake_huggingface_hub(monkeypatch):
    calls = []
    fake_module = types.ModuleType("huggingface_hub")

    def hf_hub_download(repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        return f"/fake-cache/{repo_id}/{filename}"

    fake_module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    return calls


def test_resolve_huggingface_downloads_declared_files(fake_huggingface_hub):
    resolved = REGISTRY.resolve("dinobloom", "b")
    resources = SourceResolver().resolve(resolved.sources, resolved.model_dir)

    weights = resources["weights"]
    assert weights["repo_id"] == "virtual-human-chc/DinoBloom"
    assert weights["files"]["checkpoint"].endswith("pytorch_model_b.bin")
    assert fake_huggingface_hub == [
        ("virtual-human-chc/DinoBloom", "pytorch_model_b.bin", None)
    ]


def test_resolve_huggingface_without_files_skips_download(fake_huggingface_hub):
    # ProtTrans sources declare no "files" -- transformers.from_pretrained()
    # resolves the repo itself, so SourceResolver must not touch the network.
    resolved = REGISTRY.resolve("prottrans", "prot_bert")
    resources = SourceResolver().resolve(resolved.sources, resolved.model_dir)

    assert resources["tokenizer"]["files"] == {}
    assert resources["tokenizer"]["repo_id"] == "virtual-human-chc/prot_bert"
    assert fake_huggingface_hub == []


def test_resolve_torch_hub_passes_through_without_downloading(fake_huggingface_hub):
    resolved = REGISTRY.resolve("dinobloom", "s")
    resources = SourceResolver().resolve(resolved.sources, resolved.model_dir)

    architecture = resources["architecture"]
    assert architecture["repo"] == "facebookresearch/dinov2"
    assert architecture["entrypoint"] == "dinov2_vits14"


def test_resolve_python_package_passes_through_without_importing():
    # Installed only in the model environment, so the host must not import it.
    resolved = _SOURCE_ADAPTER.validate_python(
        {"type": "python_package", "name": "definitely_not_a_real_package"}
    )
    result = SourceResolver()._resolve_python_package(resolved)
    assert result == {
        "type": "python_package",
        "name": "definitely_not_a_real_package",
        "version": None,
    }


def test_resolve_local_source_relative_to_model_dir(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"data")
    source = _SOURCE_ADAPTER.validate_python({"type": "local", "path": "weights.bin"})

    resolved = SourceResolver().resolve({"weights": source}, model_dir=tmp_path)

    assert resolved["weights"]["path"] == str(tmp_path / "weights.bin")


def test_resolve_local_source_missing_raises(tmp_path):
    source = _SOURCE_ADAPTER.validate_python({"type": "local", "path": "missing.bin"})
    with pytest.raises(FileNotFoundError):
        SourceResolver().resolve({"weights": source}, model_dir=tmp_path)


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _url_source(url="https://example.test/weights.bin", sha256=None):
    source = {"type": "url", "url": url}
    if sha256 is not None:
        source["sha256"] = sha256
    return _SOURCE_ADAPTER.validate_python(source)


def _fake_download(monkeypatch, tmp_path, *contents):
    """Serve `contents` in order, one per download (the last one repeats)."""
    downloads = []

    def download(url, path):
        downloads.append(url)
        path.write_bytes(contents[min(len(downloads), len(contents)) - 1])

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    return downloads


def _cached_urls(tmp_path):
    """Every file left in the URL cache, temporary download files included."""
    cache = tmp_path / "vhmodels-sources" / "url"
    return [path for path in cache.rglob("*") if path.is_file()]


def test_resolve_url_retries_after_interrupted_download(monkeypatch, tmp_path):
    attempts = []

    def download(url, path):
        attempts.append(path)
        path.write_bytes(b"partial")
        if len(attempts) == 1:
            raise OSError("download interrupted")
        path.write_bytes(b"complete")

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlretrieve", download)

    with pytest.raises(OSError, match="download interrupted"):
        SourceResolver()._resolve_url(_url_source())

    assert _cached_urls(tmp_path) == []

    resolved = SourceResolver()._resolve_url(_url_source())

    assert Path(resolved["path"]).name == "weights.bin"
    assert Path(resolved["path"]).read_bytes() == b"complete"
    assert _cached_urls(tmp_path) == [Path(resolved["path"])]
    assert len(attempts) == 2
    assert all(path != Path(resolved["path"]) for path in attempts)


def test_resolve_url_reuses_downloaded_file(monkeypatch, tmp_path):
    downloads = _fake_download(monkeypatch, tmp_path, b"complete")
    source = _url_source(sha256=_sha256(b"complete"))

    first = SourceResolver()._resolve_url(source)
    second = SourceResolver()._resolve_url(source)

    assert Path(first["path"]) == Path(second["path"])
    assert Path(first["path"]).name == "weights.bin"
    assert len(downloads) == 1


def test_resolve_url_keys_cache_on_url_not_just_filename(monkeypatch, tmp_path):
    downloads = _fake_download(monkeypatch, tmp_path, b"v1", b"v2")

    first = SourceResolver()._resolve_url(
        _url_source("https://example.test/v1/weights.bin")
    )
    second = SourceResolver()._resolve_url(
        _url_source("https://example.test/v2/weights.bin")
    )

    assert Path(first["path"]) != Path(second["path"])
    assert (Path(first["path"]).read_bytes(), Path(second["path"]).read_bytes()) == (
        b"v1",
        b"v2",
    )
    assert len(downloads) == 2


def test_resolve_url_downloads_again_when_pinned_sha256_changes(monkeypatch, tmp_path):
    downloads = _fake_download(monkeypatch, tmp_path, b"v1", b"v2")

    old = SourceResolver()._resolve_url(_url_source(sha256=_sha256(b"v1")))
    new = SourceResolver()._resolve_url(_url_source(sha256=_sha256(b"v2")))

    assert Path(old["path"]) != Path(new["path"])
    assert (Path(old["path"]).read_bytes(), Path(new["path"]).read_bytes()) == (
        b"v1",
        b"v2",
    )
    assert len(downloads) == 2


def test_resolve_url_downloads_corrupted_cached_file_again(monkeypatch, tmp_path):
    downloads = _fake_download(monkeypatch, tmp_path, b"complete")
    source = _url_source(sha256=_sha256(b"complete"))

    first = SourceResolver()._resolve_url(source)
    Path(first["path"]).write_bytes(b"corrupted")
    second = SourceResolver()._resolve_url(source)

    assert Path(second["path"]) == Path(first["path"])
    assert Path(second["path"]).read_bytes() == b"complete"
    assert len(downloads) == 2


def test_resolve_url_does_not_cache_wrong_checksum(monkeypatch, tmp_path):
    _fake_download(monkeypatch, tmp_path, b"unexpected")
    source = _url_source(sha256=_sha256(b"complete"))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        SourceResolver()._resolve_url(source)

    assert _cached_urls(tmp_path) == []


def _git_source(revision="pinned-revision", url="https://example.test/repository.git"):
    return _SOURCE_ADAPTER.validate_python(
        {"type": "git", "url": url, "revision": revision}
    )


@pytest.mark.parametrize("revision", [{}, {"revision": ""}])
def test_git_source_requires_a_revision(revision):
    with pytest.raises(ValidationError):
        _SOURCE_ADAPTER.validate_python(
            {"type": "git", "url": "https://example.test/repository.git", **revision}
        )


def _fake_git(monkeypatch, tmp_path):
    """Record git invocations, materializing a clone for every `git clone`."""
    commands = []

    def run(command, check):
        commands.append(command)
        if command[1] == "clone":
            (Path(command[-1]) / ".git").mkdir()

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(subprocess, "run", run)
    return commands


def _subcommands(commands):
    """The subcommand of each recorded invocation; checkout runs `git -C <dir>`."""
    return [command[3] if command[1] == "-C" else command[1] for command in commands]


def test_resolve_git_reuses_clone_for_same_revision(monkeypatch, tmp_path):
    commands = _fake_git(monkeypatch, tmp_path)
    source = _git_source("pinned-revision")

    first = SourceResolver()._resolve_git(source)
    second = SourceResolver()._resolve_git(source)

    assert Path(first["path"]) == Path(second["path"])
    assert Path(first["path"]).parent == tmp_path / "vhmodels-sources" / "git"
    assert _subcommands(commands) == ["clone", "checkout"]


def test_resolve_git_caches_each_revision_separately(monkeypatch, tmp_path):
    commands = _fake_git(monkeypatch, tmp_path)

    old = SourceResolver()._resolve_git(_git_source("old-revision"))
    new = SourceResolver()._resolve_git(_git_source("new-revision"))

    assert Path(old["path"]) != Path(new["path"])
    assert _subcommands(commands) == ["clone", "checkout", "clone", "checkout"]
    assert commands[1][-1] == "old-revision"
    assert commands[3][-1] == "new-revision"


def test_resolve_git_keys_cache_on_url_not_just_repository_name(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path)

    first = SourceResolver()._resolve_git(
        _git_source(url="https://example.test/one/repository.git")
    )
    second = SourceResolver()._resolve_git(
        _git_source(url="https://example.test/two/repository.git")
    )

    assert Path(first["path"]) != Path(second["path"])


def test_resolve_git_does_not_cache_failed_checkout(monkeypatch, tmp_path):
    source = _git_source("requested-revision")
    checkout_attempts = 0
    clone_attempts = 0

    def run(command, check):
        nonlocal checkout_attempts, clone_attempts
        if command[1] == "clone":
            clone_attempts += 1
            (Path(command[-1]) / ".git").mkdir()
            return
        checkout_attempts += 1
        if checkout_attempts == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        SourceResolver()._resolve_git(source)

    cache = tmp_path / "vhmodels-sources" / "git"
    assert [path for path in cache.iterdir() if not path.name.startswith(".")] == []

    resolved = SourceResolver()._resolve_git(source)

    assert Path(resolved["path"]).is_dir()
    assert Path(resolved["path"]).parent == cache
    assert clone_attempts == 2
    assert checkout_attempts == 2


def test_resolve_mole_sources_end_to_end(fake_huggingface_hub):
    resolved = REGISTRY.resolve("mole")
    resources = SourceResolver().resolve(resolved.sources, resolved.model_dir)

    assert set(resources["weights"]["files"]) == {"config", "checkpoint"}
    assert resources["architecture"]["name"] == "mole_package"
