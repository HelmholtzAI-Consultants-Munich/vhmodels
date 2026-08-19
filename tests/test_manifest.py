"""Tests for the manifest schema, Registry, and SourceResolver.

See docs/manifest.md for the design this exercises: model.json +
manifests/<variant>.json -> Registry.resolve() -> ResolvedManifest ->
SourceResolver -> local resources.
"""

import subprocess
import sys
import types

import pytest
from pydantic import TypeAdapter, ValidationError

from vhmodels.models import discovery
from vhmodels.models.registry import REGISTRY, Registry
from vhmodels.models.schema import (
    HuggingFaceSource,
    ModelManifest,
    Source,
    TorchHubSource,
    VariantManifest,
)
from vhmodels.models.source_resolver import SourceResolver

_SOURCE_ADAPTER = TypeAdapter(Source)

_REAL_MODELS = ["dinobloom", "hyformer", "mole", "prottrans"]


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


def test_discover_finds_real_models():
    discovered = discovery.discover()
    assert set(_REAL_MODELS) <= set(discovered)
    directory, raw = discovered["dinobloom"]
    assert directory.name == "DinoBloom"
    assert raw["implementation"]["class_path"] == "DinoBloom.model.DinoBloom"


def test_find_class_path_for_real_models():
    assert discovery.find_class_path("mole") == "MolE.model.MolE"


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


def test_has_model_and_unknown_project_raises():
    assert REGISTRY.has_model("dinobloom") is True
    assert REGISTRY.has_model("does-not-exist") is False
    with pytest.raises(ValueError, match="not registered"):
        REGISTRY.get_model("does-not-exist")


def test_list_variants_matches_manifests_directory():
    assert REGISTRY.list_variants("dinobloom") == ["b", "g", "l", "s"]
    assert REGISTRY.list_variants("mole") == ["default"]
    assert len(REGISTRY.list_variants("prottrans")) == 10


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
    assert weights.repo_id == "virtual-human-chc/DinoBloom"
    assert str(weights.files["checkpoint"]).endswith("pytorch_model_b.bin")
    assert fake_huggingface_hub == [
        ("virtual-human-chc/DinoBloom", "pytorch_model_b.bin", None)
    ]


def test_resolve_huggingface_without_files_skips_download(fake_huggingface_hub):
    # ProtTrans sources declare no "files" -- transformers.from_pretrained()
    # resolves the repo itself, so SourceResolver must not touch the network.
    resolved = REGISTRY.resolve("prottrans", "prot_bert")
    resources = SourceResolver().resolve(resolved.sources, resolved.model_dir)

    assert resources["tokenizer"].files == {}
    assert resources["tokenizer"].repo_id == "virtual-human-chc/prot_bert"
    assert fake_huggingface_hub == []


def test_resolve_torch_hub_passes_through_without_downloading(fake_huggingface_hub):
    resolved = REGISTRY.resolve("dinobloom", "s")
    resources = SourceResolver().resolve(resolved.sources, resolved.model_dir)

    architecture = resources["architecture"]
    assert architecture.repo == "facebookresearch/dinov2"
    assert architecture.entrypoint == "dinov2_vits14"


def test_resolve_python_package_found():
    resolved = _SOURCE_ADAPTER.validate_python({"type": "python_package", "name": "os"})
    result = SourceResolver()._resolve_python_package(resolved)
    assert result.name == "os"


def test_resolve_python_package_missing_raises():
    resolved = _SOURCE_ADAPTER.validate_python(
        {"type": "python_package", "name": "definitely_not_a_real_package"}
    )
    with pytest.raises(ModuleNotFoundError):
        SourceResolver()._resolve_python_package(resolved)


def test_resolve_local_source_relative_to_model_dir(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"data")
    source = _SOURCE_ADAPTER.validate_python({"type": "local", "path": "weights.bin"})

    resolved = SourceResolver().resolve({"weights": source}, model_dir=tmp_path)

    assert resolved["weights"].path == tmp_path / "weights.bin"


def test_resolve_local_source_missing_raises(tmp_path):
    source = _SOURCE_ADAPTER.validate_python({"type": "local", "path": "missing.bin"})
    with pytest.raises(FileNotFoundError):
        SourceResolver().resolve({"weights": source}, model_dir=tmp_path)


def test_resolve_mole_sources_end_to_end(fake_huggingface_hub, monkeypatch):
    # mole_package is only installed in MolE's own runtime environment, not
    # on the host running this test suite -- fake its presence.
    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "mole_package" else None,
    )

    resolved = REGISTRY.resolve("mole")
    resources = SourceResolver().resolve(resolved.sources, resolved.model_dir)

    assert set(resources["weights"].files) == {"config", "checkpoint"}
    assert resources["architecture"].name == "mole_package"
