"""Lightweight checks for the H-Optimus-0 registry entry and example script."""

import runpy
from pathlib import Path

import pytest
import vhmodels

from vhmodels.models.registry import REGISTRY


_EXAMPLE_SCRIPT = Path(__file__).resolve().parents[1] / "test.py"
_EXAMPLE_TILE = _EXAMPLE_SCRIPT.parent / "example_data/H-Optimus-0/TUM-AACHEHDV.tif"


def test_registry_declares_tile_embedder_and_pinned_checkpoint():
    manifest = REGISTRY.resolve("hoptimus0")

    assert manifest.variant == "default"
    assert manifest.implementation.class_path == "HOptimus0.model.HOptimus0"
    assert manifest.implementation.capabilities == ["embed"]
    checkpoint = manifest.sources["checkpoint"]
    assert checkpoint.repo_id == "bioptimus/H-optimus-0"
    assert len(checkpoint.revision) == 40


@pytest.mark.parametrize("runtime", ["conda", "apptainer"])
def test_example_script_embeds_the_real_tiff(monkeypatch, tmp_path, runtime):
    assert _EXAMPLE_TILE.is_file()
    assert _EXAMPLE_TILE.suffix.lower() == ".tif"
    received = {}

    class FakeModel:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def embed(self, *, input, batch_size):
            received["input"] = input
            received["batch_size"] = batch_size
            return [[0.1, 0.2]]

    def fake_load_model(**kwargs):
        received["load_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(vhmodels, "load_model", fake_load_model)
    monkeypatch.setenv("VHMODELS_RUNTIME", runtime)
    monkeypatch.setenv("VHMODELS_IMAGE_PATH", "test-hoptimus0.sif")
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(_EXAMPLE_SCRIPT), run_name="__main__")

    assert received["input"] == "example_data/H-Optimus-0/TUM-AACHEHDV.tif"
    assert received["batch_size"] == 1
    assert received["load_kwargs"]["project"] == "hoptimus0"
    assert received["load_kwargs"]["runtime"] == runtime
    if runtime == "apptainer":
        assert received["load_kwargs"]["image_path"] == "test-hoptimus0.sif"
    else:
        assert "image_path" not in received["load_kwargs"]
    output = (tmp_path / "output" / "hoptimus0_embeddings.txt").read_text()
    assert "embedding1:" in output
    assert "[0.1, 0.2]" in output


def test_worker_decodes_example_tiff_to_correct_rgb_shape():
    """Runs in a model environment with the image dependencies installed."""
    pytest.importorskip("PIL.Image")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("timm")
    from vhmodels.models.HOptimus0.model import HOptimus0

    tile = HOptimus0._open_tile(_EXAMPLE_TILE)

    assert tile.mode == "RGB"
    assert tile.size == (224, 224)
