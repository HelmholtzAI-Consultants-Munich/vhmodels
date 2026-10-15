"""Lightweight check for the H-Optimus-0 example script."""

import runpy
from pathlib import Path

import vhmodels


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_SCRIPT = _PROJECT_ROOT / "examples/hoptimus0.py"
_EXAMPLE_TILE = _PROJECT_ROOT / "example_data/H-Optimus-0/TUM-AACHEHDV.tif"


def test_example_script_passes_tiff_to_model(monkeypatch, tmp_path):
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
    monkeypatch.setenv("VHMODELS_RUNTIME", "apptainer")
    monkeypatch.setenv("VHMODELS_IMAGE_PATH", "test-hoptimus0.sif")
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(_EXAMPLE_SCRIPT), run_name="__main__")

    assert received["input"] == str(_EXAMPLE_TILE)
    assert received["batch_size"] == 1
    assert received["load_kwargs"]["project"] == "hoptimus0"
    assert received["load_kwargs"]["runtime"] == "apptainer"
    assert received["load_kwargs"]["image_path"] == "test-hoptimus0.sif"
    output = (tmp_path / "output" / "hoptimus0_embeddings.txt").read_text()
    assert "embedding1:" in output
    assert "[0.1, 0.2]" in output

