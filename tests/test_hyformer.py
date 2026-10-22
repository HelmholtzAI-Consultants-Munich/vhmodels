# The file contains (basic) tests for Hyformer
from pathlib import Path

import pytest
import vhmodels

from vhmodels.models.registry import REGISTRY


_DATA_DIR = Path(__file__).resolve().parents[1] / "example_data/Hyformer"
_MOLECULES = (_DATA_DIR / "molecules.smi").read_text().splitlines()
_PEPTIDES = (_DATA_DIR / "peptides.txt").read_text().splitlines()

# variant -> (inputs, embedding dim)
_VARIANTS = {
    "hyformer_molecules_8M": (_MOLECULES, 256),
    "hyformer_molecules_50M": (_MOLECULES, 512),
    "hyformer_peptides_34M": (_PEPTIDES, 512),
    "hyformer_peptides_34M_MIC": (_PEPTIDES, 512),
}


def test_registry_declares_all_variants():
    assert REGISTRY.list_variants("hyformer") == sorted(_VARIANTS)
    for variant in _VARIANTS:
        manifest = REGISTRY.resolve("hyformer", variant)
        assert manifest.sources["weights"].repo_id == f"virtual-human-chc/{variant}"


@pytest.mark.integration
@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_hyformer_embed_inference(variant):
    inputs, dim = _VARIANTS[variant]
    with vhmodels.load_model(project="hyformer", model=variant) as model:
        result = model.embed(input=inputs, batch_size=2)
    assert len(result) == len(inputs)
    assert all(len(row) == dim for row in result)
