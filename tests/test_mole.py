# The file contains (basic) tests for MolE
import pytest
import vhmodels

_MOLECULES = "example_data/MolE/examples_molecules.tsv"


@pytest.fixture(scope="module")
def mole():
    with vhmodels.load_model(project="mole", runtime="ray") as model:
        yield model


@pytest.mark.integration
def test_mole_embed_inference(mole):
    result = mole.embed(input=_MOLECULES)
    assert list(result) == [
        "Halicin",
        "Abaucin",
        "Diacerein",
        "Tannic acid",
        "Elivitegravir",
        "Opicapone",
        "Ebastine",
    ]
    assert all(len(row) == 1000 for row in result.values())


@pytest.mark.integration
def test_mole_predict_embeds_input(mole):
    result = mole.predict(input=_MOLECULES)
    assert isinstance(result, dict)
    assert all(0.0 <= probability <= 1.0 for probability in result.values())


@pytest.mark.integration
def test_mole_predict_from_embedding(mole):
    embedding = mole.embed(input=_MOLECULES)
    reordered_embedding = dict(reversed(embedding.items()))
    # Ray re-raises the model's exception as a subclass of its original type.
    with pytest.raises(ValueError, match="names and order must match"):
        mole.predict(input=_MOLECULES, embedding=reordered_embedding)
    result = mole.predict(input=_MOLECULES, embedding=embedding)
    assert isinstance(result, dict)
    assert all(0.0 <= probability <= 1.0 for probability in result.values())
