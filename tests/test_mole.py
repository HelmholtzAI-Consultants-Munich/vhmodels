# The file contains (basic) tests for MolE
import pytest
import vhmodels
from unittest.mock import patch


@pytest.mark.integration
def test_mole_embed_inference():
    model = vhmodels.load_model(project="mole")
    result = model.embed(input="example_data/MolE/sequences.smiles")
    assert result is not None


@pytest.mark.integration
def test_mole_predict_from_embedding():
    model = vhmodels.load_model(project="mole")
    embedding = model.embed(input="example_data/MolE/sequences.smiles")
    result = model.predict(
        input="example_data/MolE/examples_molecules.tsv", embedding=embedding
    )
    assert result is not None
    assert isinstance(result, dict)
    assert all(0.0 <= probability <= 1.0 for probability in result.values())


def test_predict_subprocess_mock():
    fake_embed_result = {"output": [[0.1, 0.2]]}
    fake_predict_result = {"output": {"Halicin:strain-a": 0.9}}
    with (
        patch("vhmodels.vh_checker.factory.CondaProcessManager") as manager_class,
        patch(
            "vhmodels.vh_checker.backends.CondaBackend.is_runtime_available",
            return_value=True,
        ),
        patch(
            "vhmodels.vh_checker.backends.CondaBackend.is_available",
            return_value=True,
        ),
    ):
        manager_class.return_value.embed.return_value = fake_embed_result
        manager_class.return_value.predict.return_value = fake_predict_result
        model = vhmodels.load_model(project="mole")
        embedding = model.embed(input="dummy.smiles")
        result = model.predict(input="dummy.tsv", embedding=embedding)

        assert embedding == [[0.1, 0.2]]
        assert result == {"Halicin:strain-a": 0.9}

        predict_kwargs = manager_class.return_value.predict.call_args.kwargs
        assert predict_kwargs["input"] == "dummy.tsv"
        assert predict_kwargs["embedding"] == [[0.1, 0.2]]
