# The file contains (basic) tests for DinoBloom
import pytest
import vhmodels
from unittest.mock import patch


@pytest.mark.integration
def test_dinobloom_inference():
    model = vhmodels.load_model(project="dinobloom", model="s")
    result = model.embed(input="example_data/DinoBloom/001.bmp")
    assert result is not None


@pytest.mark.integration
def test_transform_output_format():
    model = vhmodels.load_model(project="dinobloom", model="s")
    result = model.embed(input="example_data/DinoBloom/001.bmp")
    assert isinstance(result, (dict, list))


@pytest.mark.integration
def test_invalid_input_file():
    import vhmodels

    model = vhmodels.load_model(project="dinobloom", model="s")
    with pytest.raises(Exception):
        model.embed(input="non_existing_file.bmp")


def test_transform_subprocess_mock():
    fake_model_result = {"output": {"prediction": 42}}
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
        manager_class.return_value.embed.return_value = fake_model_result
        model = vhmodels.load_model(project="dinobloom", model="s")
        result = model.embed(input="dummy")

        assert result == {"prediction": 42}
        assert result != fake_model_result
