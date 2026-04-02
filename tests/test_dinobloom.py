# The file contains (basic) tests for DinoBloom
import pytest
import vhmodels
from unittest.mock import patch, MagicMock
import json


def test_dinobloom_inference():
    model = vhmodels.load_model(project="dinobloom", model="s")
    result = model.transform(input="example_data/DinoBloom/001.bmp")
    assert result is not None


def test_transform_output_format():
    model = vhmodels.load_model(project="dinobloom", model="s")
    result = model.transform(input="example_data/DinoBloom/001.bmp")
    assert isinstance(result, (dict, list))


def test_invalid_input_file():
    import vhmodels

    model = vhmodels.load_model(project="dinobloom", model="s")
    with pytest.raises(Exception):
        model.transform(input="non_existing_file.bmp")


def test_transform_subprocess_mock():
    fake_output = json.dumps({"output": {"prediction": 42}})

    mock_result = MagicMock()
    mock_result.stdout = fake_output

    with patch("vhmodels.vh_checker.factory.ModelProxy._env_exists", return_value=True):
        with patch("subprocess.run", return_value=mock_result):
            model = vhmodels.load_model(project="dinobloom", model="s")
            result = model.transform(input="dummy")

            assert result == {"prediction": 42}
