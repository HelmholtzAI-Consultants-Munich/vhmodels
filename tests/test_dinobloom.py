# The file contains (basic) tests for DinoBloom
import pytest
import vhmodels
from unittest.mock import patch
import json


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
    from vhmodels.vh_checker.protocol import RESULT_MARKER

    # The model returns its own envelope {"output": ...}; the child emits it
    # verbatim framed by RESULT_MARKER, and the parent unwraps a single "output".
    fake_model_result = {"output": {"prediction": 42}}
    framed_stdout = f"{RESULT_MARKER}{json.dumps(fake_model_result)}{RESULT_MARKER}\n"

    with patch(
        "vhmodels.vh_checker.backends.CondaBackend.is_available", return_value=True
    ):
        with patch(
            "vhmodels.vh_checker.factory._run_subprocess",
            return_value=(framed_stdout, ""),
        ):
            model = vhmodels.load_model(project="dinobloom", model="s")
            result = model.embed(input="dummy")

            # Single unwrap, not double-wrapped.
            assert result == {"prediction": 42}
            assert result != fake_model_result
