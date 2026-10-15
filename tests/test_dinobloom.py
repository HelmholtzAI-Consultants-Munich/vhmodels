# The file contains (basic) tests for DinoBloom
import pytest
import vhmodels


@pytest.mark.integration
def test_dinobloom_inference():
    with vhmodels.load_model(project="dinobloom", model="s", runtime="ray") as model:
        result = model.embed(input="example_data/DinoBloom/001.bmp")
        assert len(result) == 1
        assert len(result[0]) == 384

        with pytest.raises(Exception):
            model.embed(input="non_existing_file.bmp")
