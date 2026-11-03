# The file contains (basic) tests for ProtTrans
import pytest
import vhmodels


_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ"


def _embed(variant):
    with vhmodels.load_model(
        project="prottrans", model=variant, runtime="ray"
    ) as model:
        return model.embed(input=[_SEQUENCE])


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["prot_t5_xl_uniref50", "prot_t5_xxl_uniref50"])
def test_t5_embeddings_are_reproducible_across_loads(variant):
    assert _embed(variant) == _embed(variant)
