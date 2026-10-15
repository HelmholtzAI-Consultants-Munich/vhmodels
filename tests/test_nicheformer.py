import importlib
import sys
import types
from contextlib import nullcontext
from unittest.mock import MagicMock, call, patch

import pytest


def _import_nicheformer_with_fake_dependencies():
    fake_anndata = types.ModuleType("anndata")
    fake_numpy = types.ModuleType("numpy")
    fake_torch = types.ModuleType("torch")
    fake_transformers = types.ModuleType("transformers")

    fake_anndata.read_h5ad = MagicMock()
    fake_numpy.load = MagicMock()
    fake_torch.no_grad = MagicMock(return_value=nullcontext())
    fake_transformers.AutoModelForMaskedLM = MagicMock()
    fake_transformers.AutoTokenizer = MagicMock()

    dependencies = {
        "anndata": fake_anndata,
        "numpy": fake_numpy,
        "torch": fake_torch,
        "transformers": fake_transformers,
    }
    with patch.dict(sys.modules, dependencies):
        sys.modules.pop("vhmodels.models.Nicheformer.model", None)
        module = importlib.import_module("vhmodels.models.Nicheformer.model")

    return module


def test_embed_requires_loaded_model():
    module = _import_nicheformer_with_fake_dependencies()
    try:
        with pytest.raises(RuntimeError, match="Model not loaded"):
            module.Nicheformer().embed(
                {
                    "technology_mean": "technology_mean.npy",
                    "data": "cells.h5ad",
                }
            )
    finally:
        sys.modules.pop("vhmodels.models.Nicheformer.model", None)


def test_embed_batches_cells_and_honors_max_cells():
    module = _import_nicheformer_with_fake_dependencies()
    try:
        model = module.Nicheformer()
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        model.device = "cpu"

        adata = MagicMock()
        adata.n_obs = 10
        views = [MagicMock(), MagicMock()]
        batches = [MagicMock(), MagicMock()]
        for view, batch in zip(views, batches):
            view.to_memory.return_value = batch
        adata.__getitem__.side_effect = views

        tokenized = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        model.tokenizer.return_value = tokenized
        batch_outputs = [MagicMock(), MagicMock()]
        batch_outputs[0].cpu.return_value.tolist.return_value = [[1.0], [2.0]]
        batch_outputs[1].cpu.return_value.tolist.return_value = [[3.0]]
        model.model.get_embeddings.side_effect = batch_outputs

        with (
            patch.object(module.np, "load", return_value="technology mean"),
            patch.object(module.ad, "read_h5ad", return_value=adata),
        ):
            result = model.embed(
                {
                    "technology_mean": "technology_mean.npy",
                    "data": "cells.h5ad",
                },
                batch_size=2,
                max_cells=3,
            )

        assert adata.__getitem__.call_args_list == [
            call(slice(0, 2)),
            call(slice(2, 3)),
        ]
        assert model.tokenizer.call_args_list == [call(batches[0]), call(batches[1])]
        adata.file.close.assert_called_once_with()
        assert result == {"output": [[1.0], [2.0], [3.0]]}
    finally:
        sys.modules.pop("vhmodels.models.Nicheformer.model", None)


@pytest.mark.parametrize(
    ("name", "value"),
    [("batch_size", 0), ("batch_size", True), ("max_cells", -1)],
)
def test_embed_rejects_invalid_limits(name, value):
    module = _import_nicheformer_with_fake_dependencies()
    try:
        model = module.Nicheformer()
        model.model = MagicMock()
        model.tokenizer = MagicMock()

        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            model.embed(
                {
                    "technology_mean": "technology_mean.npy",
                    "data": "cells.h5ad",
                },
                **{name: value},
            )
    finally:
        sys.modules.pop("vhmodels.models.Nicheformer.model", None)
