import importlib
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
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


def test_load_and_embed_follow_model_card_flow():
    module = _import_nicheformer_with_fake_dependencies()
    loaded_model = MagicMock()
    loaded_model.to.return_value = loaded_model
    tokenizer = MagicMock()
    module.AutoModelForMaskedLM.from_pretrained.return_value = loaded_model
    module.AutoTokenizer.from_pretrained.return_value = tokenizer

    adata = MagicMock()
    adata.n_obs = 1
    batch_adata = MagicMock()
    adata.__getitem__.return_value.to_memory.return_value = batch_adata
    input_ids = MagicMock()
    attention_mask = MagicMock()
    tokenizer.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    embeddings = MagicMock()
    embeddings.cpu.return_value.tolist.return_value = [[0.1, 0.2]]
    loaded_model.get_embeddings.return_value = embeddings

    manifest = SimpleNamespace(sources={}, model_dir="/model")
    resources = {
        "weights": SimpleNamespace(repo_id="virtual-human-chc/Nicheformer")
    }

    try:
        with (
            patch.object(module.REGISTRY, "resolve", return_value=manifest) as resolve,
            patch.object(module, "SourceResolver") as resolver,
            patch.object(module, "resolve_torch_device", return_value="cpu"),
            patch.object(module.np, "load", return_value="technology mean") as load,
            patch.object(module.ad, "read_h5ad", return_value=adata) as read_h5ad,
        ):
            resolver.return_value.resolve.return_value = resources

            model = module.Nicheformer()
            model.load_model()
            result = model.embed(
                {
                    "technology_mean": "technology_mean.npy",
                    "data": "cells.h5ad",
                }
            )

        resolve.assert_called_once_with("nicheformer", "default")
        module.AutoModelForMaskedLM.from_pretrained.assert_called_once_with(
            "virtual-human-chc/Nicheformer", trust_remote_code=True
        )
        module.AutoTokenizer.from_pretrained.assert_called_once_with(
            "virtual-human-chc/Nicheformer", trust_remote_code=True
        )
        loaded_model.to.assert_called_once_with("cpu")
        loaded_model.eval.assert_called_once_with()
        load.assert_called_once_with("technology_mean.npy")
        tokenizer._load_technology_mean.assert_called_once_with("technology mean")
        read_h5ad.assert_called_once_with("cells.h5ad", backed="r")
        adata.__getitem__.assert_called_once_with(slice(0, 1))
        adata.__getitem__.return_value.to_memory.assert_called_once_with()
        tokenizer.assert_called_once_with(batch_adata)
        input_ids.to.assert_called_once_with("cpu")
        attention_mask.to.assert_called_once_with("cpu")
        loaded_model.get_embeddings.assert_called_once_with(
            input_ids=input_ids.to.return_value,
            attention_mask=attention_mask.to.return_value,
            layer=-1,
            with_context=False,
        )
        adata.file.close.assert_called_once_with()
        assert result == {"output": [[0.1, 0.2]]}
    finally:
        sys.modules.pop("vhmodels.models.Nicheformer.model", None)


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
