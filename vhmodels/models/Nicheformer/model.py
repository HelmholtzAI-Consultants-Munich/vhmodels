import anndata as ad
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from vhmodels.models.registry import REGISTRY
from vhmodels.models.source_resolver import SourceResolver
from vhmodels.utils.device import resolve_torch_device
from vhmodels.vh_checker.base import BaseModel


class Nicheformer(BaseModel):
    PROJECT = "nicheformer"
    DEFAULT_BATCH_SIZE = 32

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self, model=None, **kwargs):
        """Load the Nicheformer model and tokenizer from Hugging Face."""
        manifest = REGISTRY.resolve(self.PROJECT, model or "default")
        weights = SourceResolver().resolve(
            manifest.sources, manifest.model_dir
        )["weights"]

        self.device = resolve_torch_device(torch, kwargs.get("device", "auto"))
        self.model = AutoModelForMaskedLM.from_pretrained(
            weights.repo_id, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            weights.repo_id, trust_remote_code=True
        )
        self.model.eval()

    def embed(self, input, **kwargs):
        """Create cell embeddings from technology-mean and AnnData paths.

        Parameters
        ----------
        input : dict
            Mapping with ``technology_mean`` (a ``.npy`` path) and ``data``
            (an ``.h5ad`` path).
        batch_size : int, optional
            Number of cells tokenized and embedded at once. Defaults to 32.
        max_cells : int, optional
            Limit processing to the first cells in the dataset. Good for smoke tests. Defaults to None (process all cells).

        Returns
        -------
        dict
            A dictionary containing ``output`` as a list of cell embeddings.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        batch_size = kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")

        max_cells = kwargs.get("max_cells")
        if max_cells is not None and (
            not isinstance(max_cells, int)
            or isinstance(max_cells, bool)
            or max_cells <= 0
        ):
            raise ValueError("max_cells must be a positive integer.")

        technology_mean = np.load(input["technology_mean"])
        self.tokenizer._load_technology_mean(technology_mean)

        adata = ad.read_h5ad(input["data"], backed="r")
        embeddings = []
        try:
            cell_count = adata.n_obs
            if max_cells is not None:
                cell_count = min(cell_count, max_cells)

            for start in range(0, cell_count, batch_size):
                stop = min(start + batch_size, cell_count)
                batch_adata = adata[start:stop].to_memory()
                tokenized = self.tokenizer(batch_adata)
                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)

                with torch.no_grad():
                    batch_embeddings = self.model.get_embeddings(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        layer=-1,
                        with_context=False,
                    )

                embeddings.extend(batch_embeddings.cpu().tolist())
        finally:
            adata.file.close()

        return {"output": embeddings}

    def predict(self, input, **kwargs):
        raise NotImplementedError("Nicheformer does not support predict().")

    def generate(self, input, **kwargs):
        pass
