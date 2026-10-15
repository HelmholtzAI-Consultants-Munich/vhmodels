from vhmodels.vh_checker.base import BaseModel
from vhmodels.utils.device import resolve_torch_device

from pathlib import Path

import json
import torch


class Hyformer(BaseModel):
    PROJECT = "hyformer"

    def __init__(self):
        # The molecule checkpoints were trained with Hyformer v1 (package
        # ``hyformer``), the peptide checkpoints with Hyformer v2 (package
        # ``hyformer_v2``). The two use different config schemas, tokenizers
        # and attention layouts, so each variant is routed to its own loader.
        self.model_specs = {
            "hyformer_molecules_8M": self._load_v1,
            "hyformer_molecules_50M": self._load_v1,
            "hyformer_peptides_34M": self._load_v2,
            "hyformer_peptides_34M_MIC": self._load_v2,
        }

        self.model = None
        self.tokenizer = None
        self.device = None
        self.variant = None

    def load_model(self, model, resources, seed=1337, **kwargs):
        """
        Downloads and loads the artifacts for the specified model. The available models are:
        - hyformer_molecules_8M
        - hyformer_molecules_50M
        - hyformer_peptides_34M
        - hyformer_peptides_34M_MIC

        The function retrieves and prepares following files:
        - vocab.txt: vocabulary of the tokenizer
        - tokenizer_config.json: configuration of the tokenizer
        - model_config.json: configuration of the model
        - ckpt.pt: weights of the model

        For more information, check out https://huggingface.co/collections/virtual-human-chc/hyformer.

        Parameters
        --------
        model : str
            Name of the model

        device : str

        seed : int

        Returns
        -------
        None
        """
        if model not in self.model_specs:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Supported models: {', '.join(self.model_specs.keys())}"
            )
        self.variant = model

        weights = resources["weights"]

        self.device = resolve_torch_device(torch, kwargs.get("device", "auto"))

        self.tokenizer, self.model = self.model_specs[model](weights, seed)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_v1(weights, seed):
        from hyformer.models.auto import AutoModel
        from hyformer.utils import set_seed
        from hyformer.utils.tokenizers.auto import AutoTokenizer
        from hyformer.configs.tokenizer import TokenizerConfig
        from hyformer.configs.model import ModelConfig

        set_seed(seed)

        with open(weights["files"]["tokenizer_config"], "r") as f:
            tok_config_data = json.load(f)

        # Inject the absolute path so the tokenizer finds vocab.txt in the cache
        tok_config_data["path_to_vocabulary"] = str(weights["files"]["vocab"])

        tokenizer = AutoTokenizer.from_config(TokenizerConfig.from_dict(tok_config_data))

        model = AutoModel.from_config(
            ModelConfig.from_config_file(weights["files"]["model_config"])
        )
        model.load_pretrained(weights["files"]["checkpoint"])

        return tokenizer, model

    @staticmethod
    def _load_v2(weights, seed):
        from hyformer_v2.models.auto import AutoModel
        from hyformer_v2.utils.reproducibility import set_seed
        from hyformer_v2.tokenizers.auto import AutoTokenizer
        from hyformer_v2.configs.tokenizer import TokenizerConfig

        set_seed(seed)

        with open(weights["files"]["tokenizer_config"], "r") as f:
            tok_config_data = json.load(f)

        # Inject the absolute path so the tokenizer finds vocab.txt in the cache
        tok_config_data["vocabulary_path"] = str(weights["files"]["vocab"])

        tokenizer = AutoTokenizer.from_config(TokenizerConfig(**tok_config_data))

        # from_pretrained reads model_config.json and ckpt.pt from the local
        # snapshot directory the resolver already downloaded.
        model = AutoModel.from_pretrained(
            str(Path(weights["files"]["checkpoint"]).parent)
        )

        return tokenizer, model

    def _preprocess(self, input):
        return input

    def embed(self, input, batch_size=128, **kwargs):
        """
        Creates embeddings for the provided input.
        The function expects the tokenizer and the model to be loaded already.

        Parameters
        ----------
        input : list of str
            SMILES strings for the molecule models, amino acid sequences for
            the peptide models.

        batch_size : int
            Size of the batch

        Returns
        -------
        dict
            A dictionary containing:
            - 'output': list of lists
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self._preprocess(input)

        # Pass the device as a string: Hyformer v1 skips pinned memory only
        # when ``device == "cpu"``, which is False for ``torch.device("cpu")``.
        featurizer = self.model.to_encoder(
            self.tokenizer, batch_size, str(self.device)
        )
        embeddings = featurizer.encode(inputs)

        return {"output": embeddings.tolist()}

    def predict(self, input, **kwargs):
        raise NotImplementedError("Hyformer does not support predict().")

    def generate(self, input, **kwargs):
        pass
