from vhmodels.vh_checker.base import BaseModel
from vhmodels.models.registry import REGISTRY
from vhmodels.models.source_resolver import SourceResolver

from hyformer.models.auto import AutoModel
from hyformer.utils import set_seed
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig

import json
import torch


class Hyformer(BaseModel):
    PROJECT = "hyformer"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self, model=None, seed=1337, **kwargs):
        """
        Downloads and loads the artifacts for the specified model. The available models are:
        - hyformer_molecules_50M
        - hyformer_peptides_34M
        - hyformer_peptides_34_MIC
        - hyformer_molecules_8M

        The function retrieves and prepares following files:
        - vocab.txt: vocabulary of the tokenizer
        - tokenizer_config.json: configuration of the tokenizer
        - model_config.json: configuration of the model
        - ckpt.pt: weights of the model

        For more information, check out (link to HF).

        Parameters
        --------
        model : torch.device or str, optional
            Name of the model

        device : str

        seed : int

        Returns
        -------
        None
        """
        manifest = REGISTRY.resolve(self.PROJECT, model)
        weights = SourceResolver().resolve(manifest.sources, manifest.model_dir)[
            "weights"
        ]

        set_seed(seed)

        self.device = torch.device(
            kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        vocab_path = weights.files["vocab"]
        tok_config_path = weights.files["tokenizer_config"]

        with open(tok_config_path, "r") as f:
            tok_config_data = json.load(f)

        # Inject the absolute path so the tokenizer finds vocab.txt in the cache
        tok_config_data["path_to_vocabulary"] = str(vocab_path)

        self.tokenizer = AutoTokenizer.from_config(
            TokenizerConfig.from_dict(tok_config_data)
        )

        self.model = AutoModel.from_config(
            ModelConfig.from_config_file(weights.files["model_config"])
        )

        self.model.load_pretrained(weights.files["checkpoint"])

        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, input):
        return input

    def embed(self, input, batch_size=128, **kwargs):
        """
        Creates embeddings for the provided input.
        The function expects the tokenizer and the model to be loaded already.

        Parameters
        ----------
        input : str
            Path to the raw data file containing molecular representations.

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

        featurizer = self.model.to_encoder(
            self.tokenizer, batch_size, self.device
        )  # batch_size=128
        embeddings = featurizer.encode(inputs)

        return {"output": embeddings.tolist()}

    def predict(self, input, **kwargs):
        raise NotImplementedError("Hyformer does not support predict().")

    def generate(self, input, **kwargs):
        pass


if __name__ == "__main__":
    model = Hyformer()
    model.load_model("hyformer_molecules_50M")
    results = model.embed(
        inputs=[
            "CCCOc1cccc(-c2nn(-c3ccccc3)cc2/C=C(/C#N)C2=[N+]c3ccccc3[N-]2)c1 O=C(c1ccccc1)c1cc([N+](=O)O)c(Sc2c([N+](=O)O)cc([N+](=O)O)cc2[N+](=O)O)cc1[N+](=O)O",
            "Nc1ncc(CN2CCC3(CC2)C[C@H](c2ccccc2)CN(C2CC2)C3)cn1 O=C(c1ccco1)N(Cc1ccccc1Cl)C[C@@H]1CC(c2ccc(Cl)o2)=NO1",
            "O=C(c1cccc(/N=C(\O)CCc2ccccc2)c1)[N+]1CCCCC1",
        ]
    )
    print(results)
