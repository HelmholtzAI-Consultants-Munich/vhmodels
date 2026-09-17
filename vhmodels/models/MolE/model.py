from vhmodels.vh_checker.base import BaseModel
from vhmodels.models.registry import REGISTRY
from vhmodels.models.source_resolver import SourceResolver
from vhmodels.utils.device import resolve_torch_device

import pickle

import pandas as pd
import torch
import yaml
from mole_package import (
    dataset_representation,
    ginet_concat,
    mole_antimicrobial_prediction,
    mole_representation,
)


class MolE(BaseModel):
    PROJECT = "mole"

    def __init__(self):
        self.model = None
        self.xgb = None
        self.screening = None
        self.device = None

    @staticmethod
    def _read_molecules(input):
        """Read and validate the named SMILES table used by MolE."""
        molecules = pd.read_csv(input, sep="\t")
        required_columns = ["chem_name", "smiles"]
        missing_columns = set(required_columns).difference(molecules.columns)
        if missing_columns:
            raise ValueError(
                "MolE input is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )
        if molecules[required_columns].isna().any().any():
            raise ValueError("MolE input contains a missing chemical name or SMILES.")
        if molecules["chem_name"].duplicated().any():
            raise ValueError("MolE input contains duplicate chemical names.")

        valid_molecules = mole_representation.read_smiles_df(
            input, smile_col="smiles", id_col="chem_name"
        )
        if len(valid_molecules) != len(molecules):
            raise ValueError("MolE input contains an invalid SMILES value.")
        return valid_molecules

    def _embed_molecules(self, molecules):
        """Generate embedding vectors for a validated molecule table."""
        embedding = dataset_representation.batch_representation(
            smiles_list=molecules["smiles"].tolist(),
            dl_model=self.model,
            device=self.device,
        )
        return embedding.tolist()

    def load_model(self, model=None, **kwargs):
        """
        Downloads and loads the necessary artifacts for the MolE model from HuggingFace.

        The function retrieves and prepares the following files:
        - config.yaml: Contains the transformer configuration.
        - model.pth: Contains the model weights.

        device : torch.device or str, optional

        Returns
        -------
        None
        """
        manifest = REGISTRY.resolve(self.PROJECT, model or "default")
        resources = SourceResolver().resolve(manifest.sources, manifest.model_dir)

        self.device = resolve_torch_device(torch, kwargs.get("device", "auto"))

        weights = resources["weights"].files
        cfg = yaml.safe_load(open(weights["config"]))
        self.model = ginet_concat.GINet(**cfg["model"]).to(self.device)
        self.model.load_state_dict(
            torch.load(weights["checkpoint"], map_location=self.device)
        )

        xgb = resources["xgb"].files
        self.xgb = pickle.load(open(xgb["model"], "rb"))
        self.screening = xgb["screening"]

    def embed(self, input, **kwargs):
        """
        Creates embeddings for the provided input data.

        The model is expected to be already loaded before calling this function.

        Parameters
        ----------
        input : str
            Path to a tab-separated file with ``chem_name`` and ``smiles``
            columns.

        Returns
        -------
        dict
            A dictionary whose ``output`` maps each chemical name to its
            embedding. The default model produces 1000 values per chemical.
        """
        molecules = self._read_molecules(input)
        embedding = self._embed_molecules(molecules)
        return {
            "output": dict(zip(molecules["chem_name"].tolist(), embedding))
        }

    def predict(self, input, embedding=None, **kwargs):
        """Predict antimicrobial activity for the chemicals in ``input``.

        When ``embedding`` is omitted, it is generated from ``input``. A
        supplied embedding must contain the same chemical names in the same
        order as the input table.
        """
        molecules = self._read_molecules(input)
        chem_names = molecules["chem_name"].tolist()
        if embedding is None:
            emb_df = pd.DataFrame(
                self._embed_molecules(molecules), index=chem_names
            )
        else:
            if not isinstance(embedding, dict):
                raise ValueError("MolE embedding must map chemical names to vectors.")
            if list(embedding) != chem_names:
                raise ValueError(
                    "MolE embedding chemical names and order must match the input."
                )
            emb_df = pd.DataFrame.from_dict(embedding, orient="index")

        X = mole_antimicrobial_prediction.add_strains(emb_df, self.screening)
        probs = self.xgb.predict_proba(X)[:, 1]
        return {"output": pd.Series(probs, index=X.index).to_dict()}

    def generate(self, input, **kwargs):
        pass


if __name__ == "__main__":
    model = MolE()
    model.load_model()
    result = model.predict("example_data/MolE/examples_molecules.tsv")
    print(result)
