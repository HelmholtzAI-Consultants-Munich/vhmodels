from vhmodels.vh_checker.base import BaseModel
import torch
import yaml
from huggingface_hub import hf_hub_download
from mole_package import ginet_concat, mole_representation, dataset_representation


class MolE(BaseModel):
    def __init__(self):
        self.repo = "virtual-human-chc/MolE"
        self.model = None
        self.xgb = None
        self.device = None

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
        self.device = (
            "cuda:0" if self.device == "auto" and torch.cuda.is_available() else "cpu"
        )

        cfg = yaml.safe_load(open(hf_hub_download(self.repo, "config.yaml")))
        self.model = ginet_concat.GINet(**cfg["model"]).to(self.device)
        self.model.load_state_dict(
            torch.load(
                hf_hub_download(self.repo, "model.pth"), map_location=self.device
            )
        )

    def embed(self, input, **kwargs):
        """
        Creates embeddings for the provided input data.

        The model is expected to be already loaded before calling this function.

        Parameters
        ----------
        inputs : str
            Path to the raw data file containing molecular representations.

        Returns
        -------
        dict
            A dictionary containing:
            - 'output': list of lists
        """
        ## !! Refine the functions in the MolE package, so they don't return
        smiles = mole_representation.read_smiles(input)
        emb = dataset_representation.batch_representation(
            smiles_list=smiles, dl_model=self.model, device=self.device
        )
        return {"output": emb.tolist()}

    def predict(self, input, **kwargs):
        pass

    def generate(self, input, **kwargs):
        pass


if __name__ == "__main__":
    model = MolE()
    model.load_model()
    result = model.embed("example_data/MolE/sequences.smiles")
    print(result)
