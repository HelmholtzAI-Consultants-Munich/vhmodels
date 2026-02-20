from vhmodels import VHModel
import torch
import yaml
import pickle
import pandas as pd
from huggingface_hub import hf_hub_download
from mole_package import ginet_concat, mole_antimicrobial_prediction, mole_representation, dataset_representation
import pprint

class MolE(
    VHModel,
    name="MolE",
    description="MolE learns task-independent molecular representations of chemicals via Graph Isomorphism Networks (GINs)", 
    link="https://huggingface.co/virtual-human-chc/MolE"                       
):
    capabilities = {"predict"}

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _download(self, repo_id, filename):
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=repo_id)
    
    def load_model(self, repo=None, **kwargs):
        # Download all necessary files using the helper
        config_path = self._download(repo, "config.yaml")
        model_path = self._download(repo, "model.pth")
        xgb_path = self._download(repo, "MolE-XGBoost-08.03.2024_14.20.pkl")

        # Load config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Initialize model
        self.model = ginet_concat.GINet(**cfg["model"]).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load XGBoost model
        with open(xgb_path, "rb") as f:
            self.xgb = pickle.load(f)
    
    def encode(self, inputs, **kwargs):
        smiles_df = mole_representation.read_smiles(inputs, "smiles", "chem_name")
        emb = dataset_representation.batch_representation(smiles_df, self.model, "smiles", "chem_name", device=self.device)
        return emb
    
    def predict(self, inputs, **kwargs):
        smiles_df = mole_representation.read_smiles(inputs, "smiles", "chem_name")
        emb = dataset_representation.batch_representation(smiles_df, self.model, "smiles", "chem_name", device=self.device)
        X_input = mole_antimicrobial_prediction.add_strains(
            emb, "input\mole\maier_screening_results.tsv.gz"
        )
        probs = self.xgb.predict_proba(X_input)[:, 1]
        return pd.DataFrame(
            {"antimicrobial_predictive_probability": probs},
            index=X_input.index
        )
    
    def generate(self, num_samples, **kwargs):
        ...
    
if __name__=='__main__':
    #...
    mole = MolE()
    mole.load_model("virtual-human-chc/MolE")
    print(mole.encode("input\mole\examples_molecules.tsv"))
    