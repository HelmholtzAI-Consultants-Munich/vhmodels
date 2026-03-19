from vhmodels.vh_checker.base import BaseModel
import torch
import yaml
import pickle
import pandas as pd
from huggingface_hub import hf_hub_download
from mole_package import ginet_concat, mole_antimicrobial_prediction, mole_representation, dataset_representation

class MolE(
    BaseModel,
    #project="mole",
    #description="MolE learns task-independent molecular representations of chemicals via Graph Isomorphism Networks (GINs)", 
    #link="https://huggingface.co/virtual-human-chc/MolE"                       
):
    
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
        self.device = "cuda:0" if self.device == "auto" and torch.cuda.is_available() else "cpu"

        cfg = yaml.safe_load(open(hf_hub_download(self.repo, "config.yaml")))
        self.model = ginet_concat.GINet(**cfg["model"]).to(self.device)
        self.model.load_state_dict(torch.load(hf_hub_download(self.repo, "model.pth"), map_location=self.device))
    
    def transform(self, input, **kwargs):
        """ 
        Creates embeddings for the provided input data. 

        The model is expected to be already loaded before calling this function. 

        Parameters 
        ---------- 
        inputs : str 
            Path to the raw data file containing molecular representations. 
            The file must contain two columns: 
            - "chem_name": Name of the molecule 
            - "smiles": SMILES representation of the molecule 

        Returns 
        ------- 
        dict
        """ 
        ## !! Refine the functions in the MolE package, so they don't return DF
        smiles_tsv = input
        smiles_df = mole_representation.read_smiles(smiles_tsv, "smiles", "chem_name")
        emb = dataset_representation.batch_representation(smiles_df, self.model, "smiles", "chem_name", device=self.device)
        return {'output': emb}

if __name__=='__main__':
    ...
    
