from vhmodels.vh_checker.base import BaseModel
import torch
import yaml
import pickle
import pandas as pd
from huggingface_hub import hf_hub_download
from mole_package import ginet_concat, mole_antimicrobial_prediction, mole_representation, dataset_representation

class MolE(
    BaseModel,
    project="mole",
    description="MolE learns task-independent molecular representations of chemicals via Graph Isomorphism Networks (GINs)", 
    link="https://huggingface.co/virtual-human-chc/MolE"                       
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

        Returns 
        ------- 
        None
        """
        
        if model not in self.model_config:
            raise ValueError(
                f"Unknown model '{self.model}' in DinoBloom. Available models: {list(self.model_config.keys())}"
            )

        # Download + load
        cfg = yaml.safe_load(open(hf_hub_download(self.repo, "config.yaml")))
        self.model = ginet_concat.GINet(**cfg["model"]).to(self.device)
        self.model.load_state_dict(torch.load(hf_hub_download(self.repo, "model.pth"), map_location=self.device))
        self.xgb = pickle.load(open(hf_hub_download(self.repo, "MolE-XGBoost-08.03.2024_14.20.pkl"), "rb"))
    
    def transform(self, data, **kwargs):
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
        pandas.core.frame.DataFrame 
            A DataFrame containing the generated embeddings. 
            - The index corresponds to "chem_name". 
            - The columns correspond to the 1000 embedding dimensions. 

        Example
        -------

                         0          1          2      ...      997        998        999
        -------------------------------------------------------------------------------
        Halicin        10.230212  10.007570  0.120784  ...  -1.998857 -20.055006  16.564339
        Abaucin        26.006750  42.643391  2.415676  ...  -5.441332 -54.594162   7.772403
        Diacerein      16.235847  27.619564  1.328622  ...  -4.330855 -43.452499  22.268299
        Tannic acid    49.815845 183.626511  3.261163  ... -19.322300-193.864944 146.213165
        Elivitegravir  24.466951  49.919296  2.536460  ...  -5.996571 -60.164993  17.515989
        Opicapone      16.270607  30.276501  0.966270  ...  -4.108759 -41.224167  18.833054
        Ebastine       36.845181  69.056267  4.710568  ...  -8.217525 -82.448318   7.520126
        """ 
        smiles_tsv = data
        smiles_df = mole_representation.read_smiles(smiles_tsv, "smiles", "chem_name")
        emb = dataset_representation.batch_representation(smiles_df, self.model, "smiles", "chem_name", device=self.device)
        X_input = mole_antimicrobial_prediction.add_strains(
            emb, "input/maier_screening_results.tsv.gz"
        )
        probs = self.xgb.predict_proba(X_input)[:, 1]
        return pd.DataFrame(
            {"antimicrobial_predictive_probability": probs},
            index=X_input.index
        )

if __name__=='__main__':
    ...
    
