from .vhmodels import VHModel

class MolE(
    VHModel,
    name="MolE",
    description="MolE learns task-independent molecular representations of chemicals via Graph Isomorphism Networks (GINs)", 
    link="https://huggingface.co/virtual-human-chc/MolE"                       
):
    def __init__(self):
        ...
        
    def load_model(self, model=None, **kwargs):
        """
        Downloads and loads the necessary artifacts for the MolE model from HuggingFace. 

        The function retrieves and prepares the following files: 
        - config.yaml: Contains the transformer configuration.
        - model.pth: Contains the trained model weights. 
        - MolE-XGBoost-08.03.2024_14.20.pkl: Contains the XGBoost model used within MolE. 

        Returns 
        ------- 
        Type of the model 
            The loaded model 
        """
        ...
    
    def encode(self, inputs, **kwargs):
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
        ...
    
    def predict(self, inputs, **kwargs):
        ...
    
    def generate(self, num_samples, **kwargs):
        ...
    
if __name__=='__main__':
    ...
    
