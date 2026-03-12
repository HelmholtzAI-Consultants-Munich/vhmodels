from vhmodels.vh_checker.base import BaseModel
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from hyformer.models.auto import AutoModel
from hyformer.models.base import Encoder
from hyformer.utils import set_seed
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.utils.tokenizers.base import BaseTokenizer

class Hyformer(
    BaseModel,
    project="hyformer",
    description="A joint transformer-based model that unifies a generative decoder with a predictive encoder", 
    link="https://huggingface.co/collections/virtual-human-chc/hyformer"                       
):
    
    SEED = 1337
    set_seed(SEED)
    def __init__(self):
        self.model_config = [
            "hyformer_molecules_50M",
            "hyformer_peptides_34M",
            "hyformer_peptides_34_MIC",
            "hyformer_molecules_8M"
        ]
        self.model = None
        self.tokenizer = None
        self.device = None
        self.local = None

    def _download(self, repo_id, filename):
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=self.local)

    def load_model(self, model=None, **kwargs):
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
        - downstream_config.json: configuration for the downstream prediction task 
        - ckpt.pt: weights of the model 

        For more information, check out (link to HF). 

        Parameters
        -------- 
        model: str 
            Name of the model 
        
        device: str

        
        Returns 
        ------- 
        None
        """
        if model not in self.model_config:
            raise ValueError(
                f"Unknown model '{self.model}' in DinoBloom. Available models: {list(self.model_config.keys())}"
            )
        
        self.device = kwargs.get(
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.local = Path(f"virtual-human-chc/{self.model}")

        self._download(f"virtual-human-chc/{model}", "vocab.txt") 

        self.tokenizer = AutoTokenizer.from_config(
            TokenizerConfig.from_config_file(self._download(f"virtual-human-chc/{model}", "tokenizer_config.json"))
        )

        self.model = AutoModel.from_config(
            ModelConfig.from_config_file(self._download(f"virtual-human-chc/{model}", "model_config.json"))
        )

        self.model.load_pretrained(self._download(f"virtual-human-chc/{model}", "ckpt.pt"))
        self.model.to(self.device)
        model.eval()
    
    def preprocess(self, data):
        ...
    
    def transform(self, data, **kwargs):
        """
        Creates embeddings for the provided input. 
        The function expects the tokenizer and the model to be loaded already. 

        Parameters 
        ---------- 
        inputs : str 
            Path to the raw data file containing molecular representations. 

        batch_size : int 
            Size of the batch 

        device: str 
            Device used 

        Returns 
        ------- 
        numpy.ndarray 
            Embeddings of the input data 

        Example
        -------
        {'output': [[0.12989292, -0.04472789, 1.27521825 ... -0.31017503, -2.61905527, -0.26748869] 
        [ 0.04795801, -0.71846646, 3.47797537 ...  2.37488675, -0.28063831, 1.84492266] 
        [-0.00499679, 0.72711295, 0.48343059 ... -1.17737067, 0.93289232, 0.32299849]]} 
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.preprocess(data)

        featurizer = self.model.to_encoder(self.tokenizer, 128, self.device) # batch_size=128
        embeddings = featurizer.encode(inputs)
        
        return {'output': embeddings} 
    
if __name__=='__main__':
    ...