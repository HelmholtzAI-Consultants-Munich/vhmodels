from .vhmodels import VHModel
from pprint import pprint
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.models.auto import AutoModel
from hyformer.utils import set_seed

SEED = 1337
set_seed(SEED)

class Hyformer(
    VHModel,
    name="hyformer",
    description="A joint transformer-based model that unifies a generative decoder with a predictive encoder", 
    link="https://huggingface.co/collections/virtual-human-chc/hyformer"                       
):
    capabilities = {"encode", "predict", "generate"}
    
    def load_model(self, repo=None, **kwargs):
        self._download(repo, "vocab.txt")

        tokenizer = AutoTokenizer.from_config(
            TokenizerConfig.from_config_file(self._download(repo, "tokenizer_config.json"))
        )

        self.tokenizer = tokenizer

        model = AutoModel.from_config(
            ModelConfig.from_config_file(self._download(repo, "downstream_config.json")),
            downstream_task="classification",
            num_tasks=1,
        )

        model.load_pretrained(self._download(repo, "ckpt.pt"))

        self.model = model
        return model
    
    def encode(self, inputs, **kwargs):
        self.model.to(kwargs.get('device', None) or self.device)
        self.model.eval()

        featurizer = self.model.to_encoder(self.tokenizer, 128, kwargs.get('device', None) or self.device) # batch_size=128
        embeddings = featurizer.encode(inputs)
        return embeddings

    def predict(self, inputs, **kwargs):
        model = self.model.to_predictor(self.tokenizer, batch_size=kwargs.get('batch_size', None) or 128, device=kwargs.get('device', None) or self.device) #batch_size = 128
        return model.predict(inputs)
    
    def generate(self, num_samples, **kwargs):
        generator = self.model.to_generator(self.tokenizer, 256, 0.9, 25, device=kwargs.get('device', None) or self.device) # batch_size=256, temperature=0.9, top_k=25
        sequences = generator.generate(num_samples)
        return sequences
    
if __name__=='__main__':
    
    hyformer = Hyformer()
    hyformer.load_model('virtual-human-chc/hyformer_molecules_50M')
    sequences = Path("input/sequences.smiles").read_text().splitlines()
    print(hyformer.predict(sequences, batch_size=128))
    print(hyformer.encode(sequences))
    #print(hyformer.generate(num_samples=3))
    pprint(hyformer.list_sources())