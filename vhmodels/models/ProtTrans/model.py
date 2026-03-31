from vhmodels.vh_checker.base import BaseModel
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

class ProtTrans(
    BaseModel,
    #project="prottrans",
    #description="State of the art pre-trained models for proteins", 
    #link="https://huggingface.co/virtual-human-chc/prot_t5_xl_uniref50"                       
):

    def __init__(self):
        self.available_models = [
            "prot_bert_bfd", 
            "prot_t5_xl_uniref50", 
            "prot_t5_xxl_bfd",
            "prot_t5_xxl_uniref50",
            "prot_xlnet",
            "prot_electra_generator_bfd",
            "prot_electra_discriminator_bfd",
            "prot_albert", 
            "prot_t5_xl_bfd",
        ]

        self.tokenizer = None
        self.model = None
        self.device = None

    def load_model(self, model=None, **kwargs):
        """    
        Downloads and loads the artifacts for the specified model in the ProtTrans HF repository. Possible options are: 
        - prot_bert_bfd 
        - prot_t5_xl_uniref50
        - prot_t5_xxl_bfd 
        - prot_t5_xxl_uniref50
        - prot_xlnet 
        - prot_electra_generator_bfd
        - prot_electra_discriminator_bfd
        - prot_albert
        - prot_t5_xl_bfd
        
        For more information, check (link to the HF repo). 

        Parameters
        -------- 
        model: str 
            Name of the model 

        device: str
            Device to work on  

        Returns 
        ------- 
        None  
        """
        if model == "prot_t5_xl_uniref50":
            self.device = kwargs.get(
                "device",
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            # Load the tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(f'virtual-human-chc/{model}', do_lower_case=False)

            # Load the model
            self.model = T5EncoderModel.from_pretrained(f'virtual-human-chc/{model}').to(self.device)
      
            # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
            if self.device == torch.device("cpu"):
                self.model.to(torch.float32)
        else:
            ... # implement for other models
        
    def _preprocess(self, input):
        """
        Expects input like "MKVILLLLAVVAFGHALCRV".

        Example input: [“PRTEINO”, “SEQWENCE”] 
        """
        return [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in input]   
    
    def transform(self, input, **kwargs):
        """
        Creates the embeddings for the input data. The function expects the model to be loaded already. 

        The input should be already preprocessed. See function "preprocess".

        The returned embeddings have different length. You should remove the padding. To get the first sequence without padding:
        - embedding_repr.last_hidden_state[0,:7]

        To get per protein embedding:
        - emb_0.mean(dim=0)

        Parameters
        ---------- 
        inputs: numpy.ndarray 
            Protein sequences to be encoded 

        Returns
        ---------- 
        dict
            A dictionary containing:
            - 'output': list of lists
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        input = self._preprocess(input)
        
        ids = self.tokenizer.batch_encode_plus(input, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        return {'output': embedding_repr}
    
if __name__=='__main__':
    ...