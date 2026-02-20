from .vhmodels import VHModel

class ProtTrans(
    VHModel,
    name="ProtTrans",
    description="State of the art pre-trained models for proteins", 
    link="https://huggingface.co/virtual-human-chc/prot_t5_xl_uniref50"                       
):
        
    def load_model(self, model=None, **kwargs):
        """    
        Downloads and loads the artifacts for the specified model in the ProtTrans HF repository. Possible options are: 
        - prot_bert_bfd: loads encoder and two prediction models - prot_bert_bfd_membrane and prot_bert_bfd_ss3 
        - prot_t5_xl_uniref50: loads  encoder 
        - prot_t5_xxl_bfd: loads encoder 
        - prot_t5_xxl_uniref50: loads encoder 
        - prot_xlnet: loads encoder 
        - prot_electra_generator_bfd: loads encoder 
        - prot_electra_discriminator_bfd: loads encoder 
        - prot_albert: loads encoder 
        - prot_t5_xl_bfd: loads encoder 
        
        For more information, check (link to the HF repo). 

        Parameters
        -------- 
        model: str 
            Name of the model 
        
        Returns 
        ------- 

        Type of the model 
            The loaded model 
        """
        ...
    
    def encode(self, inputs, **kwargs):
        """
        Creates the embeddings for the input data. The function expects the model to be loaded already. 

        Example input: [“PRTEINO”, “SEQWENCE”] 

        Parameters
        ---------- 
        inputs: numpy.ndarray 
            Protein sequences to be encoded 

        Returns
        ---------- 
        numpy.ndarray 
            Embeddings of the input data 
        """
        ...

    def predict(self, inputs, **kwargs):
        ...
    
    def generate(self, num_samples, **kwargs):
        ...
    
if __name__=='__main__':
    ...