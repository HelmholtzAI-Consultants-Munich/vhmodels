from .vhmodels import VHModel

class Hyformer(
    VHModel,
    name="hyformer",
    description="A joint transformer-based model that unifies a generative decoder with a predictive encoder", 
    link="https://huggingface.co/collections/virtual-human-chc/hyformer"                       
):
        
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
        
        Returns 
        ------- 
        Type of the model 
            The loaded model 
        """
        ...
    
    def encode(self, inputs, **kwargs):
        """
        Creates embeddings for the provided input. 
        The function expects the tokenizer and the model to be loaded already. 

        Parameters 
        ---------- 
        inputs : str 
            Path to the raw data file containing molecular representations. 

        Batch_size : int 
            Size of the batch 

        Device: str 
            Device used 

        Returns 
        ------- 
        numpy.ndarray 
            Embeddings of the input data 

        Example
        -------
        [[0.12989292, -0.04472789, 1.27521825 ... -0.31017503, -2.61905527, -0.26748869] 
        [ 0.04795801, -0.71846646, 3.47797537 ...  2.37488675, -0.28063831, 1.84492266] 
        [-0.00499679, 0.72711295, 0.48343059 ... -1.17737067, 0.93289232, 0.32299849]] 
        """
        ...

    def predict(self, inputs, **kwargs):
        ...
    
    def generate(self, num_samples, **kwargs):
        ...
    
if __name__=='__main__':
    ...