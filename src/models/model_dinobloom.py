from .vhmodels import VHModel

class DinoBloom(
    VHModel,
    name="DinoBloom",
    description="a ViT (Vision Transformer) built upon DINOv2 (Meta AI) and trained on data of single cells from peripheral blood and bone marrow", 
    link="https://huggingface.co/virtual-human-chc/DinoBloom"                       
):
        
    def load_model(self, repo=None, **kwargs):
        """    
        Downloads and loads the specified model version - S, B, L, G - of the DinoBloom models. This includes the following steps: 
        - Loading “facebookresearch/dinov2” 
        - Load “pytorch_model_{variant}.bin” 
        
        For more information, check (link to HF repo).

        Parameters
        -------
        model : str 
            Version of the model 

        Returns 
        ------- 

        dinov2.models.vision_transformer.DinoVisionTransformer (inherits from nn.torch.Module) 
            The loaded model 
        """
        ...
    
    def encode(self, inputs, **kwargs):
        """
        Creates the embeddings for the input data. The function expects the model to be loaded already. 

        Transforms the input - resize and normalize. 

        Parameters
        -------
        inputs: str
            Path to folder or image

        Returns 
        ------- 

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