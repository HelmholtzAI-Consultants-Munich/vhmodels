from vhmodels.vh_checker.base import BaseModel
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from PIL import Image

class DinoBloom(
    BaseModel,
    project="dinobloom",
    description="a ViT (Vision Transformer) built upon DINOv2 (Meta AI) and trained on data of single cells from peripheral blood and bone marrow", 
    link="https://huggingface.co/virtual-human-chc/DinoBloom"                       
):
    def __init__(self):
        self.model_config = {
            "s": ("dinov2_vits14", 384),
            "b": ("dinov2_vitb14", 768),
            "l": ("dinov2_vitl14", 1024),
            "g": ("dinov2_vitg14", 1536),
        }

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.model = None
        self.device = None
        
    def load_model(self, model=None, **kwargs):
        """    
        Downloads and loads the specified model version - S, B, L, G - of the DinoBloom models. This includes the following steps: 
        - Loading “facebookresearch/dinov2” 
        - Load “pytorch_model_{model}.bin” 
        
        For more information, check (link to HF repo).

        Parameters
        -------
        model : str 
            Version of the model 
        
        device : str
            The device that should be used. Either "cuda" or "cpu"
        
        Returns
        ------
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

        self.device = torch.device(self.device)

        dinov2_model, embed_dim = self.model_config[model]

        # Load base DINOv2 model
        self.model = torch.hub.load("facebookresearch/dinov2", dinov2_model)

        # Download DinoBloom weights
        ckpt_path = hf_hub_download(
            repo_id="virtual-human-chc/DinoBloom",
            filename=f"pytorch_model_{model}.bin"
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")

        num_tokens = int(1 + (224 / 14) ** 2)
        self.model.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.model.load_state_dict(ckpt, strict=True)
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, data):
        """
        Preprocess images for transformer models.

        Applies the following transformation:
        - Resize the image to 224x224
        - ToTensor
        - Normalize

        Parameters
        ----------
        inputs : str | Path | PIL.Image | list
            - Path to image
            - Path to folder of images
            - PIL image
            - List of images or paths

        Returns
        -------
        torch.Tensor
            Batch tensor of shape (N, 3, 224, 224)
        """

        if data.get('inputs', None) is None:    
            raise ValueError("'data' should be a dict with key 'inputs'")
        
        inputs = data.get('inputs', None)

        images = []

        # convert Path objects
        if isinstance(inputs, Path):
            inputs = str(inputs)

        # case 1: path input
        if isinstance(inputs, str):
            p = Path(inputs)

            if p.is_dir():
                for img_path in sorted(p.iterdir()):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)

            elif p.is_file():
                img = Image.open(p).convert("RGB")
                images.append(img)

            else:
                raise ValueError("Invalid image path")

        # case 2: single PIL image
        elif isinstance(inputs, Image.Image):
            images.append(inputs)

        # case 3: list of images
        elif isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, str):
                    img = Image.open(item).convert("RGB")
                    images.append(img)
                elif isinstance(item, Image.Image):
                    images.append(item)
                else:
                    raise ValueError(f"Unsupported type: {type(item)}")

        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

        # apply transforms
        tensors = [self.img_transform(img) for img in images]

        # stack batch
        return torch.stack(tensors)

    def transform(self, data, **kwargs):
        """
        Creates the embeddings for the input data. The function expects the model to be loaded already. 

        Parameters
        -------
        inputs : torch.Tensor
            Preprocessed images (N, 3, 224, 224)
        device : torch.device or str, optional

        Returns 
        ------- 
        torch.Tensor
            Feature embeddings (N, D)
        """
        # if data.get('inputs') is None:    
        #     raise ValueError("'data' should be a dict with key 'inputs'")
    
        # raw_inputs = data.get('inputs')
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.preprocess(data)

        # 2. Device handling
        device = kwargs.get("device") or self.device
        self.model.to(device)
        inputs = inputs.to(device)

        # 3. Inference
        self.model.eval()
        with torch.no_grad():
            features = self.model(inputs)

        # 4. Handle Output: Move to CPU and convert to List for JSON serialization
        # This is crucial for the Runner to send data back to the Proxy
        return {'output': features.cpu().tolist()}
    
if __name__=='__main__':
    ...