from vhmodels.vh_checker.base import BaseModel

from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from PIL import Image


class DinoBloom(BaseModel):
    def __init__(self):
        self.model_config = {
            "s": ("dinov2_vits14", 384),
            "b": ("dinov2_vitb14", 768),
            "l": ("dinov2_vitl14", 1024),
            "g": ("dinov2_vitg14", 1536),
        }

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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

        device : torch.device or str, optional
            The device that should be used. Either "cuda" or "cpu".

        Returns
        ------
        None
        """
        # Check if the selected model exists
        if model not in self.model_config:
            raise ValueError(
                f"Unknown model '{self.model}' in DinoBloom. Available models: {list(self.model_config.keys())}"
            )

        # Get user's input for device; otherwise, fall back to pytorch function
        self.device = torch.device(
            kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        dinov2_model, embed_dim = self.model_config[model]

        # Load base DINOv2 model
        self.model = torch.hub.load("facebookresearch/dinov2", dinov2_model)

        # Download DinoBloom weights
        ckpt_path = hf_hub_download(
            repo_id="virtual-human-chc/DinoBloom", filename=f"pytorch_model_{model}.bin"
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")

        num_tokens = int(1 + (224 / 14) ** 2)
        self.model.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.model.load_state_dict(ckpt, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, input):
        """
        Preprocess images for transformer models.

        Applies the following transformation:
        - Resize the image to 224x224
        - ToTensor
        - Normalize

        Parameters
        ----------
        input : str | Path | PIL.Image | list
            - Path to image
            - Path to folder of images
            - PIL image
            - List of images or paths

        Returns
        -------
        torch.Tensor
            Batch tensor of shape (N, 3, 224, 224)
        """
        # if data.get('inputs', None) is None:
        #     raise ValueError("'data' should be a dict with key 'inputs'")

        # inputs = data.get('inputs', None)

        images = []

        # convert Path objects
        if isinstance(input, Path):
            input = str(input)

        # case 1: path input
        if isinstance(input, str):
            p = Path(input)

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
        elif isinstance(input, Image.Image):
            images.append(input)

        # case 3: list of images
        elif isinstance(input, list):
            for item in input:
                if isinstance(item, str):
                    img = Image.open(item).convert("RGB")
                    images.append(img)
                elif isinstance(item, Image.Image):
                    images.append(item)
                else:
                    raise ValueError(f"Unsupported type: {type(item)}")

        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # apply transforms
        tensors = [self.img_transform(img) for img in images]

        # stack batch
        return torch.stack(tensors)

    def embed(self, input, batch_size=32, **kwargs):
        """
        Creates the embeddings for the input data. The function expects the model to be loaded already.

        Parameters
        -------
        input : str | Path | PIL.Image | list
            - Path to image
            - Path to folder of images
            - PIL image
            - List of images or paths

        batch_size : int, optional (default=32)
            Number of samples to process per batch.

        **kwargs : dict, optional
            Additional keyword arguments for compatibility or future extensions (currently unused).

        Returns
        -------
        dict
            A dictionary containing:
            - 'output': list of lists
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_inputs = self._preprocess(input)

        # If model isn't on the given device, move it there
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        self.model.eval()
        all_features = []

        with torch.no_grad():
            for i in range(0, len(all_inputs), batch_size):
                batch = all_inputs[i : i + batch_size].to(self.device)
                features = self.model(batch)
                all_features.append(features.cpu())

        final_tensor = torch.cat(all_features, dim=0)

        return {"output": final_tensor.tolist()}

    def predict(self, input, **kwargs):
        pass

    def generate(self, input, **kwargs):
        pass


if __name__ == "__main__":
    db = DinoBloom()

    test_image = Image.new("RGB", (224, 224), color=(73, 109, 137))

    db.load_model(model="s", device="cpu")
    result = db.embed(data=test_image)
    print(result)
    batch_result = db.embed(data=[test_image] * 5)
    print(batch_result)
