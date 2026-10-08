"""Tile-level feature extraction with Bioptimus H-Optimus-0."""

from contextlib import nullcontext
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

from vhmodels.models.registry import REGISTRY
from vhmodels.models.source_resolver import SourceResolver
from vhmodels.utils.device import resolve_torch_device
from vhmodels.vh_checker.base import BaseModel


class HOptimus0(BaseModel):
    """Embed 224 x 224 H&E tiles with H-Optimus-0."""

    PROJECT = "hoptimus0"
    TILE_SIZE = (224, 224)
    EMBEDDING_DIM = 1536
    SUPPORTED_MODES = {"RGB", "RGBA"}
    SUPPORTED_SUFFIXES = {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(self):
        self.model = None
        self.device = None
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

    def load_model(self, model=None, **kwargs):
        """Load the gated checkpoint through ``timm``.

        First accept the Hugging Face model's access conditions and make a
        read token available to this worker (for example via ``HF_TOKEN``).
        """
        manifest = REGISTRY.resolve(self.PROJECT, model)
        resources = SourceResolver().resolve(manifest.sources, manifest.model_dir)
        checkpoint = resources["checkpoint"]

        model_name = f"hf-hub:{checkpoint.repo_id}"
        if checkpoint.revision:
            model_name = f"{model_name}@{checkpoint.revision}"

        self.device = resolve_torch_device(torch, kwargs.get("device", "auto"))
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def _validate_tile(cls, image, source):
        """Check the image mode and size (both available from the header)."""
        if image.mode not in cls.SUPPORTED_MODES:
            raise ValueError(
                "H-Optimus-0 expects 8-bit RGB(A) tiles; "
                f"{source} has image mode {image.mode!r}."
            )
        if image.size != cls.TILE_SIZE:
            raise ValueError(
                "H-Optimus-0 expects 224 x 224 tiles sampled at 0.5 microns "
                f"per pixel; {source} has size {image.size[0]} x {image.size[1]}. "
                "Whole-slide images must be tiled before calling embed()."
            )

    @classmethod
    def _collect_tiles(cls, input):
        """Return the ordered tiles (paths or PIL images) after validating them.

        Only file headers are read here; pixels are decoded per batch in
        ``_preprocess``, so memory scales with batch size, not input size.
        """
        # case 1: path input (file or folder of tiles)
        if isinstance(input, (str, Path)):
            path = Path(input)
            if path.is_dir():
                tiles = [
                    candidate
                    for candidate in sorted(path.iterdir())
                    if candidate.is_file()
                    and candidate.suffix.lower() in cls.SUPPORTED_SUFFIXES
                ]
                if not tiles:
                    raise ValueError(f"No supported image tiles found in: {path}")
            else:
                tiles = [path]

        # case 2: single PIL image
        elif isinstance(input, Image.Image):
            tiles = [input]

        # case 3: list of images or paths
        elif isinstance(input, list):
            if not input:
                raise ValueError("At least one image tile is required.")
            tiles = input

        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # validate every tile from its header before any decoding
        for tile in tiles:
            if isinstance(tile, Image.Image):
                cls._validate_tile(tile, "a supplied image")
            elif isinstance(tile, (str, Path)):
                if not Path(tile).is_file():
                    raise FileNotFoundError(f"Image file not found: {tile}")
                with Image.open(tile) as image:
                    cls._validate_tile(image, tile)
            else:
                raise ValueError(f"Unsupported list item type: {type(tile)}")
        return tiles

    def _preprocess(self, tiles):
        """Decode a batch of validated tiles into a (N, 3, 224, 224) tensor.

        Applies RGB conversion, ToTensor, and H-Optimus-0 normalization.
        """
        tensors = []
        for tile in tiles:
            if isinstance(tile, Image.Image):
                image = tile.convert("RGB")
            else:
                with Image.open(tile) as opened:
                    image = opened.convert("RGB")
            tensors.append(self.img_transform(image))
        return torch.stack(tensors)

    def _autocast_context(self, amp):
        if amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def embed(self, input, batch_size=1, amp=True, **kwargs):
        """Return one 1,536-dimensional embedding per 224 x 224 tile.

        ``input`` can be a tile path, directory, PIL image, or ordered list of
        paths/images. Whole-slide tiling and slide-level prediction are outside
        this model's feature-extraction interface.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")

        tiles = self._collect_tiles(input)
        all_features = []

        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(tiles), batch_size):
                batch = self._preprocess(tiles[start : start + batch_size]).to(
                    self.device
                )
                with self._autocast_context(amp):
                    features = self.model(batch)
                if features.ndim != 2 or features.shape[1] != self.EMBEDDING_DIM:
                    raise RuntimeError(
                        "Unexpected H-Optimus-0 output shape: "
                        f"{tuple(features.shape)}; expected (batch, {self.EMBEDDING_DIM})."
                    )
                all_features.append(features.float().cpu())

        return {"output": torch.cat(all_features, dim=0).tolist()}

    def predict(self, input, **kwargs):
        raise NotImplementedError("H-Optimus-0 does not provide a prediction head.")

    def generate(self, input, **kwargs):
        raise NotImplementedError("H-Optimus-0 does not support generation.")
