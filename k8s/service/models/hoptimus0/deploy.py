"""Ray job: start H-Optimus-0 as a named, detached actor on the model worker.

The actor outlives this job, so the inference API can look it up by name.
Packages come from Ray's runtime_env (CPU-only torch), so no custom image is
needed. Same checkpoint, revision and preprocessing as
vhmodels/models/HOptimus0/model.py.

With --gpu, the actor is "hoptimus0-gpu" on the GPU worker (ray-worker-gpu.yaml): CUDA
torch, and fp16 autocast as vhmodels uses on CUDA by default (amp=True).
"""

import io
import sys

import ray

GPU = "--gpu" in sys.argv[1:]
TORCH = "cu126" if GPU else "cpu"  # the CUDA build vhmodels pins

MODEL = "hf-hub:bioptimus/H-optimus-0@b145cc1e6c6b30d3251aa8b1f844e6974188a743"
MEAN = (0.707223, 0.578729, 0.703617)
STD = (0.211883, 0.230117, 0.177517)

RUNTIME_ENV = {
    "pip": {
        "packages": [
            f"torch==2.9.1+{TORCH}",
            f"torchvision==0.24.1+{TORCH}",
            "timm==1.0.29",
            "huggingface_hub==1.1.7",
        ],
        "pip_install_options": ["--extra-index-url", f"https://download.pytorch.org/whl/{TORCH}"],
    }
}


@ray.remote
class HOptimus0:
    def __init__(self, device):
        import timm
        import torch
        from huggingface_hub import hf_hub_download
        from torchvision import transforms

        # A memory-mapped checkpoint keeps the peak near the 4.5GB of weights,
        # instead of ~9GB for timm's pretrained=True path.
        self.torch = torch
        self.model = timm.create_model(
            MODEL, pretrained=False, init_values=1e-5, dynamic_img_size=False
        ).eval()
        repo, revision = MODEL.removeprefix("hf-hub:").split("@")
        checkpoint = hf_hub_download(repo, "pytorch_model.bin", revision=revision)
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        self.model.load_state_dict(state)
        del state
        self.device = torch.device(device)
        self.model.to(self.device)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
        )

    def _forward(self, batch):
        with self.torch.inference_mode(), self.torch.autocast(
            "cuda", dtype=self.torch.float16, enabled=self.device.type == "cuda"
        ):
            return self.model(batch.to(self.device)).float().cpu()

    def warm_up(self):
        """Run one blank tile through the model and return the embedding size."""
        return self._forward(self.torch.zeros(1, 3, 224, 224)).shape[1]

    def gpu_info(self):
        """Name and memory of the actor's GPU, and the CUDA version torch uses."""
        props = self.torch.cuda.get_device_properties(self.device)
        return f"{props.name}, {props.total_memory / 1024**3:.0f}GiB, CUDA {self.torch.version.cuda}"

    def embed(self, image_bytes):
        """Embed one 224 x 224 RGB tile, given as encoded image bytes."""
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.size != (224, 224):
            raise ValueError(f"expected a 224 x 224 tile, got {image.size}")
        return self._forward(self.transform(image).unsqueeze(0))[0].tolist()


ray.init(namespace="vhmodels")
name = "hoptimus0-gpu" if GPU else "hoptimus0"
# Only the GPU worker has a GPU, only the CPU model worker the "hoptimus0" resource.
placement = {"num_gpus": 1} if GPU else {"resources": {"hoptimus0": 1}}
actor = HOptimus0.options(
    name=name,
    lifetime="detached",
    get_if_exists=True,
    num_cpus=2,
    memory=5 * 1024**3,
    runtime_env=RUNTIME_ENV,
    **placement,
).remote("cuda" if GPU else "cpu")

# The first call waits for pip install, the ~4.5GB checkpoint download and
# model loading.
dim = ray.get(actor.warm_up.remote())
print(f"actor {name!r} ready, embedding dim {dim}")
if GPU:
    print("GPU:", ray.get(actor.gpu_info.remote()))
