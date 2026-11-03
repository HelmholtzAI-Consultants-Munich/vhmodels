"""Embeds the H-Optimus-0 example tiles on the CPU actor (fp32) and the GPU
actor (fp16 autocast) and compares them: per tile the latency on each, and the
cosine similarity of the two embeddings. Exits 1 if any tile's similarity is
below MIN_COSINE, i.e. the GPU path computes something else.

Usage: python3 compare.py API_URL
"""

import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MIN_COSINE = 0.999


def embed(url, tile):
    request = urllib.request.Request(url, data=tile, headers={"Content-Type": "application/octet-stream"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        embedding = json.load(response)["embedding"]
    return time.perf_counter() - start, embedding


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    api = sys.argv[1].rstrip("/")
    worst = 1.0
    for tile in sorted((ROOT / "example_data/H-Optimus-0").glob("*.tif")):
        cpu_seconds, cpu = embed(f"{api}/embed/hoptimus0", tile.read_bytes())
        gpu_seconds, gpu = embed(f"{api}/embed/hoptimus0-gpu", tile.read_bytes())
        similarity = cosine(cpu, gpu)
        worst = min(worst, similarity)
        print(f"{tile.name:<20} CPU {cpu_seconds:6.2f}s  GPU {gpu_seconds:6.2f}s  cosine {similarity:.6f}")
    if worst < MIN_COSINE:
        print(f"--- GPU embeddings differ from CPU: cosine {worst:.6f} < {MIN_COSINE}")
        sys.exit(1)
    print(f"--- GPU embeddings match CPU (cosine >= {MIN_COSINE})")


if __name__ == "__main__":
    main()
