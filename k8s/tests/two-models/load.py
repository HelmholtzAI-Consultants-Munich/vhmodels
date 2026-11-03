"""Load test for the inference API, by default for H-Optimus-0 and ProtT5.

Phase "solo": a few sequential requests per model, one model at a time, as the
latency baseline. Phase "together": CONCURRENCY clients per model hit all
models at once for DURATION seconds. Reports, per model and phase, the request
count, failures, latency percentiles and throughput, and whether repeated
inputs returned the same embedding. Exits 1 if any request failed.

Usage: python3 load.py API_URL [DURATION] [CONCURRENCY] [MODEL ...]
MODEL is a route under /embed/: hoptimus0, hoptimus0-gpu or prottrans.
"""

import json
import random
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SOLO_REQUESTS = 3


def build_inputs():
    """(label, body, content type) per input, keyed by model route."""
    tiles = [
        (tile.name, tile.read_bytes(), "application/octet-stream")
        for tile in sorted((ROOT / "example_data/H-Optimus-0").glob("*.tif"))
    ]
    # Fixed seed, so every run sends the same batches: from one short protein
    # to one at the 1000-residue scale.
    rng = random.Random(0)
    batches = [("ubiquitin 76aa x1", [UBIQUITIN])]
    for length, count in ((100, 4), (300, 2), (1000, 1)):
        sequences = ["".join(rng.choices(AMINO_ACIDS, k=length)) for _ in range(count)]
        batches.append((f"random {length}aa x{count}", sequences))
    proteins = [
        (label, json.dumps({"sequences": sequences}).encode(), "application/json")
        for label, sequences in batches
    ]
    return {"hoptimus0": tiles, "hoptimus0-gpu": tiles, "prottrans": proteins}


def post(url, body, content_type):
    request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    first = payload["embedding"] if "embedding" in payload else payload["embeddings"][0]
    # A few leading values identify the embedding when comparing repeats.
    return time.perf_counter() - start, tuple(round(v, 3) for v in first[:8])


def client(api, model, inputs, offset, stop, results):
    """Sends the inputs round-robin until stop() is true."""
    i = offset
    while not stop(i - offset):
        label, body, content_type = inputs[i % len(inputs)]
        i += 1
        try:
            latency, fingerprint = post(f"{api}/embed/{model}", body, content_type)
            results.append((model, label, latency, fingerprint))
        except Exception as e:
            detail = e.read().decode() if hasattr(e, "read") else ""
            results.append((model, label, None, f"{e} {detail}".strip()))


def report(phase, results, seconds):
    for model in sorted({r[0] for r in results}):
        rows = [r for r in results if r[0] == model]
        latencies = sorted(r[2] for r in rows if r[2] is not None)
        failed = [r for r in rows if r[2] is None]
        if latencies:
            p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            print(
                f"{phase:<9} {model:<13} ok {len(latencies):>4}  failed {len(failed):>3}  "
                f"p50 {statistics.median(latencies):6.2f}s  p95 {p95:6.2f}s  "
                f"max {latencies[-1]:6.2f}s  {len(latencies) / seconds:5.2f} req/s"
            )
        else:
            print(f"{phase:<9} {model:<13} ok    0  failed {len(failed):>3}")
        for _, label, _, error in failed[:3]:
            print(f"    failed: {label}: {error}")
    return sum(r[2] is None for r in results)


def main():
    api = sys.argv[1].rstrip("/")
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    models = sys.argv[4:] or ["hoptimus0", "prottrans"]
    available = build_inputs()
    inputs = {model: available[model] for model in models}
    failures = 0
    everything = []

    print(f"--- solo: {SOLO_REQUESTS} rounds of each input, one model at a time")
    for model, items in inputs.items():
        results = []
        start = time.perf_counter()
        client(api, model, items, 0, lambda n: n >= SOLO_REQUESTS * len(items), results)
        failures += report("solo", results, time.perf_counter() - start)
        everything += results

    print(f"--- together: {concurrency} clients per model for {duration:.0f}s")
    results = []
    deadline = time.monotonic() + duration
    with ThreadPoolExecutor(len(inputs) * concurrency) as pool:
        for model, items in inputs.items():
            for offset in range(concurrency):
                pool.submit(client, api, model, items, offset, lambda _: time.monotonic() > deadline, results)
    failures += report("together", results, duration)
    everything += results

    print("--- per-input latency (all phases)")
    for model, items in inputs.items():
        for label, _, _ in items:
            rows = [r for r in everything if r[0] == model and r[1] == label and r[2] is not None]
            if rows:
                distinct = len({r[3] for r in rows})
                same = "same embedding" if distinct == 1 else f"{distinct} DIFFERENT embeddings"
                latencies = [r[2] for r in rows]
                print(f"{model:<13} {label:<20} n {len(rows):>3}  p50 {statistics.median(latencies):6.2f}s  {same}")

    print("--- load test", "FAILED" if failures else "passed", f"({failures} failed requests)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
