"""Download a Hugging Face model repo with hf_transfer's parallel downloader (read-only).

    python benchmarks/hf_fast_download.py Qwen/Qwen3.5-2B-Base <revision|main> /root/models/qwen3.5-2b-base
huggingface_hub 2.x no longer routes downloads through hf_transfer (HF_HUB_ENABLE_HF_TRANSFER is ignored) and its Xet
path can stall on rented boxes; calling hf_transfer.download directly (parallel range requests) has reached the
box's line rate (~60 MB/s). Files already complete (right size, and right sha256 for LFS files) are skipped; every
downloaded LFS file is checked against its sha256. Uses $HF_TOKEN if set (higher rate limits), works without for public repos.
When the repo has safetensors weights, other copies of the same weights are skipped (pytorch_model*.bin and its index,
consolidated .pth/.pt, TensorFlow / Flax / Rust files, original/): vLLM and transformers load the safetensors, and some
repos ship both. HF_FAST_ALL_FILES=1 downloads everything.
"""
import fnmatch

import hashlib
import os
import sys
import time

import hf_transfer
import requests
from huggingface_hub import HfApi, hf_hub_url

# Parallel range requests. One connection gets 3-5 MB/s on some rented boxes, 8 get ~37 MB/s; 96 made HF's Xet
# bridge reset connections ("error decoding response body"), so 32 by default.
CONNECTIONS = int(os.environ.get("HF_FAST_CONNECTIONS", "32"))
CHUNK_MB = int(os.environ.get("HF_FAST_CHUNK_MB", "16"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(64 << 20), b""):
            h.update(block)
    return h.hexdigest()


DUPLICATE_WEIGHTS = ["pytorch_model*.bin", "pytorch_model.bin.index.json", "consolidated*.pth", "consolidated*.pt",
                     "tf_model.h5", "flax_model.msgpack", "rust_model.ot", "original/*"]


def wanted(siblings):
    """The repo's files minus duplicate weight copies when safetensors weights are there (module docstring)."""
    if os.environ.get("HF_FAST_ALL_FILES") == "1" or not any(s.rfilename.endswith(".safetensors") for s in siblings):
        return list(siblings), []
    keep, skip = [], []
    for s in siblings:
        (skip if any(fnmatch.fnmatch(s.rfilename, pat) for pat in DUPLICATE_WEIGHTS) else keep).append(s)
    return keep, skip


def main():
    repo, revision, out = sys.argv[1], sys.argv[2], sys.argv[3]
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=token)
    files, skipped = wanted(info.siblings)
    print(f"{repo}@{info.sha}: {len(files)} files, {sum(s.size or 0 for s in files) / 1e9:.2f} GB"
          + (f" ({len(skipped)} duplicate weight files skipped, {sum(s.size or 0 for s in skipped) / 1e9:.2f} GB)"
             if skipped else ""))
    os.makedirs(out, exist_ok=True)
    t_all, total = time.time(), 0
    for s in files:
        dst = os.path.join(out, s.rfilename)
        # hf_transfer pre-allocates the whole file, so an interrupted download has the right size: check the content
        if (os.path.exists(dst) and s.size is not None and os.path.getsize(dst) == s.size
                and (s.lfs is None or sha256(dst) == s.lfs.sha256)):
            continue
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        t = time.time()
        url = hf_hub_url(repo, s.rfilename, revision=info.sha)
        # LFS files redirect to a signed CDN / Xet-bridge URL; hf_transfer's range requests only fan out against the
        # final URL (against the redirecting one it ends up on a single connection), so resolve the redirect first
        r = requests.head(url, headers=headers, allow_redirects=False, timeout=30)
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location", "").startswith("http"):
            url, hdrs = r.headers["location"], None          # signed: no token needed (or wanted) on the CDN
        else:
            hdrs = headers
        hf_transfer.download(url=url, filename=dst, max_files=CONNECTIONS, chunk_size=CHUNK_MB << 20,
                             parallel_failures=CONNECTIONS, max_retries=10, headers=hdrs)
        dt = time.time() - t
        size = os.path.getsize(dst)
        total += size
        ok = s.size is None or size == s.size
        if ok and s.lfs is not None:
            ok = sha256(dst) == s.lfs.sha256
        print(f"  {s.rfilename}: {size / 1e6:.1f} MB in {dt:.1f}s ({size / max(dt, 1e-6) / 1e6:.0f} MB/s)"
              f"{'' if ok else '  CHECK FAILED'}", flush=True)
        if not ok:
            sys.exit(f"size/sha256 mismatch for {s.rfilename}")
    with open(os.path.join(out, ".revision"), "w") as f:
        f.write(f"{repo} {info.sha}\n")
    print(f"done: {total / 1e9:.2f} GB in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
