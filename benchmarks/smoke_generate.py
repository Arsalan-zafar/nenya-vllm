"""Smoke test: load Nenya in vLLM and generate greedily.

    python benchmarks/smoke_generate.py --model /root/models/nenya-base [--eager] [--max-tokens 32]
"""

import argparse
import time

from vllm import SamplingParams

from _common import make_llm

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which plants",
    "1, 2, 3, 4, 5,",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/nenya-base")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    args = ap.parse_args()

    t0 = time.time()
    llm = make_llm(model=args.model, enforce_eager=args.eager, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=False, max_num_seqs=64)
    print(f"load time {time.time() - t0:.1f}s")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t0 = time.time()
    outs = llm.generate(PROMPTS, sp)
    print(f"generate time {time.time() - t0:.2f}s")
    for o in outs:
        print("=" * 80)
        print(repr(o.prompt))
        print("->", repr(o.outputs[0].text))
        print("ids", o.outputs[0].token_ids[:16])


if __name__ == "__main__":
    main()
