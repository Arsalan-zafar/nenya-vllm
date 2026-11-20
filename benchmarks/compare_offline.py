"""Speed, energy and memory of one model under vLLM, offline (LLM class), with settings shared by every model compared.

    python benchmarks/compare_offline.py --model /root/models/qwen3.5-2b-base --name qwen3.5-2b \
        [--extra '{"limit_mm_per_prompt": {"image": 0, "video": 0}}'] [--text FILE] [--quick]
Writes results/<name>/offline.json. Prompts are windows of the same real text (--text; by default
benchmarks/data/origin_of_species.txt, an excerpt of Charles Darwin's On the Origin of Species, 1859, public domain),
tokenized by each model's own tokenizer, so every model reads comparable text at identical token counts; ignore_eos
throughout, greedy decoding, prefix caching off (every prompt is prefilled in full).

  prefill         one request, max_tokens=1, at 128 / 512 / 2K / 8K / 16K prompt tokens: latency (= TTFT) and tok/s
  prefill_batch   64 x 2,048 and 16 x 8,192 prompt tokens, max_tokens=1: prefill tok/s
  decode          one request, 256 new tokens after a 128 / 2K / 8K / 16K prompt: ms/token = (T(257) - T(1)) / 256
  decode_batch    B requests of 512 prompt + 256 new tokens, B = 1..256: aggregate and per-stream decode tok/s,
                  decode time = T(prompts, 256 new) - T(prompts, 1 new)
  mixed           256 x (1,024 in, 256 out): output and total tok/s
NVML samples power, SM and memory-controller utilisation every 50 ms; every test reports the averages over its window
(the engine's own process included: NVML is per device) and energy per generated token.
"""

import argparse
import json
import os
import statistics
import threading
import time

import pynvml
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from _common import make_llm

HERE = os.path.dirname(os.path.abspath(__file__))


class NvmlSampler(threading.Thread):
    def __init__(self, period=0.05):
        super().__init__(daemon=True)
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.period, self.samples, self.stop_flag = period, [], False

    def run(self):
        while not self.stop_flag:
            u = pynvml.nvmlDeviceGetUtilizationRates(self.h)
            self.samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0, u.gpu,
                                 u.memory, pynvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**30,
                                 pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)))
            time.sleep(self.period)

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return {}
        mean = lambda i: round(statistics.fmean(x[i] for x in s), 2)  # noqa: E731
        return {"power_w": mean(1), "sm_util_pct": mean(2), "mem_util_pct": mean(3), "mem_used_gib": mean(4),
                "sm_clock_mhz": mean(5), "samples": len(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--extra", default="{}", help="JSON dict of extra LLM(...) kwargs")
    ap.add_argument("--quick", action="store_true", help="fewer sizes, for a smoke test")
    ap.add_argument("--text", default=os.path.join(HERE, "data", "origin_of_species.txt"),
                    help="the text the prompts are cut from")
    args = ap.parse_args()
    out_dir = os.path.join("results", args.name)
    os.makedirs(out_dir, exist_ok=True)

    nv = NvmlSampler()
    nv.start()
    kw = dict(model=args.model, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, max_num_seqs=256, max_num_batched_tokens=16384, seed=0)
    kw.update(json.loads(args.extra))
    t0 = time.perf_counter()
    llm = make_llm(**kw)
    startup_s = time.perf_counter() - t0
    cc = llm.llm_engine.vllm_config.cache_config
    tok = llm.get_tokenizer()
    text = open(args.text, encoding="utf-8").read()
    ids = tok(text, add_special_tokens=False, split_special_tokens=True).input_ids
    corpus = len(ids)
    offset = [0]

    def window(n):
        """the next n-token window of the corpus (cyclic), a different one on every call"""
        s = offset[0] % corpus
        offset[0] += 7919
        return TokensPrompt(prompt_token_ids=[ids[(s + i) % corpus] for i in range(n)])

    def timed(prompts, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        t1 = time.perf_counter()
        return t1 - t, t, t1, sum(len(o.outputs[0].token_ids) for o in outs)

    # warm-up: every batch shape the tests use, single requests included (first-use JIT / autotuning of decode
    # kernels otherwise lands in the first measured run)
    for n, L, m in [(1, 128, 64), (1, 2048, 64), (8, 512, 32), (64, 512, 16), (256, 512, 8), (1, 16000, 8)]:
        timed([window(L) for _ in range(n)], m)

    def repeat(fn, min_s=2.0, min_runs=3, max_runs=50):
        """run fn() until it has run min_runs times AND for min_s seconds in total (NVML averages need >= ~1-2 s);
        returns the list of (seconds, t_start, t_end, tokens)"""
        runs = []
        while len(runs) < min_runs or (sum(r[0] for r in runs) < min_s and len(runs) < max_runs):
            runs.append(fn())
        return runs
    res = {"name": args.name, "model": args.model, "settings": {k: v for k, v in kw.items() if k != "model"},
           "startup_s": round(startup_s, 1), "corpus_file": os.path.basename(args.text), "corpus_tokens": corpus,
           "cache_config": {"mamba_ssm_cache_dtype": str(getattr(cc, "mamba_ssm_cache_dtype", None)),
                            "mamba_cache_dtype": str(getattr(cc, "mamba_cache_dtype", None)),
                            "cache_dtype": str(cc.cache_dtype), "block_size": cc.block_size,
                            "num_gpu_blocks": cc.num_gpu_blocks}}

    lengths = [128, 2048, 16000] if args.quick else [128, 512, 2048, 8192, 16000]
    res["prefill"] = []
    for L in lengths:
        runs = repeat(lambda: timed([window(L)], 1))
        dt = statistics.median(r[0] for r in runs)
        res["prefill"].append({"prompt_tokens": L, "ttft_ms": round(1000 * dt, 2), "tok_per_s": round(L / dt),
                               "runs": len(runs), **nv.window(runs[0][1], runs[-1][2])})
        print("prefill", res["prefill"][-1], flush=True)

    res["prefill_batch"] = []
    for n, L in ([(64, 2048)] if args.quick else [(64, 2048), (16, 8192)]):
        dt, a, b, _n = timed([window(L) for _ in range(n)], 1)
        res["prefill_batch"].append({"requests": n, "prompt_tokens": L, "seconds": round(dt, 3),
                                     "tok_per_s": round(n * L / dt), **nv.window(a, b)})
        print("prefill_batch", res["prefill_batch"][-1], flush=True)

    res["decode"] = []
    for L in ([128, 16000] if args.quick else [128, 2048, 8192, 16000]):
        d1 = statistics.median(timed([window(L)], 1)[0] for _ in range(3))
        runs = repeat(lambda: timed([window(L)], 257), min_s=2.0, min_runs=3)
        dt = statistics.median(r[0] for r in runs)
        ms = 1000 * (dt - d1) / 256
        res["decode"].append({"context_tokens": L, "ms_per_token": round(ms, 3), "tok_per_s": round(1000 / ms, 1),
                              "energy_j_per_token": None, **nv.window(runs[0][1], runs[-1][2])})
        w = res["decode"][-1]
        if w.get("power_w"):
            w["energy_j_per_token"] = round(w["power_w"] * ms / 1000, 4)
        print("decode", w, flush=True)

    res["decode_batch"] = []
    for B in ([1, 64] if args.quick else [1, 8, 32, 64, 128, 256]):
        prompts = [window(512) for _ in range(B)]
        p_dt = statistics.median(timed(prompts, 1)[0] for _ in range(3))
        runs = repeat(lambda: timed(prompts, 257), min_s=2.0, min_runs=2)
        dt = statistics.median(r[0] for r in runs)
        a, b = runs[0][1], runs[-1][2]
        dec = dt - p_dt
        row = {"batch": B, "prefill_s": round(p_dt, 3), "total_s": round(dt, 3), "decode_s": round(dec, 3),
               "aggregate_tok_per_s": round(B * 256 / dec, 1), "per_stream_tok_per_s": round(256 / dec, 1),
               "ms_per_step": round(1000 * dec / 256, 3), **nv.window(a, b)}
        if row.get("power_w"):
            row["energy_j_per_token"] = round(row["power_w"] * dec / (B * 256), 5)
        res["decode_batch"].append(row)
        print("decode_batch", row, flush=True)

    dt, a, b, n = timed([window(1024) for _ in range(256)], 256)
    res["mixed"] = {"requests": 256, "prompt_tokens": 1024, "output_tokens": 256, "seconds": round(dt, 2),
                    "output_tok_per_s": round(n / dt, 1), "total_tok_per_s": round((n + 256 * 1024) / dt, 1),
                    **nv.window(a, b)}
    print("mixed", res["mixed"], flush=True)

    nv.stop_flag = True
    json.dump(res, open(os.path.join(out_dir, "offline.json"), "w"), indent=1)
    print("wrote", os.path.join(out_dir, "offline.json"))


if __name__ == "__main__":
    main()
