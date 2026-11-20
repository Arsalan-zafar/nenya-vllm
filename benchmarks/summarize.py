"""The README's benchmark tables from the results of benchmarks/compare_models.sh.

    python benchmarks/summarize.py [--results results]

Reads results/<model>_r<round>/offline.json (compare_offline.py) and results/<model>_r<round>/serve/<scenario>.json
(compare_serve.sh) and prints markdown tables. Each cell is the mean over the model's rounds; the best value of each
row is bold. The last line gives the largest run-to-run spread (highest minus lowest run, as a share of the mean).
"""

import argparse
import glob
import json
import os
import re
import statistics

LABELS = {"nenya": "Nenya-2.5B-A0.5B", "qwen3-0.6b": "Qwen3-0.6B", "qwen3.5-2b": "Qwen3.5-2B"}


def _find(rows, key, value, field):
    return next((r[field] for r in rows if r.get(key) == value), None)


# (row label, "max" if higher is better else "min", getter on offline.json)
OFFLINE = [
    *[(f"Decode, one stream, {lab} context (tok/s)", "max",
       lambda o, c=c: _find(o.get("decode", []), "context_tokens", c, "tok_per_s"))
      for c, lab in ((128, "128-token"), (2048, "2K"), (8192, "8K"), (16000, "16K"))],
    *[(f"Decode, batch {b} (tok/s)", "max",
       lambda o, b=b: _find(o.get("decode_batch", []), "batch", b, "aggregate_tok_per_s"))
      for b in (1, 8, 32, 64, 128, 256)],
    *[(f"Time to first token, {lab}-token prompt (ms)", "min",
       lambda o, p=p: _find(o.get("prefill", []), "prompt_tokens", p, "ttft_ms"))
      for p, lab in ((128, "128"), (512, "512"), (2048, "2K"), (8192, "8K"), (16000, "16K"))],
    ("Batched prefill, 64 × 2K-token prompts (tok/s)", "max",
     lambda o: _find(o.get("prefill_batch", []), "prompt_tokens", 2048, "tok_per_s")),
    ("Batched prefill, 16 × 8K-token prompts (tok/s)", "max",
     lambda o: _find(o.get("prefill_batch", []), "prompt_tokens", 8192, "tok_per_s")),
    ("Offline batch, 256 × (1,024 in, 256 out), output (tok/s)", "max",
     lambda o: (o.get("mixed") or {}).get("output_tok_per_s")),
]

SCENARIOS = [
    ("chat_c1", "Chat, 1,024 in / 256 out, 1 concurrent"),
    ("chat_c16", "Chat, 1,024 in / 256 out, 16 concurrent"),
    ("chat_c64", "Chat, 1,024 in / 256 out, 64 concurrent"),
    ("chat_c256", "Chat, 1,024 in / 256 out, 256 concurrent"),
    ("long_c1", "Long prompt, 8,192 in / 256 out, 1 concurrent"),
    ("long_c16", "Long prompt, 8,192 in / 256 out, 16 concurrent"),
    ("tools_c32", "Shared prefixes, 4 × 4,096 + 256 in / 128 out, 32 concurrent"),
]


def load(results):
    runs = {}
    for d in sorted(glob.glob(os.path.join(results, "*_r*"))):
        m = re.fullmatch(r"(.+)_r(\d+)", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        off_path = os.path.join(d, "offline.json")
        off = json.load(open(off_path)) if os.path.isfile(off_path) else {}
        srv = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(d, "serve", "*.json"))}
        runs.setdefault(m.group(1), []).append({"dir": os.path.basename(d), "offline": off, "serve": srv})
    order = [k for k in LABELS if k in runs] + sorted(k for k in runs if k not in LABELS)
    return {k: runs[k] for k in order}


def fmt(v):
    if v is None:
        return "–"
    return f"{v:.1f}" if v < 100 else f"{v:,.0f}"


def table(header, rows, runs, spreads):
    """rows: (label, better, getter on one run); returns markdown lines."""
    models = list(runs)
    out = ["| " + header + " | " + " | ".join(LABELS.get(m, m) for m in models) + " |",
           "|---" * (len(models) + 1) + "|"]
    for label, better, get in rows:
        means = []
        for m in models:
            vals = [v for v in (get(r) for r in runs[m]) if v is not None]
            mean = statistics.fmean(vals) if vals else None
            means.append(mean)
            if len(vals) > 1 and mean:
                spreads.append(((max(vals) - min(vals)) / mean, label, m))
        present = [x for x in means if x is not None]
        best = (max if better == "max" else min)(present) if len(present) > 1 else None
        cells = [f"**{fmt(x)}**" if best is not None and x == best else fmt(x) for x in means]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    runs = load(args.results)
    if not runs:
        raise SystemExit(f"no <model>_r<round> folders in {args.results}")
    spreads = []
    print("Runs: " + "; ".join(f"{LABELS.get(m, m)} {len(r)} ({', '.join(x['dir'] for x in r)})"
                               for m, r in runs.items()))
    print("\nOffline (vLLM's LLM class, benchmarks/compare_offline.py)\n")
    print("\n".join(table("Workload", [(lab, b, lambda r, g=g: g(r["offline"])) for lab, b, g in OFFLINE],
                          runs, spreads)))
    get = lambda name, field: lambda r: (r["serve"].get(name) or {}).get(field)  # noqa: E731
    print("\nOnline, output tokens per second (vllm serve, benchmarks/compare_serve.sh)\n")
    print("\n".join(table("Scenario", [(lab, "max", get(s, "output_throughput")) for s, lab in SCENARIOS],
                          runs, spreads)))
    print("\nOnline, median time to first token (ms)\n")
    print("\n".join(table("Scenario", [(lab, "min", get(s, "median_ttft_ms")) for s, lab in SCENARIOS],
                          runs, spreads)))
    print("\nOnline, median time per output token after the first (ms)\n")
    print("\n".join(table("Scenario", [(lab, "min", get(s, "median_tpot_ms")) for s, lab in SCENARIOS],
                          runs, spreads)))
    if spreads:
        s, label, m = max(spreads)
        print(f"\nLargest run-to-run spread: {100 * s:.1f}% ({label}, {LABELS.get(m, m)})")


if __name__ == "__main__":
    main()
