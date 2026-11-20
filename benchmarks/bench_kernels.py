"""Microbenchmarks of the decode kernels, timed as CUDA-graph replays like vLLM's
decode step.

    python benchmarks/bench_kernels.py chain          # launch floor: a chain of trivial kernels, with and without PDL
    python benchmarks/bench_kernels.py gemv           # cuBLAS vs a Triton GEMV (with PDL weight prefetch) per shape
"""

import argparse
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def graph_time_us(fn, calls_per_graph: int, replays: int = 50) -> float:
    """Average time of one fn() call, from a CUDA graph holding calls_per_graph calls."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls_per_graph):
            fn()
    replays = max(3, replays // max(1, calls_per_graph // 200)) if calls_per_graph > 200 else replays
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(replays):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / (replays * calls_per_graph)


@triton.jit
def _trivial(x_ptr, y_ptr, N: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        tl.extra.cuda.gdc_wait()
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + offs, x + 1)


def bench_chain():
    x = torch.zeros(1024, device="cuda")
    y = torch.zeros(1024, device="cuda")
    for pdl in (False, True):
        for grid in (1, 32, 170):
            def fn():
                _trivial[(grid,)](x, y, N=1024, PDL=pdl, launch_pdl=pdl, num_warps=4)
                _trivial[(grid,)](y, x, N=1024, PDL=pdl, launch_pdl=pdl, num_warps=4)
            t = graph_time_us(fn, 200) / 2
            print(f"trivial kernel chain  grid {grid:4d}  PDL {pdl!s:5s}: {t:6.2f} us per kernel")
    a = torch.zeros(1024, device="cuda", dtype=torch.bfloat16)

    def fn2():
        a.add_(1)
    print(f"torch add_ chain: {graph_time_us(fn2, 200):6.2f} us per kernel")


@triton.jit
def _gemv(x_ptr, w_ptr, y_ptr, N, K: tl.constexpr, KP2: tl.constexpr, BN: tl.constexpr, PDL: tl.constexpr,
          PREFETCH: tl.constexpr):
    """y[n] = bf16(sum_k w[n, k] x[k]), one program per BN rows, the whole K in one tile."""
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, KP2)
    mask = (rows < N)[:, None] & (offs_k < K)[None, :]
    if PREFETCH:
        w = tl.load(w_ptr + rows.to(tl.int64)[:, None] * K + offs_k[None, :], mask=mask, other=0.0)
    if PDL:
        tl.extra.cuda.gdc_wait()
    if not PREFETCH:
        w = tl.load(w_ptr + rows.to(tl.int64)[:, None] * K + offs_k[None, :], mask=mask, other=0.0)
    x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    acc = tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.bfloat16), mask=rows < N)


def triton_gemv(x, w, y, BN, pdl, prefetch, num_warps):
    N, K = w.shape
    _gemv[(triton.cdiv(N, BN),)](x, w, y, N, K=K, KP2=triton.next_power_of_2(K), BN=BN, PDL=pdl, PREFETCH=prefetch,
                                 launch_pdl=pdl, num_warps=num_warps)


SHAPES = {  # name: (N, K)
    "gdn in_proj": (7936, 960),
    "gdn o_proj": (960, 1536),
    "gdn f_b": (1536, 128),
    "moe w_in": (1920, 960),
    "moe w_out": (960, 1152),
    "mla q": (1344, 960),
    "mla kv_a": (352, 960),
    "mla [kv_a|q|g]": (2656, 960),
    "engram kv": (4800, 960),
    "lm head": (100352, 960),
}


def bench_gemv():
    """Each GEMV cycles through enough weight copies (>= 300 MB) that the weights come from DRAM, not L2, as in a
    decode step (~1.1 GB of weights)."""
    torch.manual_seed(0)
    for name, (N, K) in SHAPES.items():
        mb = N * K * 2 / 1e6
        copies = max(2, int(300 / mb) + 1)
        ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x.float(), ws[0].float())
        y = torch.empty(1, N, device="cuda", dtype=torch.bfloat16)

        def chain(f):
            return lambda: [f(w) for w in ws]

        t_cublas = graph_time_us(chain(lambda w: torch.mm(x, w.t(), out=y)), 1) / copies
        line = f"{name:16s} {N:6d}x{K:5d} {mb:7.2f} MB  cuBLAS {t_cublas:7.2f} us ({mb / t_cublas * 1e3:5.0f} GB/s)"
        best = ({}, {})
        for BN in (2, 4, 8, 16, 32):
            for nw in (2, 4, 8):
                if BN * triton.next_power_of_2(K) > 32768:
                    continue
                for pdl, pre in ((False, False), (True, True)):
                    triton_gemv(x, ws[0], y, BN, pdl, pre, nw)
                    err = (y.float() - ref).abs().max().item()
                    assert err < 0.05, (name, BN, nw, err)
                    t = graph_time_us(chain(lambda w: triton_gemv(x, w, y, BN, pdl, pre, nw)), 1) / copies
                    if pdl not in best[0] or t < best[0][pdl]:
                        best[0][pdl] = t
                        best[1][pdl] = (BN, nw)
        for pdl in (False, True):
            t = best[0][pdl]
            line += f" | triton{'+pdl' if pdl else ''} {t:7.2f} us ({mb / t * 1e3:5.0f} GB/s) {best[1][pdl]}"
        print(line, flush=True)
        del ws


def bench_boundary():
    """mHC boundary (post of one sublayer + pre of the next): the fused kernel's two variants vs today's three
    kernels (post, theta GEMM, pre). 600 distinct theta matrices (110 MB) so theta comes from DRAM as in a step."""
    import nenya_vllm.ops.mhc_boundary as mb
    from nenya_vllm.layers.mhc import NenyaMHC
    from nenya_vllm.ops.mhc_kernels import mhc_post

    torch.manual_seed(0)
    mods = []
    for _ in range(600):
        m = NenyaMHC(4, 960, 20).cuda()
        with torch.no_grad():
            m.theta.weight.normal_(0, 0.02)
            m.beta_res.normal_(0, 3)
        m.finalize()
        mods.append(m)
    w = torch.rand(960, device="cuda") + 0.5
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096):
        x = (torch.randn(T, 4, 960, device="cuda") * 2).to(torch.bfloat16)
        layer = torch.randn(T, 960, device="cuda").to(torch.bfloat16)
        _, _, coef = mods[0].pre(x, w, 1e-5)
        n = 600 if T <= 256 else 60
        res = {}

        def unfused():
            for m in mods[:n]:
                xn = mhc_post(x, layer, coef)
                m.pre(xn, w, 1e-5)
        res["unfused"] = graph_time_us(unfused, 1) / n
        for name, thr in (("small", 1 << 30), ("large", 0)):
            mb.SMALL_T = thr

            def fused():
                for m in mods[:n]:
                    mb.mhc_boundary(x, layer, coef, m, w, 1e-5)
            res[name] = graph_time_us(fused, 1) / n
        mb.SMALL_T = 512
        print(f"T {T:5d}: " + "  ".join(f"{k} {v:7.2f} us" for k, v in res.items()), flush=True)


def bench_boundary_bt():
    """the fused boundary kernel's tokens per block (1, 2, 4) and warps at 4..128 tokens"""
    import nenya_vllm.ops.mhc_boundary as mb
    from nenya_vllm.layers.mhc import NenyaMHC

    torch.manual_seed(0)
    mods = []
    for _ in range(600):
        m = NenyaMHC(4, 960, 20).cuda()
        with torch.no_grad():
            m.theta.weight.normal_(0, 0.02)
            m.beta_res.normal_(0, 3)
        m.finalize()
        mods.append(m)
    w = torch.rand(960, device="cuda") + 0.5
    mb.SMALL_T = 1 << 30
    for T in (4, 8, 16, 32, 64, 128, 256):
        x = (torch.randn(T, 4, 960, device="cuda") * 2).to(torch.bfloat16)
        layer = torch.randn(T, 960, device="cuda").to(torch.bfloat16)
        _, _, coef = mods[0].pre(x, w, 1e-5)
        res = {}
        for bt in (1, 2, 4):
            for nw in (4, 8):
                mb.BT_FIXED, mb.WARPS = bt, nw

                def fused():
                    for m in mods:
                        mb.mhc_boundary(x, layer, coef, m, w, 1e-5)
                res[f"bt{bt}w{nw}"] = graph_time_us(fused, 1) / len(mods)
        mb.BT_FIXED, mb.WARPS = 0, 4
        print(f"T {T:5d}: " + "  ".join(f"{k} {v:6.2f}" for k, v in res.items()), flush=True)


def bench_gdn_in():
    """gdn2_in_proj's tensor-core path (GEMM + conv epilogue): tiles x split-K at 8..64 tokens, vs cuBLAS alone"""
    import nenya_vllm.ops.gdn2_decode as gd

    torch.manual_seed(0)
    NOUT, D, NCONV = 7936, 960, 4608
    copies = 22
    ws = [torch.randn(NOUT, D, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    cw = torch.randn(NCONV, 4, device="cuda") * 0.3
    for T in [int(t) for t in os.environ.get("TS", "8,16,32,64").split(",")]:
        x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
        conv = torch.randn(T + 1, 3, NCONV, device="cuda").to(torch.bfloat16).transpose(-1, -2)
        idx = torch.arange(1, T + 1, device="cuda", dtype=torch.int32)
        y = torch.empty(T, NOUT, device="cuda", dtype=torch.bfloat16)
        t_cublas = graph_time_us(lambda: [torch.mm(x, w.t(), out=y) for w in ws], 1) / copies
        res = []
        for BN in (32, 64):
            for BK in (64, 128):
                for sk in (1, 2, 3, 4, 6):
                    gd.IN_PROJ_CONFIG = f"{BN},{BK},{3 if BK == 64 else 2},{sk}"
                    try:
                        t = graph_time_us(lambda: [gd.gdn2_in_proj(x, w, conv, cw, idx, T) for w in ws], 1, replays=20)
                    except Exception as e:
                        print("fail", gd.IN_PROJ_CONFIG, str(e)[:100])
                        continue
                    res.append((t / copies, gd.IN_PROJ_CONFIG))
        gd.IN_PROJ_CONFIG = ""
        t_def = graph_time_us(lambda: [gd.gdn2_in_proj(x, w, conv, cw, idx, T) for w in ws], 1) / copies
        res.sort()
        print(f"gdn in_proj T {T:3d}: cublas {t_cublas:6.2f}  default {t_def:6.2f}  best {res[0][0]:6.2f} {res[0][1]} "
              f"({NOUT * D * 2 / (res[0][0] * 1e3):.0f} GB/s)  2nd {res[1][0]:6.2f} {res[1][1]}  3rd {res[2][0]:6.2f} "
              f"{res[2][1]}", flush=True)


def bench_moe_in():
    """moe_in_route (w_in + fp32 router + top-k): router tokens per program x w_in split-K target, 8..64 tokens"""
    import nenya_vllm.ops.moe_small as ms

    torch.manual_seed(0)
    copies = 30
    gws = [torch.randn(120, 960, device="cuda") * 0.05 for _ in range(copies)]
    wins = [(torch.randn(1920, 960, device="cuda") * 0.03).to(torch.bfloat16) for _ in range(copies)]
    bias = torch.randn(120, device="cuda") * 0.1
    for T in [int(t) for t in os.environ.get("TS", "8,16,32,64").split(",")]:
        x = torch.randn(T, 960, device="cuda")
        res = {}
        for tbr in (1, 2, 4, 8):
            for prog in (1, 340):
                ms.MOE_TBR, ms.MOE_IN_PROGRAMS = tbr, prog
                res[f"tbr{tbr}/{'split' if prog > 1 else 'nosplit'}"] = graph_time_us(
                    lambda: [ms.moe_in_route(x, g, bias, w, 6, 384) for g, w in zip(gws, wins)], 1) / copies
        ms.MOE_TBR, ms.MOE_IN_PROGRAMS = 0, 340
        best = min((v, k) for k, v in res.items())
        print(f"T {T:3d}: " + "  ".join(f"{k} {v:6.2f}" for k, v in res.items()) + f"   best {best[1]}", flush=True)


def bench_gdn_varlen():
    """the short-prefill GDN-2 path (prep + sequential recurrence) per layer: value tile x warps, for one sequence of
    32..1024 tokens and for batches of sequences"""
    import nenya_vllm.ops.gdn2_decode as gd

    torch.manual_seed(0)
    H, K = 12, 128
    KD = H * K
    for lens in ([32], [128], [512], [1024], [128] * 8, [16] * 32):
        N, T = len(lens), sum(lens)
        proj = torch.randn(T, 5 * KD + 2 * K, device="cuda").to(torch.bfloat16)
        f = (torch.randn(T, KD, device="cuda") * 0.5).to(torch.bfloat16)
        conv = torch.randn(N + 2, 3, 3 * KD, device="cuda").to(torch.bfloat16).transpose(-1, -2)
        cw = torch.randn(3 * KD, 4, device="cuda") * 0.5
        ssm = torch.randn(N + 2, H, K, K, device="cuda") * 0.1
        A_log, dtb = torch.rand(H, device="cuda") * 2 - 1, torch.randn(KD, device="cuda") * 0.5
        qsl = torch.zeros(N + 1, dtype=torch.int32, device="cuda")
        qsl[1:] = torch.cumsum(torch.tensor(lens, device="cuda"), 0)
        idx = torch.arange(1, N + 1, dtype=torch.int32, device="cuda")
        has_init = torch.ones(N, dtype=torch.bool, device="cuda")
        core = torch.empty(T, KD, dtype=torch.bfloat16, device="cuda")
        res = {}
        for cfg in ((32, 4), (32, 2), (16, 4), (16, 2), (16, 1), (8, 2), (8, 1), (64, 4), (64, 8)):
            gd.VARLEN_CFG = cfg
            try:
                res["%d,%d" % cfg] = graph_time_us(lambda: gd.gdn2_prefill_short(
                    proj, f, conv, cw, ssm, qsl, idx, has_init, A_log, dtb, N, H, K, core), 1, replays=20)
            except Exception as e:
                print("  skip", cfg, str(e)[:80])
        gd.VARLEN_CFG = ()
        best = min((v, k) for k, v in res.items())
        print(f"lens {str(lens[:2]) + ('x%d' % N if N > 2 else ''):14s}: " + "  ".join(f"{k} {v:7.1f}" for k, v in res.items())
              + f"   best {best[1]}", flush=True)


def bench_moe_grouped():
    """moe_experts_grouped (9..64 tokens): up / down kernel tiles, routed like a real batch (6 of 120 experts per
    token, skewed: a Zipf-like preference), 30 layers' worth of weights cycled out of L2"""
    import nenya_vllm.ops.moe_small as ms

    torch.manual_seed(0)
    E, H, LAT, HS, K = 120, 384, 384, 768, 6
    copies = 8
    w13s = [(torch.randn(E, 2 * H, LAT, device="cuda") * 0.05).to(torch.bfloat16) for _ in range(copies)]
    w2s = [(torch.randn(E, LAT, H, device="cuda") * 0.05).to(torch.bfloat16) for _ in range(copies)]
    pref = 1.0 / torch.arange(1, E + 1, device="cuda").float() ** 0.6
    for T in [int(t) for t in os.environ.get("TS", "9,16,32,64").split(",")]:
        lat_in = torch.randn(T, LAT, device="cuda").to(torch.bfloat16)
        sh = torch.randn(T, 2 * HS, device="cuda").to(torch.bfloat16)
        ids = torch.stack([torch.multinomial(pref, K) for _ in range(T)]).to(torch.int32)
        wts = torch.softmax(torch.randn(T, K, device="cuda"), -1)
        res = {}
        for up in ((32, 128, 4), (32, 64, 4), (16, 128, 4)):
            for down in ((32, 128, 4), (16, 128, 4), (64, 128, 4), (64, 128, 8), (32, 64, 4), (16, 64, 2)):
                ms.GROUPED_UP_CFG, ms.GROUPED_DOWN_CFG = up, down
                try:
                    t = graph_time_us(lambda: [ms.moe_experts_grouped(lat_in, sh, ids, wts, a, b, False)
                                               for a, b in zip(w13s, w2s)], 1, replays=20) / copies
                except Exception as e:
                    print("  skip", up, down, str(e)[:60])
                    continue
                res[(up, down)] = t
        ms.GROUPED_UP_CFG, ms.GROUPED_DOWN_CFG = (32, 128, 4), (64, 128, 4)
        best = sorted(res.items(), key=lambda kv: kv[1])[:4]
        print(f"T {T:3d}: default {res.get(((32, 128, 4), (64, 128, 4)), float('nan')):6.2f} us  best: "
              + "  ".join(f"{v:6.2f} up{k[0]} down{k[1]}" for k, v in best), flush=True)


def bench_gemm():
    """decode_linear's tensor-core path vs its GEMV path vs cuBLAS at 8..64 tokens (weights cycled out of L2)."""
    from nenya_vllm.ops import decode_gemv as dg

    torch.manual_seed(0)
    for name, (N, K) in SHAPES.items():
        if N > 50000:
            continue
        mb = N * K * 2 / 1e6
        copies = max(2, int(300 / mb) + 1)
        ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        for T in (8, 16, 32, 64):
            x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
            y = torch.empty(T, N, device="cuda", dtype=torch.bfloat16)
            res = {"cublas": graph_time_us(lambda: [torch.mm(x, w.t(), out=y) for w in ws], 1) / copies}
            for label, thr in (("mma", 5), ("gemv", 1000)):
                dg.GEMM_MIN_T = thr
                res[label] = graph_time_us(lambda: [dg.decode_linear(x, w, out=y) for w in ws], 1) / copies
            dg.GEMM_MIN_T = 5
            print(f"{name:16s} T {T:3d}: " + "  ".join(f"{k} {v:6.2f} us" for k, v in res.items()), flush=True)
        del ws


def bench_gemm_tune():
    """sweep the tensor-core path's tiles per shape and token count"""
    from nenya_vllm.ops import decode_gemv as dg

    torch.manual_seed(0)
    for name, (N, K) in SHAPES.items():
        if N > 50000:
            continue
        mb = N * K * 2 / 1e6
        copies = max(2, int(300 / mb) + 1)
        ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        for T in (8, 16, 32, 64):
            x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
            y = torch.empty(T, N, device="cuda", dtype=torch.bfloat16)
            t_cublas = graph_time_us(lambda: [torch.mm(x, w.t(), out=y) for w in ws], 1) / copies
            res = []
            for BN in (16, 32, 64, 128):
                for BK in (64, 128):
                    for nw in (4, 8):
                        for st in (2, 3):
                            cfg = (BN, BK, nw, st)
                            try:
                                t = graph_time_us(lambda: [dg.decode_linear(x, w, out=y, config=cfg) for w in ws], 1)
                            except Exception:
                                continue
                            res.append((t / copies, cfg))
            res.sort()
            print(f"{name:16s} T {T:3d}: cublas {t_cublas:6.2f}  best {res[0][0]:6.2f} {res[0][1]}  "
                  f"2nd {res[1][0]:6.2f} {res[1][1]}", flush=True)
        del ws


def bench_gemm_split():
    """split-K sweep of the tensor-core path per decode shape and token count (with the prologue each call site
    uses), against cuBLAS on the plain product"""
    from nenya_vllm.ops import decode_gemv as dg

    torch.manual_seed(0)
    # name: (N, K, prologue, group)
    shapes = {"gdn o_proj (gated norm)": (960, 1536, dg.GATED_RMSNORM, 128), "mla out (sigmoid gate)": (960, 1536, dg.SIGMOID_GATE, 1),
              "moe w_out": (960, 1152, dg.NONE, 1), "mla [kv_a|q|g]": (2656, 960, dg.NONE, 1),
              "mlp down (silu mul)": (960, 2560, dg.SILU_MUL, 1), "mlp gate_up": (5120, 960, dg.NONE, 1),
              "moe w_in": (1920, 960, dg.NONE, 1), "gdn in_proj": (7936, 960, dg.NONE, 1),
              "moe w_out (f32 input)": (960, 1152, dg.F32, 1), "engram kv (gather)": (4800, 960, dg.GATHER, 96)}
    only = os.environ.get("SHAPES")
    for name, (N, K, pro, grp) in shapes.items():
        if only and not any(o in name for o in only.split(",")):
            continue
        mb = N * K * 2 / 1e6
        copies = max(2, int(300 / mb) + 1)
        ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        nw_ = torch.rand(grp, device="cuda") + 0.5
        for T in [int(t) for t in os.environ.get("TS", "8,16,32,64").split(",")]:
            if pro == dg.F32:
                xin = torch.randn(T, K, device="cuda")
            elif pro == dg.GATHER:
                xin = torch.randint(0, 20000, (T, K // grp), device="cuda")
            else:
                xin = torch.randn(T, 2 * K if pro == dg.SILU_MUL else K, device="cuda", dtype=torch.bfloat16)
            aux = (torch.randn(20000, grp, device="cuda", dtype=torch.bfloat16) if pro == dg.GATHER
                   else torch.randn(T, K, device="cuda", dtype=torch.bfloat16))
            y = torch.empty(T, N, device="cuda", dtype=torch.bfloat16)
            xc = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
            t_cublas = graph_time_us(lambda: [torch.mm(xc, w.t(), out=y) for w in ws], 1) / copies
            kw = dict(prologue=pro, aux=aux, norm_weight=nw_, eps=1e-5, group=grp, out=y)
            res = []
            cur = graph_time_us(lambda: [dg.decode_linear(xin, w, **kw, config=dg._gemm_config(N, K, T, pro)[:4] + (1,))
                                         for w in ws], 1) / copies
            auto = graph_time_us(lambda: [dg.decode_linear(xin, w, **kw) for w in ws], 1) / copies   # the table
            for BN in (16, 32, 64):
                for BK in ((128,) if pro == dg.GATED_RMSNORM else (64, 128)):
                    for nw in (4,):
                        for st in ((3,) if BK == 64 else (2,)):
                            for sk in (1, 2, 3, 4, 6, 8):
                                cfg = (BN, BK, nw, st, sk)
                                if sk > (K + BK - 1) // BK:
                                    continue
                                try:
                                    t = graph_time_us(lambda: [dg.decode_linear(xin, w, **kw, config=cfg) for w in ws],
                                                      1, replays=20)
                                except Exception:
                                    continue
                                res.append((t / copies, cfg))
            res.sort()
            gbs = N * K * 2 / (res[0][0] * 1e3)
            print(f"{name:24s} T {T:3d}: cublas {t_cublas:6.2f}  nosplit {cur:6.2f}  table {auto:6.2f}  best {res[0][0]:6.2f} "
                  f"{res[0][1]} ({gbs:.0f} GB/s)  2nd {res[1][0]:6.2f} {res[1][1]}  3rd {res[2][0]:6.2f} {res[2][1]}",
                  flush=True)
        del ws


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["chain", "gemv", "boundary", "gemm", "gemm_tune", "gemm_split", "boundary_bt", "gdn_in",
                                        "moe_in", "gdn_varlen", "moe_grouped"])
    args = ap.parse_args()
    {"chain": bench_chain, "gemv": bench_gemv, "boundary": bench_boundary, "gemm": bench_gemm,
     "gemm_tune": bench_gemm_tune, "gemm_split": bench_gemm_split,
     "boundary_bt": bench_boundary_bt, "gdn_in": bench_gdn_in,
     "moe_in": bench_moe_in, "gdn_varlen": bench_gdn_varlen, "moe_grouped": bench_moe_grouped}[args.what]()
