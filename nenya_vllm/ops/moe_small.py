# SPDX-License-Identifier: Apache-2.0
# Nenya's MoE for decode-sized batches, after the Metal engine's layout (edge_device_optimisations/deep_model):
#   1. [expert_down | shared fc1 | shared fc2] GEMV (bf16) and the router GEMV (fp32) in one kernel; per token,
#      the last router program to finish takes the balancer-biased top-6 and the softmax of the chosen logits
#   2. gate/up + SiLU-mul over the 6 chosen experts, and the shared experts' SiLU-mul                 one kernel
#   3. down projection of the 6 experts + routing-weighted combine                                   one kernel
#   4. [expert_up | shared fc3] GEMV over the concatenated input (ops/decode_gemv.py)
# instead of the grouped path's router (2 cuBLAS kernels) + route + 5 expert kernels + 4 GEMMs + an add. All four run
# under programmatic dependent launch. The experts compute in fp32 from bf16 weights and round once, where the release
# rounds after every bf16 matmul: the same maths with less rounding. Each (token, expert) pair reads its expert's
# weights, so this pays only for a few tokens; larger batches take the grouped kernels (layers/moe.py picks by batch
# size).

import os

import torch

from vllm.triton_utils import tl, triton

from nenya_vllm.ops.decode_gemv import PDL, pdl, counters, repeats


@triton.jit
def _route_store(logits, bias, offs, mask, weights_ptr, ids_ptr, row, K: tl.constexpr):
    """Nenya's routing of one token's fp32 router logits [BLOCK_E]: choose by logits + balancer bias, weight by the
    softmax of the chosen *unbiased* logits. Each expert's rank in (biased desc, index asc) order comes from one
    [E, E] comparison -- the experts of rank < K are exactly the sequential argmax picks of
    small_kernels._moe_route_kernel, in the same order -- and the chosen ones are scattered to their slots (3
    reductions instead of 3 K). Non-finite logits count as -1e30, so garbage padding rows stay valid."""
    logits = tl.where((logits == logits) & (tl.abs(logits) < 3.0e38), logits, -1.0e30)
    biased = tl.where(mask, logits + bias, float("-inf"))
    beats = (biased[None, :] > biased[:, None]) | ((biased[None, :] == biased[:, None]) & (offs[None, :] < offs[:, None]))
    rank = tl.sum(beats.to(tl.int32), axis=1)
    sel = mask & (rank < K)
    mx = tl.max(tl.where(sel, logits, -1.0e30), axis=0)
    ex = tl.where(sel, tl.exp(logits - mx), 0.0)
    tl.store(weights_ptr + row * K + rank, ex / tl.sum(ex, axis=0), mask=sel)
    tl.store(ids_ptr + row * K + rank, offs.to(tl.int32), mask=sel)


@triton.jit
def _moe_in_kernel(x_ptr, x_stride,              # [T, D] fp32 normalised input (the router's; bf16 of it for w_in)
                   gw_ptr, bias_ptr,             # [E, D] fp32 router weight, [E] fp32 balancer bias
                   win_ptr,                      # [LAT + NSH, D] bf16 [expert_down | shared fc1 | shared fc2]
                   lat_ptr,                      # [T, LAT] bf16 out: the routed experts' input
                   sh_ptr,                       # [T, NSH] bf16 out: the shared experts' gate | up
                   logits_ptr,                   # [T, E] fp32 out: router logits
                   weights_ptr, ids_ptr,         # [T, K] fp32 / int32 out
                   cnt_ptr, T,
                   ws_ptr, wcnt_ptr,             # split-K partials [SPLITK, M_PAD, NIN] fp32 / per-tile counters
                   E: tl.constexpr, D: tl.constexpr, DP2: tl.constexpr, LAT: tl.constexpr, NSH: tl.constexpr,
                   K: tl.constexpr, BE: tl.constexpr, NEB: tl.constexpr, TBR: tl.constexpr, BN: tl.constexpr,
                   TB: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, MMA: tl.constexpr, KS: tl.constexpr,
                   SPLITK: tl.constexpr, M_PAD: tl.constexpr, BLOCK_E: tl.constexpr, PDL: tl.constexpr):
    pid = tl.program_id(0)
    n_router = tl.cdiv(T, TBR) * NEB
    NIN: tl.constexpr = LAT + NSH
    offs_d = tl.arange(0, DP2)
    mask_d = offs_d < D
    if pid < n_router:
        # router: BE experts of TBR tokens (the weight block read once), fp32 products and accumulation over D
        t0r = (pid // NEB) * TBR
        rows_r = (pid % NEB) * BE + tl.arange(0, BE)
        mask_rr = rows_r < E
        w_r = tl.load(gw_ptr + rows_r[:, None] * D + offs_d[None, :], mask=mask_rr[:, None] & mask_d[None, :],
                      other=0.0)
        bias_r = tl.load(bias_ptr + tl.arange(0, BLOCK_E), mask=tl.arange(0, BLOCK_E) < E, other=0.0)
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        for i in tl.static_range(TBR):
            tr = t0r + i
            if tr < T:
                x_r = tl.load(x_ptr + tr.to(tl.int64) * x_stride + offs_d, mask=mask_d, other=0.0)
                tl.store(logits_ptr + tr * E + rows_r, tl.sum(w_r * x_r[None, :], axis=1), mask=mask_rr)
        tl.debug_barrier()
        for i in tl.static_range(TBR):
            tr = t0r + i
            if tr < T:
                done = tl.atomic_add(cnt_ptr + tr, 1, sem="acq_rel")
                if done == NEB - 1:
                    # the token's last router program: top-k + softmax
                    tl.store(cnt_ptr + tr, 0)
                    offs = tl.arange(0, BLOCK_E)
                    mask = offs < E
                    lg = tl.load(logits_ptr + tr * E + offs, mask=mask, other=0.0, cache_modifier=".cg")
                    _route_store(lg, bias_r, offs, mask, weights_ptr, ids_ptr, tr, K)
    else:
        # [expert_down | fc1 | fc2]: the experts' input is the bf16 rounding of the normalised input (the boundary's
        # bf16 output); output rows < LAT go to lat, the rest to sh (BN divides LAT)
        q = pid - n_router
        NB: tl.constexpr = (NIN + BN - 1) // BN
        nb = q % NB
        rows_w = nb * BN + tl.arange(0, BN)
        mask_rw = rows_w < NIN
        to_lat = nb * BN < LAT
        if MMA:
            # program (column block, token block, K slice); split-K partials summed in slice order by the tile's
            # last program
            mb = (q // NB) % (M_PAD // BM)
            ks = q // (NB * (M_PAD // BM))
            k_lo = ks * KS
            k_hi = tl.minimum(k_lo + KS, D)
            offs_m = mb * BM + tl.arange(0, BM)
            mask_m = offs_m < T
            m64 = offs_m.to(tl.int64)[:, None]
            wrow = win_ptr + rows_w.to(tl.int64)[:, None] * D
            offs_k0 = k_lo + tl.arange(0, BK)
            w0 = tl.load(wrow + offs_k0[None, :], mask=mask_rw[:, None] & (offs_k0 < k_hi)[None, :], other=0.0)
            if PDL:
                tl.extra.cuda.gdc_wait()
                tl.extra.cuda.gdc_launch_dependents()
            x0 = tl.load(x_ptr + m64 * x_stride + offs_k0[None, :], mask=mask_m[:, None] & (offs_k0 < k_hi)[None, :],
                         other=0.0)
            acc = tl.dot(x0.to(tl.bfloat16), tl.trans(w0))
            for k0 in tl.range(k_lo + BK, k_hi, BK, num_stages=3):
                offs_k = k0 + tl.arange(0, BK)
                mask_k = offs_k < k_hi
                w = tl.load(wrow + offs_k[None, :], mask=mask_rw[:, None] & mask_k[None, :], other=0.0)
                x = tl.load(x_ptr + m64 * x_stride + offs_k[None, :], mask=mask_m[:, None] & mask_k[None, :],
                            other=0.0)
                acc = tl.dot(x.to(tl.bfloat16), tl.trans(w), acc)
            last = True
            if SPLITK > 1:
                wsp = ws_ptr + m64 * NIN + rows_w[None, :]
                tl.store(wsp + ks * (M_PAD * NIN), acc, mask=mask_rw[None, :])
                tl.debug_barrier()
                tile = mb * NB + nb
                last = tl.atomic_add(wcnt_ptr + tile, 1, sem="acq_rel") == SPLITK - 1
                if last:
                    acc = tl.load(wsp, mask=mask_rw[None, :], other=0.0, cache_modifier=".cg")
                    for s in tl.static_range(1, SPLITK):
                        acc += tl.load(wsp + s * (M_PAD * NIN), mask=mask_rw[None, :], other=0.0,
                                       cache_modifier=".cg")
                    tl.store(wcnt_ptr + tile, 0)
            if last:
                out = acc.to(tl.bfloat16)
                mo = mask_m[:, None] & mask_rw[None, :]
                if to_lat:
                    tl.store(lat_ptr + m64 * LAT + rows_w[None, :], out, mask=mo)
                else:
                    tl.store(sh_ptr + m64 * NSH + (rows_w - LAT)[None, :], out, mask=mo)
        else:
            t0 = (q // NB) * TB
            w_w = tl.load(win_ptr + rows_w.to(tl.int64)[:, None] * D + offs_d[None, :],
                          mask=mask_rw[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            if PDL:
                tl.extra.cuda.gdc_wait()
                tl.extra.cuda.gdc_launch_dependents()
            for i in tl.static_range(TB):
                t = t0 + i
                if t < T:
                    x_w = tl.load(x_ptr + t.to(tl.int64) * x_stride + offs_d, mask=mask_d, other=0.0)
                    x_w = x_w.to(tl.bfloat16).to(tl.float32)
                    acc_w = tl.sum(w_w * x_w[None, :], axis=1).to(tl.bfloat16)
                    if to_lat:
                        tl.store(lat_ptr + t.to(tl.int64) * LAT + rows_w, acc_w, mask=mask_rw)
                    else:
                        tl.store(sh_ptr + t.to(tl.int64) * NSH + rows_w - LAT, acc_w, mask=mask_rw)


# router tokens per program and w_in split-K target (NENYA_MOE_TBR, NENYA_MOE_IN_PROGRAMS)
MOE_TBR = int(os.environ.get("NENYA_MOE_TBR", "0"))
MOE_IN_PROGRAMS = int(os.environ.get("NENYA_MOE_IN_PROGRAMS", "340"))


def moe_in_route(x_f32: torch.Tensor, gate_weight: torch.Tensor, bias: torch.Tensor, w_in: torch.Tensor, k: int,
                 lat: int):
    """lat_in = bf16(bf16(x) @ expert_down^T) [T, lat], sh = bf16(bf16(x) @ [fc1 | fc2]^T) [T, NIN - lat], the fp32
    router logits [T, E] and Nenya's routing of them (weights fp32 [T, k], ids int32 [T, k]), in one kernel."""
    T, D = x_f32.shape
    E = gate_weight.shape[0]
    NIN = w_in.shape[0]
    dev = x_f32.device
    lat_in = torch.empty((T, lat), dtype=torch.bfloat16, device=dev)
    sh = torch.empty((T, NIN - lat), dtype=torch.bfloat16, device=dev)
    logits = torch.empty((T, E), dtype=torch.float32, device=dev)
    weights = torch.empty((T, k), dtype=torch.float32, device=dev)
    ids = torch.empty((T, k), dtype=torch.int32, device=dev)
    if T:
        from nenya_vllm.ops.decode_gemv import GEMM_MIN_T

        mma = T >= GEMM_MIN_T
        BE = 8
        NEB = triton.cdiv(E, BE)
        TBR = MOE_TBR or (2 if T > 32 else 1)      # measured: one token per router program up to 32 tokens
        splitk, KS, M_PAD = 1, D, 1
        if mma:
            BM = 16 if T <= 16 else 32 if T <= 32 else 64
            BN, BK, TB = (16, 128, 1) if T <= 32 else (32, 64, 1)
            M_PAD = triton.cdiv(T, BM) * BM
            n_tiles = triton.cdiv(NIN, BN) * (M_PAD // BM)
            # measured (benchmarks/bench_kernels.py moe_in): splitting w_in pays from 16 tokens
            splitk = max(1, min(triton.cdiv(D, BK), round(MOE_IN_PROGRAMS / n_tiles))) if T >= 16 else 1
            KS = triton.cdiv(triton.cdiv(D, BK), splitk) * BK
            splitk = triton.cdiv(D, KS)
            n_w = n_tiles * splitk
        else:
            BM, BN, BK, TB = 16, 4, 64, min(T, 8)
            n_w = triton.cdiv(NIN, BN) * triton.cdiv(T, TB)
        assert lat % BN == 0
        ws = torch.empty((splitk, M_PAD, NIN), dtype=torch.float32, device=dev) if splitk > 1 else logits
        wcnt = counters(n_w, dev, "moe_in_split") if splitk > 1 else logits
        for _ in range(repeats("moe_in")):
            _moe_in_kernel[(triton.cdiv(T, TBR) * NEB + n_w,)](
                x_f32, x_f32.stride(0), gate_weight, bias, w_in, lat_in, sh, logits, weights, ids,
                counters(T, dev, "moe_route"), T, ws, wcnt, E=E, D=D, DP2=triton.next_power_of_2(D), LAT=lat,
                NSH=NIN - lat, K=k, BE=BE, NEB=NEB, TBR=TBR, BN=BN, TB=TB, BM=BM, BK=BK, MMA=mma, KS=KS,
                SPLITK=splitk, M_PAD=M_PAD, BLOCK_E=triton.next_power_of_2(E), PDL=pdl("moe_in"), launch_pdl=pdl("moe_in"),
                num_warps=8 if not mma else 4)
    return lat_in, sh, logits, weights, ids


@triton.jit
def _moe_up_kernel(lat_ptr, lat_stride,          # [T, LAT] bf16: the routed experts' input (expert_down(x))
                   ids_ptr,                       # [T, K] int32
                   w13_ptr,                       # [E, 2H, LAT] bf16 (vLLM's layout; halves gate|up or up|gate)
                   h_ptr,                         # [T * K, H] fp32: SiLU(gate) * up per (token, expert) pair
                   sh_ptr, sh_stride,             # [T, 2 HS] bf16: the shared experts' gate | up
                   buf_ptr, buf_stride,           # [T, LAT + HS] bf16: shared SiLU-mul goes to columns LAT..
                   P, T,
                   K: tl.constexpr, LAT: tl.constexpr, LATP2: tl.constexpr, H: tl.constexpr, HS: tl.constexpr,
                   GATE_FIRST: tl.constexpr, BN: tl.constexpr, BS: tl.constexpr, PDL: tl.constexpr):
    pid = tl.program_id(0)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    NB: tl.constexpr = H // BN
    n_routed = P * NB
    if pid < n_routed:
        p = pid // NB
        nb = pid % NB
        tr = (p // K).to(tl.int64)
        e = tl.load(ids_ptr + p).to(tl.int64)
        offs_n = nb * BN + tl.arange(0, BN)
        if GATE_FIRST:
            gate_rows = offs_n
            up_rows = offs_n + H
        else:
            gate_rows = offs_n + H
            up_rows = offs_n
        base = w13_ptr + e * (2 * H * LAT)
        offs_k = tl.arange(0, LATP2)
        mk = offs_k < LAT
        x = tl.load(lat_ptr + tr * lat_stride + offs_k, mask=mk, other=0.0).to(tl.float32)
        wg = tl.load(base + gate_rows[:, None] * LAT + offs_k[None, :], mask=mk[None, :], other=0.0).to(tl.float32)
        wu = tl.load(base + up_rows[:, None] * LAT + offs_k[None, :], mask=mk[None, :], other=0.0).to(tl.float32)
        acc_g = tl.sum(wg * x[None, :], axis=1)
        acc_u = tl.sum(wu * x[None, :], axis=1)
        tl.store(h_ptr + p.to(tl.int64) * H + offs_n, acc_g * tl.sigmoid(acc_g) * acc_u)
    else:
        q = pid - n_routed
        idx = q * BS + tl.arange(0, BS)
        smask = idx < T * HS
        ts = (idx // HS).to(tl.int64)
        j = idx % HS
        g = tl.load(sh_ptr + ts * sh_stride + j, mask=smask, other=0.0).to(tl.float32)
        u = tl.load(sh_ptr + ts * sh_stride + HS + j, mask=smask, other=0.0).to(tl.float32)
        # vLLM's SiluAndMul: fp32 maths, one rounding
        tl.store(buf_ptr + ts * buf_stride + LAT + j, (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=smask)


@triton.jit
def _moe_down_kernel(h_ptr,                       # [T * K, H] fp32
                     ids_ptr, wts_ptr,            # [T, K] int32 / fp32
                     w2_ptr,                      # [E, LAT, H] bf16 (out, in)
                     buf_ptr, buf_stride,         # [T, LAT + HS] bf16: routed output goes to columns 0..LAT
                     K: tl.constexpr, KP2: tl.constexpr, LAT: tl.constexpr, H: tl.constexpr, HP2: tl.constexpr,
                     BN: tl.constexpr, PDL: tl.constexpr):
    # one program: BN output rows of one token over all K experts, every load issued at once
    t = tl.program_id(0).to(tl.int64)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    offs_k = tl.arange(0, KP2)
    kmask = offs_k < K
    e = tl.load(ids_ptr + t * K + offs_k, mask=kmask, other=0).to(tl.int64)
    wt = tl.load(wts_ptr + t * K + offs_k, mask=kmask, other=0.0)
    offs_j = tl.arange(0, HP2)
    mj = offs_j < H
    hv = tl.load(h_ptr + (t * K + offs_k)[:, None] * H + offs_j[None, :], mask=kmask[:, None] & mj[None, :],
                 other=0.0)
    w = tl.load(w2_ptr + e[:, None, None] * (LAT * H) + offs_n[None, :, None] * H + offs_j[None, None, :],
                mask=kmask[:, None, None] & mj[None, None, :], other=0.0).to(tl.float32)
    per_expert = tl.sum(w * hv[:, None, :], axis=2)                # [KP2, BN]
    acc = tl.sum(wt[:, None] * per_expert, axis=0)
    tl.store(buf_ptr + t * buf_stride + offs_n, acc.to(tl.bfloat16))


def moe_experts_small(lat_in: torch.Tensor, sh: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                      w13: torch.Tensor, w2: torch.Tensor, gate_first: bool, buf: torch.Tensor) -> None:
    """lat_in [T, lat] = expert_down(x), sh [T, 2 hs] = [shared gate | shared up] (bf16); writes buf [T, lat + hs]
    bf16 = [routed experts' combined output | shared SiLU(gate) * up]."""
    T, lat = lat_in.shape
    hs = sh.shape[1] // 2
    if T == 0:
        return
    K = ids.shape[1]
    H = w2.shape[2]
    P = T * K
    h = torch.empty((P, H), dtype=torch.float32, device=lat_in.device)
    BN, BS = 8, 1024
    grid = (P * (H // BN) + triton.cdiv(T * hs, BS),)
    for _ in range(repeats("moe_up")):
      _moe_up_kernel[grid](lat_in, lat_in.stride(0), ids, w13, h, sh, sh.stride(0), buf, buf.stride(0), P, T,
                         K=K, LAT=lat, LATP2=triton.next_power_of_2(lat), H=H, HS=hs, GATE_FIRST=gate_first, BN=BN,
                         BS=BS, PDL=pdl("moe_up"), launch_pdl=pdl("moe_up"), num_warps=4)
    BN_D = 4
    for _ in range(repeats("moe_down")):
      _moe_down_kernel[(T, lat // BN_D)](h, ids, weights, w2, buf, buf.stride(0), K=K, KP2=triton.next_power_of_2(K),
                                       LAT=lat, H=H, HP2=triton.next_power_of_2(H), BN=BN_D, PDL=pdl("moe_down"),
                                       launch_pdl=pdl("moe_down"), num_warps=8)


def shared_silu_mul(sh: torch.Tensor, lat: int, hs: int, buf: torch.Tensor) -> None:
    """Only the shared experts' SiLU(gate) * up (sh = [gate | up], [T, 2 hs]) into buf[:, lat:] (the grouped
    path)."""
    T = sh.shape[0]
    if T == 0:
        return
    BS = 1024
    dummy = torch.empty(0, device=sh.device)
    _moe_up_kernel[(triton.cdiv(T * hs, BS),)](sh, sh.stride(0), dummy, dummy, dummy, sh,
                                               sh.stride(0), buf, buf.stride(0), 0, T, K=1, LAT=lat,
                                               LATP2=triton.next_power_of_2(lat), H=16, HS=hs, GATE_FIRST=True,
                                               BN=16, BS=BS, PDL=False, num_warps=4)


# ---- 9..64 tokens: the routed experts grouped by expert ------------------------------------------------------------
# Each program owns one expert and a block of output columns: it finds its (token, slot) pairs in the routing itself
# (a comparison + prefix sum over all T*K ids: deterministic, ascending pair order, no sort kernel or atomics), reads
# the expert's weight tile once for all of them, and computes on the tensor cores. Pair outputs stay fp32; a rows
# kernel then adds each token's K weighted outputs in slot order next to the shared experts' SiLU-mul, and w_out
# reads that fp32 input as bf16 hi + residual (decode_gemv.F32). Replaces the grouped CUTLASS path's 8 launches
# (expert maps, row expansion, 2 GEMMs, finalize, copy, shared SiLU-mul, w_out) with 4, and its bf16 roundings of
# the hidden activations and the per-pair outputs with fp32.

@triton.jit
def _pairs_of_expert(ids_ptr, P, e, c, PP2: tl.constexpr, BM: tl.constexpr):
    """pairs c..c+BM-1 (in ascending pair order) of the pairs routed to expert e: (pair ids [BM], valid [BM], n_e)"""
    offs_p = tl.arange(0, PP2)
    ids = tl.load(ids_ptr + offs_p, mask=offs_p < P, other=-1)
    match = ids == e
    n_e = tl.sum(match.to(tl.int32), axis=0)
    pos = tl.cumsum(match.to(tl.int32), axis=0) - 1
    slot = c + tl.arange(0, BM)
    sel = match[None, :] & (pos[None, :] == slot[:, None])
    pairs = tl.sum(tl.where(sel, offs_p[None, :], 0), axis=1)
    return pairs, slot < n_e, n_e


@triton.jit
def _moe_grouped_up_kernel(lat_ptr, lat_stride,     # [T, LAT] bf16: the routed experts' input
                           ids_ptr,                  # [P] int32 expert per (token, slot) pair, P = T * K
                           w13_ptr,                  # [E, 2H, LAT] bf16
                           h_ptr,                    # [P, H] fp32 out: SiLU(gate) * up per pair
                           P,
                           K: tl.constexpr, LAT: tl.constexpr, H: tl.constexpr, PP2: tl.constexpr,
                           GATE_FIRST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           PDL: tl.constexpr):
    e = tl.program_id(0)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    if GATE_FIRST:
        gate_rows = offs_n
        up_rows = offs_n + H
    else:
        gate_rows = offs_n + H
        up_rows = offs_n
    base = w13_ptr + e.to(tl.int64) * (2 * H * LAT)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pairs0, valid0, n_e = _pairs_of_expert(ids_ptr, P, e, 0, PP2, BM)
    for c in range(0, n_e, BM):
        pairs, valid, n_c = _pairs_of_expert(ids_ptr, P, e, c, PP2, BM)
        tok = (pairs // K).to(tl.int64)
        acc_g = tl.zeros([BM, BN], dtype=tl.float32)
        acc_u = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in tl.static_range(0, LAT, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = tl.load(lat_ptr + tok[:, None] * lat_stride + offs_k[None, :], mask=valid[:, None], other=0.0)
            wg = tl.load(base + gate_rows[:, None] * LAT + offs_k[None, :])
            wu = tl.load(base + up_rows[:, None] * LAT + offs_k[None, :])
            acc_g = tl.dot(x, tl.trans(wg), acc_g)
            acc_u = tl.dot(x, tl.trans(wu), acc_u)
        h = acc_g * tl.sigmoid(acc_g) * acc_u
        tl.store(h_ptr + pairs.to(tl.int64)[:, None] * H + offs_n[None, :], h, mask=valid[:, None])


@triton.jit
def _moe_grouped_down_kernel(h_ptr,                 # [P, H] fp32
                             ids_ptr, wts_ptr,       # [P] int32 / fp32
                             w2_ptr,                 # [E, LAT, H] bf16
                             y_ptr,                  # [P, LAT] fp32 out: routing weight * expert output
                             P,
                             LAT: tl.constexpr, H: tl.constexpr, PP2: tl.constexpr, BM: tl.constexpr,
                             BN: tl.constexpr, BK: tl.constexpr, PDL: tl.constexpr):
    e = tl.program_id(0)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    base = w2_ptr + e.to(tl.int64) * (LAT * H)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pairs0, valid0, n_e = _pairs_of_expert(ids_ptr, P, e, 0, PP2, BM)
    for c in range(0, n_e, BM):
        pairs, valid, n_c = _pairs_of_expert(ids_ptr, P, e, c, PP2, BM)
        p64 = pairs.to(tl.int64)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in tl.static_range(0, H, BK):
            offs_k = k0 + tl.arange(0, BK)
            hv = tl.load(h_ptr + p64[:, None] * H + offs_k[None, :], mask=valid[:, None], other=0.0)
            hh = hv.to(tl.bfloat16)
            w = tl.load(base + offs_n[:, None] * H + offs_k[None, :])
            acc = tl.dot(hh, tl.trans(w), acc)
            acc = tl.dot((hv - hh.to(tl.float32)).to(tl.bfloat16), tl.trans(w), acc)
        wt = tl.load(wts_ptr + pairs, mask=valid, other=0.0)
        tl.store(y_ptr + p64[:, None] * LAT + offs_n[None, :], acc * wt[:, None], mask=valid[:, None])


@triton.jit
def _moe_out_rows_kernel(y_ptr,                     # [T * K, LAT] fp32 weighted pair outputs
                         sh_ptr, sh_stride,          # [T, 2 HS] bf16 shared gate | up
                         out_ptr,                    # [T, LAT + HS] fp32: w_out's input
                         K: tl.constexpr, LAT: tl.constexpr, HS: tl.constexpr, BLOCK: tl.constexpr,
                         PDL: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    routed = offs < LAT
    acc = tl.load(y_ptr + (t * K) * LAT + offs, mask=routed, other=0.0)
    for k in tl.static_range(1, K):
        acc += tl.load(y_ptr + (t * K + k) * LAT + offs, mask=routed, other=0.0)
    shared = (offs >= LAT) & (offs < LAT + HS)
    j = tl.where(shared, offs - LAT, 0)
    g = tl.load(sh_ptr + t * sh_stride + j, mask=shared, other=0.0).to(tl.float32)
    u = tl.load(sh_ptr + t * sh_stride + HS + j, mask=shared, other=0.0).to(tl.float32)
    x = tl.where(routed, acc, g * tl.sigmoid(g) * u)
    tl.store(out_ptr + t * (LAT + HS) + offs, x, mask=offs < LAT + HS)


def moe_experts_grouped(lat_in: torch.Tensor, sh: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                        w13: torch.Tensor, w2: torch.Tensor, gate_first: bool) -> torch.Tensor:
    """-> w_out's input [T, lat + hs] fp32: [routed experts' weighted sum | shared SiLU(gate) * up]."""
    T, lat = lat_in.shape
    hs = sh.shape[1] // 2
    K = ids.shape[1]
    E, H = w2.shape[0], w2.shape[2]
    P = T * K
    dev = lat_in.device
    out = torch.empty((T, lat + hs), dtype=torch.float32, device=dev)
    if T == 0:
        return out
    h = torch.empty((P, H), dtype=torch.float32, device=dev)
    y = torch.empty((P, lat), dtype=torch.float32, device=dev)
    PP2 = triton.next_power_of_2(P)
    BM = 16
    BN_U, BK_U, NW_U = GROUPED_UP_CFG
    BN_D, BK_D, NW_D = GROUPED_DOWN_CFG
    assert H % BN_U == 0 and lat % BN_D == 0 and lat % BK_U == 0 and H % BK_D == 0
    for _ in range(repeats("moe_gup")):
        _moe_grouped_up_kernel[(E, H // BN_U)](lat_in, lat_in.stride(0), ids, w13, h, P, K=K, LAT=lat, H=H, PP2=PP2,
                                               GATE_FIRST=gate_first, BM=BM, BN=BN_U, BK=BK_U, PDL=pdl("moe_gup"),
                                               launch_pdl=pdl("moe_gup"), num_warps=NW_U)
    for _ in range(repeats("moe_gdown")):
        _moe_grouped_down_kernel[(E, lat // BN_D)](h, ids, weights, w2, y, P, LAT=lat, H=H, PP2=PP2, BM=BM, BN=BN_D,
                                                   BK=BK_D, PDL=pdl("moe_gdown"), launch_pdl=pdl("moe_gdown"), num_warps=NW_D)
    BLOCK = 512
    _moe_out_rows_kernel[(T, triton.cdiv(lat + hs, BLOCK))](y, sh, sh.stride(0), out, K=K, LAT=lat, HS=hs,
                                                             BLOCK=BLOCK, PDL=pdl("moe_out"), launch_pdl=pdl("moe_out"), num_warps=4)
    return out


# (output columns per program, K chunk, warps) of the grouped up / down kernels
# (measured, benchmarks/bench_kernels.py moe_grouped: down 64-wide 8-15% faster than 32 at 9..64 tokens)
GROUPED_UP_CFG = tuple(int(v) for v in os.environ.get("NENYA_MOE_GUP_CFG", "32,128,4").split(","))
GROUPED_DOWN_CFG = tuple(int(v) for v in os.environ.get("NENYA_MOE_GDOWN_CFG", "64,128,4").split(","))
