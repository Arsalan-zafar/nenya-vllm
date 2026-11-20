# SPDX-License-Identifier: Apache-2.0
# Nenya's mHC boundary for decode-sized batches in one kernel (the Metal engine's "boundary"): the post of sublayer i
# (mix the streams, write the sublayer output into them) and the pre of sublayer i+1 (theta projection of the new
# streams, gates, read + RMSNorm), instead of three launches per sublayer (post kernel, theta GEMM, pre kernel).
#
# Same maths and rounding points as ops/mhc_kernels.py + the cuBLAS theta GEMM:
#   x_new_i = bf16( sum_j H_ij x_j + post_i * layer_out )              (stored; the next pre reads these values)
#   theta   = bf16( theta_W . bf16(flat(x_new)) )                       (fp32 accumulation, rounded like cuBLAS)
#   r       = rsqrt(mean(flat^2) + 1e-20);  m = theta * r
#   pre     = sigmoid(a_pre m[:n] + b_pre);  x = sum_i pre_i x_i -> RMSNorm (fp32 weight)
#   post    = 2 sigmoid(a_post m[n:2n] + b_post)
#   H       = Sinkhorn(a_res m[2n:] + b_res), 20 iterations, columns then rows
# Layout for decode-sized batches: grid (blocks of <= 4 tokens, 30 chunks of D); each program posts its chunk and
# computes partial theta dots (theta loaded before the PDL wait), the last program of a token block reduces them in a
# fixed order and does the gates, the read + RMSNorm over the whole width, and the Sinkhorn. Sinkhorn: multiplicative
# (one exp, then divisions) when every token's 4x4 logits span < 60, else the log domain (training's op) -- the Metal
# engine's rule: in that range the two agree to ~1e-7, beyond it fp32 underflows in the multiplicative form. Larger
# batches (prefill) keep the three kernels, measured faster there.

import os

import torch

from vllm.triton_utils import tl, triton

from nenya_vllm.ops.decode_gemv import PDL, pdl, counters, repeats


@triton.jit
def _sinkhorn_block(z, valid, ITERS: tl.constexpr):
    """z: [BT, N, N] fp32 logits -> doubly-stochastic H (columns normalised first, rows last). valid: [BT] rows
    that are real tokens (padding rows must not force the slow path)."""
    zmax = tl.max(tl.max(z, axis=2), axis=1)
    zmin = tl.min(tl.min(z, axis=2), axis=1)
    if tl.max(tl.where(valid, zmax - zmin, 0.0), axis=0) < 60.0:
        p = tl.exp(z - zmax[:, None, None])
        for _ in tl.static_range(ITERS):
            p = p / tl.sum(p, axis=1)[:, None, :]
            p = p / tl.sum(p, axis=2)[:, :, None]
        h = p
    else:
        for _ in tl.static_range(ITERS):
            cmax = tl.max(z, axis=1)
            z = z - (cmax[:, None, :] + tl.log(tl.sum(tl.exp(z - cmax[:, None, :]), axis=1))[:, None, :])
            rmax = tl.max(z, axis=2)
            z = z - (rmax[:, :, None] + tl.log(tl.sum(tl.exp(z - rmax[:, :, None]), axis=2))[:, :, None])
        h = tl.exp(z)
    return h


@triton.jit
def _theta_tile(theta_ptr, col, ND: tl.constexpr, N: tl.constexpr):
    """theta columns `col` as a [32, CH] tile with the rows reordered: 0-15 = res (theta rows 2N..2N+15),
    16-19 = pre (rows 0..3), 20-23 = post (rows 4..7), 24-31 = zero."""
    r = tl.arange(0, 32)
    src = tl.where(r < 16, r + 2 * N, r - 16)
    return tl.load(theta_ptr + src[:, None] * ND + col[None, :], mask=(r < 24)[:, None], other=0.0)


@triton.jit
def _mhc_boundary_kernel(
    x_ptr,           # [T, N, D] streams before the boundary (bf16; fp32 for the model's first sublayer)
    layer_ptr,       # [T, D] bf16 output of the finishing sublayer          (DO_POST)
    coef_in_ptr,     # [T, N + N*N] fp32 post gates + H of the finishing sublayer (DO_POST)
    xnew_ptr,        # [T, N, D] bf16 new streams                              (DO_POST)
    theta_ptr,       # [NM, N*D] bf16 theta of the next sublayer (row-major)
    alpha_ptr,       # [3] fp32
    beta_ptr,        # [NM] fp32
    norm_w_ptr,      # [D] fp32
    out_f32_ptr,     # [T, D] normalised input of the next sublayer, fp32
    out_bf16_ptr,    # [T, D] the same in bf16
    coef_out_ptr,    # [T, N + N*N] fp32
    part_ptr,        # [n_blocks, NC, BT, 32] fp32 scratch: per-chunk theta partial dots, sum of squares in col 24
    cnt_ptr,         # [n_blocks] int32, zero between launches: chunks of the token block that are done
    T, norm_eps,
    N: tl.constexpr, D: tl.constexpr, DP2: tl.constexpr, CH: tl.constexpr, NC: tl.constexpr,
    NCP2: tl.constexpr, BT: tl.constexpr, ITERS: tl.constexpr, DO_POST: tl.constexpr, PDL: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # grid (token blocks of BT <= 4, NC chunks of D [+ 1 if SPLIT]). Every chunk program does the post and the
    # partial theta dots of one chunk; the last chunk program of a token block to finish (atomic counter) reduces the
    # partials in a fixed order (deterministic) and does the gates and the read + RMSNorm over the whole width.
    # The Sinkhorn: SPLIT -- an extra program per token block (launched last, so its wait cannot hold up the chunk
    # programs) waits for the counter to reach NC and computes it from the same partials in parallel with that tail;
    # else the tail runs it after the norm.
    ND: tl.constexpr = N * D
    pid = tl.program_id(0)
    pc = tl.program_id(1)
    offs_t = pid * BT + tl.arange(0, BT)
    mask_t = offs_t < T
    t64 = offs_t.to(tl.int64)
    col = tl.arange(0, 32)
    rows_b = tl.arange(0, BT)
    if SPLIT:
        if pc == NC:
            a_res_s = tl.load(alpha_ptr + 2)
            b_res_s = tl.load(beta_ptr + tl.where(col < 16, col + 2 * N, col - 16), mask=col < 24, other=0.0)
            if PDL:
                tl.extra.cuda.gdc_wait()
                tl.extra.cuda.gdc_launch_dependents()
            arrived = tl.atomic_add(cnt_ptr + pid, 0, sem="acquire")
            while arrived < NC:
                arrived = tl.atomic_add(cnt_ptr + pid, 0, sem="acquire")
            tl.store(cnt_ptr + pid, 0)                          # every chunk has arrived: ready for the next launch
            offs_cs = tl.arange(0, NCP2)
            parts_s = tl.load(part_ptr + ((pid * NC + offs_cs[:, None, None]) * BT + rows_b[None, :, None]) * 32
                              + col[None, None, :], mask=(offs_cs < NC)[:, None, None], other=0.0, cache_modifier=".cg")
            tot_s = tl.sum(parts_s, axis=0)
            ss_s = tl.sum(tl.where(col[None, :] == 24, tot_s, 0.0), axis=1)
            m_s = tot_s.to(tl.bfloat16).to(tl.float32) * (1.0 / tl.sqrt(ss_s / (N * D) + 1e-20))[:, None]
            z_s = a_res_s * m_s + b_res_s[None, :]
            z3s = tl.reshape(z_s, [BT, 2, 16])
            zres_s = tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, z3s, 0.0), axis=1)
            hres_s = _sinkhorn_block(tl.reshape(zres_s, [BT, N, N]), mask_t, ITERS)
            offs_is = tl.arange(0, N)
            cout3s = coef_out_ptr + t64[:, None, None] * (N + N * N)
            tl.store(cout3s + N + offs_is[None, :, None] * N + offs_is[None, None, :], hres_s, mask=mask_t[:, None, None])
            return
    offs_d = pc * CH + tl.arange(0, CH)
    m2 = mask_t[:, None]

    # theta is never written by a kernel: load it before waiting on the previous kernel (PDL)
    th0 = _theta_tile(theta_ptr, 0 * D + offs_d, ND, N).to(tl.float32)
    th1 = _theta_tile(theta_ptr, 1 * D + offs_d, ND, N).to(tl.float32)
    th2 = _theta_tile(theta_ptr, 2 * D + offs_d, ND, N).to(tl.float32)
    th3 = _theta_tile(theta_ptr, 3 * D + offs_d, ND, N).to(tl.float32)
    # the tail's parameters too, in every program (which one runs the tail is decided at run time): off its critical
    # path, where a cold load from DRAM costs ~1 us
    offs_dd = tl.arange(0, DP2)
    w = tl.load(norm_w_ptr + offs_dd, mask=offs_dd < D, other=0.0)
    a_pre = tl.load(alpha_ptr)
    a_post = tl.load(alpha_ptr + 1)
    a_res = tl.load(alpha_ptr + 2)
    bvec = tl.load(beta_ptr + tl.where(col < 16, col + 2 * N, col - 16), mask=col < 24, other=0.0)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if DO_POST:
        cin = coef_in_ptr + t64 * (N + N * N)
        lay = tl.load(layer_ptr + t64[:, None] * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
        x0 = tl.load(x_ptr + t64[:, None] * ND + 0 * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + t64[:, None] * ND + 1 * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + t64[:, None] * ND + 2 * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + t64[:, None] * ND + 3 * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
    sq = tl.zeros([BT, CH], dtype=tl.float32)
    prod = tl.zeros([BT, 32, CH], dtype=tl.float32)
    for i in tl.static_range(N):
        if i == 0:
            th = th0
        elif i == 1:
            th = th1
        elif i == 2:
            th = th2
        else:
            th = th3
        if DO_POST:
            hp = tl.load(cin + i, mask=mask_t, other=0.0)
            h0 = tl.load(cin + N + i * N + 0, mask=mask_t, other=0.0)
            h1 = tl.load(cin + N + i * N + 1, mask=mask_t, other=0.0)
            h2 = tl.load(cin + N + i * N + 2, mask=mask_t, other=0.0)
            h3 = tl.load(cin + N + i * N + 3, mask=mask_t, other=0.0)
            # same accumulation order as _mhc_post_kernel: post * layer, then + H[i, j] x_j for j = 0..3
            xi = hp[:, None] * lay
            xi += h0[:, None] * x0
            xi += h1[:, None] * x1
            xi += h2[:, None] * x2
            xi += h3[:, None] * x3
            xb = xi.to(tl.bfloat16)
            tl.store(xnew_ptr + t64[:, None] * ND + i * D + offs_d[None, :], xb, mask=m2)
            xf = xb.to(tl.float32)
            xt = xf
        else:
            xf = tl.load(x_ptr + t64[:, None] * ND + i * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
            # theta's input is the bf16 streams (cuBLAS on bf16 x_res); the norm's sum of squares the stored streams
            xt = xf.to(tl.bfloat16).to(tl.float32)
        sq += xf * xf
        prod += xt[:, None, :] * th[None, :, :]
    # fp32 products and accumulation; one cross-thread reduction for the theta dots, one for the sum of squares
    acc = tl.sum(prod, axis=2)
    acc = tl.where(col[None, :] == 24, tl.sum(sq, axis=1)[:, None], acc)
    tl.store(part_ptr + ((pid * NC + pc) * BT + rows_b)[:, None] * 32 + col[None, :], acc)
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
    if done == NC - 1:
        if not SPLIT:
            tl.store(cnt_ptr + pid, 0)                      # ready for the next launch / graph replay
        if DO_POST:
            src = xnew_ptr
        else:
            src = x_ptr
        # every load of the tail at once: the partials and the new streams (written by the other programs)
        offs_c = tl.arange(0, NCP2)
        parts = tl.load(part_ptr + ((pid * NC + offs_c[:, None, None]) * BT + rows_b[None, :, None]) * 32
                        + col[None, None, :], mask=(offs_c < NC)[:, None, None], other=0.0, cache_modifier=".cg")
        md = m2 & (offs_dd < D)[None, :]
        base = src + t64[:, None] * ND + offs_dd[None, :]
        s0 = tl.load(base, mask=md, other=0.0, cache_modifier=".cg").to(tl.float32)
        s1 = tl.load(base + D, mask=md, other=0.0, cache_modifier=".cg").to(tl.float32)
        s2 = tl.load(base + 2 * D, mask=md, other=0.0, cache_modifier=".cg").to(tl.float32)
        s3 = tl.load(base + 3 * D, mask=md, other=0.0, cache_modifier=".cg").to(tl.float32)
        tot = tl.sum(parts, axis=0)                         # [BT, 32]
        # the per-token scalars (sum of squares, gates) go through this program's scratch row instead of
        # cross-thread reductions
        own = part_ptr + ((pid * NC + pc) * BT + rows_b)[:, None] * 32 + col[None, :]
        tl.debug_barrier()
        tl.store(own, tot)
        tl.debug_barrier()
        ss = tl.load(part_ptr + ((pid * NC + pc) * BT + rows_b) * 32 + 24)
        r = 1.0 / tl.sqrt(ss / (N * D) + 1e-20)
        # cuBLAS rounds theta's output to bf16
        m = tot.to(tl.bfloat16).to(tl.float32) * r[:, None]
        alpha = tl.where(col < 16, a_res, tl.where(col < 20, a_pre, a_post))
        z = alpha[None, :] * m + bvec[None, :]              # [BT, 32]: res logits | pre | post | -
        sg = tl.sigmoid(z)
        cout = coef_out_ptr + t64[:, None] * (N + N * N)
        tl.store(cout + (col[None, :] - 20), 2.0 * sg, mask=m2 & ((col >= 20) & (col < 24))[None, :])
        tl.debug_barrier()
        tl.store(own, sg)
        tl.debug_barrier()
        prow = part_ptr + ((pid * NC + pc) * BT + rows_b) * 32 + 16
        p0 = tl.load(prow)
        p1 = tl.load(prow + 1)
        p2 = tl.load(prow + 2)
        p3 = tl.load(prow + 3)
        xin = p0[:, None] * s0
        xin += p1[:, None] * s1
        xin += p2[:, None] * s2
        xin += p3[:, None] * s3
        inv = 1.0 / tl.sqrt(tl.sum(xin * xin, axis=1) / D + norm_eps)
        xn = xin * inv[:, None] * w[None, :]
        tl.store(out_f32_ptr + t64[:, None] * D + offs_dd[None, :], xn, mask=md)
        tl.store(out_bf16_ptr + t64[:, None] * D + offs_dd[None, :], xn.to(tl.bfloat16), mask=md)
        if not SPLIT:
            z3 = tl.reshape(z, [BT, 2, 16])
            zres = tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, z3, 0.0), axis=1)
            hres = _sinkhorn_block(tl.reshape(zres, [BT, N, N]), mask_t, ITERS)
            offs_i = tl.arange(0, N)
            cout3 = coef_out_ptr + t64[:, None, None] * (N + N * N)
            tl.store(cout3 + N + offs_i[None, :, None] * N + offs_i[None, None, :], hres,
                     mask=mask_t[:, None, None])


# the Sinkhorn on its own program, in parallel with the read + RMSNorm tail (NENYA_MHC_SPLIT=0: in the tail)
SPLIT = os.environ.get("NENYA_MHC_SPLIT", "1") == "1"
# Up to 128 tokens the fused kernel; above, the three-kernel path (post, theta GEMM, pre), which measured faster
# from ~256 tokens up on an RTX 5090 (benchmarks/bench_kernels.py boundary: 4.1 vs 11.3 us at 1 token, 5.5 vs 30 us at
# 16, 10.8 vs 12.0 us at 128, 18.7 vs 13.3 us at 256). NENYA_MHC_SMALL_T overrides. (A one-pass kernel for prefill
# batches -- post, theta on the tensor cores, gates, read + RMSNorm per block of 16-64 tokens -- was slower than the
# three kernels at every size from 64 to 16K tokens, e.g. 60 vs 34 us at 2K: one program per block leaves the GPU
# underfilled.)
SMALL_T = int(os.environ.get("NENYA_MHC_SMALL_T", "128"))
# tokens per block of the fused kernel (0: 1, 2 or 4 by batch size) and its warps per program
BT_FIXED = int(os.environ.get("NENYA_MHC_BT", "0"))
WARPS = int(os.environ.get("NENYA_MHC_WARPS", "4"))


def mhc_boundary(x_res: torch.Tensor, layer_out: torch.Tensor | None, coef_in: torch.Tensor | None, mhc,
                 norm_weight: torch.Tensor, norm_eps: float):
    """post (if layer_out is given) + the next pre. Returns (x_new [T, N, D] bf16 -- x_res itself when there is no
    post --, x_norm fp32 [T, D], x_norm bf16 [T, D], coef fp32 [T, N + N*N])."""
    T, N, D = x_res.shape
    assert N == 4, "the boundary kernel is written for 4 streams"
    do_post = layer_out is not None
    if T > SMALL_T:
        from nenya_vllm.ops.mhc_kernels import mhc_post

        x_new = mhc_post(x_res, layer_out, coef_in) if do_post else x_res
        x32, x16, coef = mhc.pre(x_new, norm_weight, norm_eps)
        return x_new, x32, x16, coef
    dev = x_res.device
    x_res = x_res.contiguous()
    x_new = torch.empty((T, N, D), dtype=torch.bfloat16, device=dev) if do_post else x_res
    out_f32 = torch.empty((T, D), dtype=torch.float32, device=dev)
    out_bf16 = torch.empty((T, D), dtype=torch.bfloat16, device=dev)
    coef = torch.empty((T, N + N * N), dtype=torch.float32, device=dev)
    if T:
        BT = BT_FIXED or (1 if T == 1 else 2 if T == 2 else 4)
        CH = 32
        NC = D // CH
        n_blocks = triton.cdiv(T, BT)
        part = torch.empty((n_blocks, NC, BT, 32), dtype=torch.float32, device=dev)
        for _ in range(repeats("mhc")):
            _mhc_boundary_kernel[(n_blocks, NC + SPLIT)](
                x_res, layer_out.contiguous() if do_post else x_res, coef_in if do_post else coef, x_new,
                mhc.theta.weight, mhc.alpha, mhc.beta, norm_weight, out_f32, out_bf16, coef, part,
                counters(n_blocks, dev, "mhc"), T, norm_eps,
                N=N, D=D, DP2=triton.next_power_of_2(D), CH=CH, NC=NC, NCP2=triton.next_power_of_2(NC), BT=BT,
                ITERS=mhc.iters, DO_POST=do_post, PDL=pdl("mhc"), SPLIT=SPLIT, launch_pdl=pdl("mhc"), num_warps=WARPS)
    return x_new, out_f32, out_bf16, coef


@triton.jit
def _mhc_post_final_kernel(x_ptr, layer_ptr, coef_ptr, norm_w_ptr, out_ptr, norm_eps,
                           N: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr, PDL: tl.constexpr):
    """The last sublayer's post, the sum of the streams and the final RMSNorm, with the unfused path's roundings:
    x_new = bf16(post), h = bf16(sum of the bf16 streams), out = bf16(RMSNorm_fp32w(h))."""
    t = tl.program_id(0).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    w = tl.load(norm_w_ptr + offs_d, mask=mask_d, other=0.0)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    base = coef_ptr + t * (N + N * N)
    layer = tl.load(layer_ptr + t * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + t * N * D + 0 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + t * N * D + 1 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + t * N * D + 2 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + t * N * D + 3 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    h = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in tl.static_range(N):
        xi = tl.load(base + i) * layer
        xi += tl.load(base + N + i * N + 0) * x0
        xi += tl.load(base + N + i * N + 1) * x1
        xi += tl.load(base + N + i * N + 2) * x2
        xi += tl.load(base + N + i * N + 3) * x3
        h += xi.to(tl.bfloat16).to(tl.float32)
    h = h.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(h * h, axis=0) / D
    tl.store(out_ptr + t * D + offs_d, (h * (1.0 / tl.sqrt(var + norm_eps)) * w).to(tl.bfloat16), mask=mask_d)


def mhc_post_final(x_res: torch.Tensor, layer_out: torch.Tensor, coef: torch.Tensor, norm_weight: torch.Tensor,
                   norm_eps: float) -> torch.Tensor:
    """[T, D] bf16 final hidden state."""
    T, N, D = x_res.shape
    out = torch.empty((T, D), dtype=torch.bfloat16, device=x_res.device)
    if T:
        _mhc_post_final_kernel[(T,)](x_res.contiguous(), layer_out.contiguous(), coef, norm_weight, out, norm_eps,
                                     N=N, D=D, BLOCK_D=triton.next_power_of_2(D), PDL=pdl("mhc_final"), launch_pdl=pdl("mhc_final"),
                                     num_warps=4)
    return out
