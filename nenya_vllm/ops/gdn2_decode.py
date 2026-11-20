# SPDX-License-Identifier: Apache-2.0
# One decode step of Nenya's GDN-2 token mixer in two kernels (plus the output projection), replacing in_proj, f_b
# and g_b GEMVs, the gate kernel, vLLM's short-convolution update, a copy and fla's recurrent kernel:
#   1. in_proj GEMV whose epilogue runs the q/k/v short convolutions: each output channel belongs to one program,
#      which reads and rewrites that channel's history -- no other program touches it
#   2. the recurrence, one program per (token, head, 32 value columns) as fla's kernel, with f_b, g_b and the
#      erase/write gates computed in its prologue
# The output gate's RMSNorm and the output projection run in the next kernel (ops/decode_gemv.py, GATED_RMSNORM).
#
# Same maths and rounding points as the unfused decode path (layers/gdn2.py), which runs the kernels training ran:
#   x      = bf16(in_proj(x))                                                  (cuBLAS: fp32 sums, bf16 output)
#   q,k,v  = bf16(SiLU(sum of the 4 taps)), fp32 taps on the bf16 history       (vLLM causal_conv1d_update)
#   f      = bf16(f_b(f_a)),  gate = bf16(g_b(g_a) + bias),  b, w = bf16(sigmoid(.))
#   q, k  -> x / sqrt(sum(x^2) + 1e-6), q *= K^-0.5;  g = -exp(A_log) softplus(f + dt_bias)
#   S     *= exp(g) (key axis);  v' = w v - (b k)^T S;  S += k v'^T;  o = bf16(S^T q)   (fla 0.5.2 fused_recurrent)
# with fla's own exp and softplus (inline PTX, threshold 20). Rows of padded CUDA-graph batches (state slot = vLLM's
# null block 0, or past the real rows) skip the history and state updates and output zeros.

import os

import torch

from vllm.triton_utils import tl, triton

from fla.ops.utils.softplus import softplus

from nenya_vllm.ops.decode_gemv import PDL, pdl, SPLITK_PROGRAMS, counters, repeats

# in_proj's tensor-core path: split-K slices (0 = the measured choice, -1 = ~SPLITK_PROGRAMS programs) and an optional
# "BN,BK,stages,splitk" override (benchmarks)
IN_PROJ_SPLITK = int(os.environ.get("NENYA_GDN_IN_SPLITK", "0"))
IN_PROJ_CONFIG = os.environ.get("NENYA_GDN_IN_CONFIG", "")
# the short-prefill recurrence: value columns per program and warps ("BV,warps"; default by sequence count)
VARLEN_CFG = tuple(int(v) for v in os.environ.get("NENYA_GDN_VARLEN_CFG", "").split(",") if v)


@triton.jit
def _gdn2_in_proj_kernel(x_ptr, x_stride,                 # [T, D] bf16 normalised input
                         w_ptr,                           # [NOUT, D] bf16 in_proj weight (q k v b w f_a g_a rows)
                         y_ptr, y_stride,                 # [T, NOUT] bf16 output; q/k/v columns hold SiLU(conv)
                         cs_ptr, cs_s_slot, cs_s_dim, cs_s_tok,   # conv history [slots, NCONV, 3] bf16
                         cw_ptr,                          # [NCONV, 4] fp32 conv taps
                         idx_ptr, T, n_real,
                         D: tl.constexpr, DP2: tl.constexpr, NOUT: tl.constexpr, NCONV: tl.constexpr,
                         BN: tl.constexpr, TB: tl.constexpr, PDL: tl.constexpr):
    pid_n = tl.program_id(0)
    t0 = tl.program_id(1) * TB
    rows = pid_n * BN + tl.arange(0, BN)
    mask_n = rows < NOUT
    offs_k = tl.arange(0, DP2)
    mask_k = offs_k < D
    # weights first: no kernel writes them, so under PDL these loads overlap the previous kernel
    w = tl.load(w_ptr + rows.to(tl.int64)[:, None] * D + offs_k[None, :], mask=mask_n[:, None] & mask_k[None, :],
                other=0.0).to(tl.float32)
    is_conv = pid_n * BN < NCONV                          # BN divides NCONV: a block is all-conv or none
    crow = tl.where(rows < NCONV, rows, 0)
    w0 = tl.load(cw_ptr + crow * 4 + 0)
    w1 = tl.load(cw_ptr + crow * 4 + 1)
    w2 = tl.load(cw_ptr + crow * 4 + 2)
    w3 = tl.load(cw_ptr + crow * 4 + 3)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    for i in tl.static_range(TB):
        t = t0 + i
        if t < T:
            x = tl.load(x_ptr + t.to(tl.int64) * x_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            xb = tl.sum(w * x[None, :], axis=1).to(tl.bfloat16)
            slot = tl.load(idx_ptr + t, mask=(t < n_real) & is_conv, other=0).to(tl.int64)
            if slot != 0:
                # vLLM's causal_conv1d_update: taps summed in order in fp32 on the bf16 history, SiLU, bf16
                base = cs_ptr + slot * cs_s_slot + rows.to(tl.int64) * cs_s_dim
                c0 = tl.load(base)
                c1 = tl.load(base + cs_s_tok)
                c2 = tl.load(base + 2 * cs_s_tok)
                acc = tl.zeros([BN], dtype=tl.float32)
                acc += c0 * w0
                acc += c1 * w1
                acc += c2 * w2
                acc += xb.to(tl.float32) * w3
                tl.store(y_ptr + t.to(tl.int64) * y_stride + rows, (acc / (1 + tl.exp(-acc))).to(tl.bfloat16))
                # every warp has read the history (the [BN] vectors are replicated) before it is rewritten
                tl.debug_barrier()
                tl.store(base, c1)
                tl.store(base + cs_s_tok, c2)
                tl.store(base + 2 * cs_s_tok, xb)
            else:
                tl.store(y_ptr + t.to(tl.int64) * y_stride + rows, xb, mask=mask_n)


@triton.jit
def _gdn2_in_proj_mma_kernel(x_ptr, x_stride, w_ptr, y_ptr, y_stride, cs_ptr, cs_s_slot, cs_s_dim, cs_s_tok, cw_ptr,
                             idx_ptr, T, n_real, ws_ptr, cnt_ptr,
                             D: tl.constexpr, NOUT: tl.constexpr, NCONV: tl.constexpr, BM: tl.constexpr,
                             BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr, KS: tl.constexpr,
                             SPLITK: tl.constexpr, M_PAD: tl.constexpr, PDL: tl.constexpr):
    """gdn2_in_proj for 5..128 tokens: [BM tokens, BK] x [BK, BN] tensor-core tiles (fp32 accumulation, bf16 output
    like cuBLAS), the same convolution epilogue on the [BM, BN] output tile. Split-K (SPLITK > 1): K slices summed
    into fp32 partials, added in slice order by the tile's last program, which runs the epilogue."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)
    k_lo = pid_s * KS
    k_hi = tl.minimum(k_lo + KS, D)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < NOUT
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < T
    m64 = offs_m.to(tl.int64)[:, None]
    wrow = w_ptr + offs_n.to(tl.int64)[:, None] * D
    offs_k0 = k_lo + tl.arange(0, BK)
    w0 = tl.load(wrow + offs_k0[None, :], mask=mask_n[:, None] & (offs_k0 < k_hi)[None, :], other=0.0)
    is_conv = pid_n * BN < NCONV
    crow = tl.where(offs_n < NCONV, offs_n, 0)
    w0c = tl.load(cw_ptr + crow * 4 + 0)
    w1c = tl.load(cw_ptr + crow * 4 + 1)
    w2c = tl.load(cw_ptr + crow * 4 + 2)
    w3c = tl.load(cw_ptr + crow * 4 + 3)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x0 = tl.load(x_ptr + m64 * x_stride + offs_k0[None, :], mask=mask_m[:, None] & (offs_k0 < k_hi)[None, :],
                 other=0.0)
    acc = tl.dot(x0, tl.trans(w0))
    for k0 in tl.range(k_lo + BK, k_hi, BK, num_stages=STAGES):
        offs_k = k0 + tl.arange(0, BK)
        mask_k = offs_k < k_hi
        w = tl.load(wrow + offs_k[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        x = tl.load(x_ptr + m64 * x_stride + offs_k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    if SPLITK > 1:
        wsp = ws_ptr + m64 * NOUT + offs_n[None, :]
        tl.store(wsp + pid_s * (M_PAD * NOUT), acc, mask=mask_n[None, :])
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(0) + pid_n
        if tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel") == SPLITK - 1:
            acc = tl.load(wsp, mask=mask_n[None, :], other=0.0, cache_modifier=".cg")
            for s in tl.static_range(1, SPLITK):
                acc += tl.load(wsp + s * (M_PAD * NOUT), mask=mask_n[None, :], other=0.0, cache_modifier=".cg")
            tl.store(cnt_ptr + tile, 0)
            _gdn2_in_proj_epilogue(acc, y_ptr, y_stride, m64, offs_m, mask_m, offs_n, mask_n, is_conv, cs_ptr,
                                   cs_s_slot, cs_s_dim, cs_s_tok, w0c, w1c, w2c, w3c, idx_ptr, n_real, BM, BN)
    else:
        _gdn2_in_proj_epilogue(acc, y_ptr, y_stride, m64, offs_m, mask_m, offs_n, mask_n, is_conv, cs_ptr, cs_s_slot,
                               cs_s_dim, cs_s_tok, w0c, w1c, w2c, w3c, idx_ptr, n_real, BM, BN)


@triton.jit
def _gdn2_in_proj_epilogue(acc, y_ptr, y_stride, m64, offs_m, mask_m, offs_n, mask_n, is_conv, cs_ptr, cs_s_slot,
                           cs_s_dim, cs_s_tok, w0c, w1c, w2c, w3c, idx_ptr, n_real, BM: tl.constexpr,
                           BN: tl.constexpr):
    """bf16 output tile; q/k/v columns: vLLM's causal_conv1d_update on it (taps in order in fp32, SiLU, bf16) and
    the history shifted"""
    xb = acc.to(tl.bfloat16)
    yp = y_ptr + m64 * y_stride + offs_n[None, :]
    if is_conv:
        slot = tl.load(idx_ptr + offs_m, mask=mask_m & (offs_m < n_real), other=0).to(tl.int64)
        real = slot != 0
        base = cs_ptr + slot[:, None] * cs_s_slot + offs_n.to(tl.int64)[None, :] * cs_s_dim
        mreal = real[:, None] & mask_n[None, :]
        c0 = tl.load(base, mask=mreal, other=0.0)
        c1 = tl.load(base + cs_s_tok, mask=mreal, other=0.0)
        c2 = tl.load(base + 2 * cs_s_tok, mask=mreal, other=0.0)
        cacc = tl.zeros([BM, BN], dtype=tl.float32)
        cacc += c0 * w0c[None, :]
        cacc += c1 * w1c[None, :]
        cacc += c2 * w2c[None, :]
        cacc += xb.to(tl.float32) * w3c[None, :]
        y = tl.where(mreal, (cacc / (1 + tl.exp(-cacc))).to(tl.bfloat16), xb)
        tl.store(yp, y, mask=mask_m[:, None] & mask_n[None, :])
        tl.debug_barrier()
        tl.store(base, c1, mask=mreal)
        tl.store(base + cs_s_tok, c2, mask=mreal)
        tl.store(base + 2 * cs_s_tok, xb, mask=mreal)
    else:
        tl.store(yp, xb, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _gdn2_recurrent_kernel(
    proj_ptr, proj_stride,            # [T, 5 KD + 2 K] bf16: SiLU(conv(q k v)) | b w | f_a | g_a
    fb_ptr,                           # [KD, K] bf16 f_b weight
    gb_ptr, gbb_ptr,                  # [KD, K] bf16 g_b weight, [KD] bf16 bias
    alog_ptr, dtb_ptr,                # [H] fp32, [KD] fp32
    ssm_ptr, ssm_s_slot,              # recurrent state [slots, H, K, V] fp32 (inner dims contiguous)
    idx_ptr,                          # [T] int32 state slots
    core_ptr, gate_ptr,               # [T, KD] bf16 outputs: o, output gate
    n_real, scale,
    K: tl.constexpr, KD: tl.constexpr, BV: tl.constexpr, PDL: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    iv = tl.program_id(2)
    t64 = t.to(tl.int64)
    offs = tl.arange(0, K)
    offs_v = iv * BV + tl.arange(0, BV)
    rows = h * K + offs                       # this head's key channels
    vrows = h * K + offs_v                    # this program's value channels
    # weights first: no kernel writes them, so under PDL these loads overlap the previous kernel
    fbw = tl.load(fb_ptr + rows[:, None] * K + offs[None, :])
    gbw = tl.load(gb_ptr + vrows[:, None] * K + offs[None, :])
    gbb = tl.load(gbb_ptr + vrows).to(tl.float32)
    a_log = tl.load(alog_ptr + h)
    dtb = tl.load(dtb_ptr + rows)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    prow = proj_ptr + t64 * proj_stride
    g_a = tl.load(prow + 5 * KD + K + offs).to(tl.float32)
    gate = tl.sum(gbw.to(tl.float32) * g_a[None, :], axis=1) + gbb
    tl.store(gate_ptr + t64 * KD + vrows, gate.to(tl.bfloat16))
    slot = tl.load(idx_ptr + t, mask=t < n_real, other=0).to(tl.int64)
    if slot == 0:
        tl.store(core_ptr + t64 * KD + vrows, tl.zeros([BV], dtype=tl.bfloat16))
    else:
        f_a = tl.load(prow + 5 * KD + offs).to(tl.float32)
        f = tl.sum(fbw.to(tl.float32) * f_a[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
        b = tl.sigmoid(tl.load(prow + 3 * KD + rows).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        w = tl.sigmoid(tl.load(prow + 4 * KD + vrows).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        q = tl.load(prow + rows).to(tl.float32)
        k = tl.load(prow + KD + rows).to(tl.float32)
        v = tl.load(prow + 2 * KD + vrows).to(tl.float32)
        # fla's recurrence on the [K, BV] state tile
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q = q * scale
        gk = -tl.exp(a_log) * softplus(f + dtb)
        sp = ssm_ptr + slot * ssm_s_slot + h * K * K + offs[:, None] * K + offs_v[None, :]
        s = tl.load(sp)
        s *= tl.exp(gk[:, None])
        bk = b * k
        erase = tl.sum(s * bk[:, None], 0)
        v_new = w * v - erase
        s += k[:, None] * v_new[None, :]
        o = tl.sum(s * q[:, None], 0)
        tl.store(core_ptr + t64 * KD + vrows, o.to(tl.bfloat16))
        tl.debug_barrier()
        tl.store(sp, s)


def gdn2_in_proj(x: torch.Tensor, weight: torch.Tensor, conv_state: torch.Tensor, conv_weight: torch.Tensor,
                 state_idx: torch.Tensor, n_real: int) -> torch.Tensor:
    """bf16(x @ weight^T) [T, NOUT] with the first NCONV = 3 KD columns (q, k, v) replaced by SiLU of their short
    convolution; updates conv_state ([slots, NCONV, 3] canonical view) for the rows < n_real with a real slot."""
    T, D = x.shape
    NOUT = weight.shape[0]
    NCONV = conv_weight.shape[0]
    y = torch.empty((T, NOUT), dtype=torch.bfloat16, device=x.device)
    from nenya_vllm.ops.decode_gemv import GEMM_MIN_T

    if T >= GEMM_MIN_T:
        BM = 16 if T <= 16 else 32 if T <= 32 else 64
        # no split-K here. With it and this kernel launched under PDL, decode after prefix-cache hits gave worse
        # log-probs that differed between identical runs with identical batching -- a race (mean |dlogprob|
        # 0.0215-0.0238 over three runs vs 0.0160); with PDL off
        # for this kernel (NENYA_PDL_OFF=gdn_in) split-K was bit-reproducible and level (0.0161). Mechanism not
        # found; it is the only split-K epilogue that writes per-request state. Split-K saved ~1 us per call at 8-16
        # tokens. NENYA_GDN_IN_SPLITK re-enables it, for investigating only.
        BN, BK, stages, splitk = (64, 128, 2, 1) if T <= 16 else (32, 64, 3, 1)
        if IN_PROJ_SPLITK:
            splitk = IN_PROJ_SPLITK
        if IN_PROJ_CONFIG:
            BN, BK, stages, splitk = (int(v) for v in IN_PROJ_CONFIG.split(","))
        assert NCONV % BN == 0 and x.stride(1) == 1 and n_real <= min(T, state_idx.shape[0])
        n_tiles = triton.cdiv(NOUT, BN) * triton.cdiv(T, BM)
        if splitk < 0:
            splitk = max(1, min(triton.cdiv(D, BK), round(SPLITK_PROGRAMS / n_tiles)))
        KS = triton.cdiv(triton.cdiv(D, BK), splitk) * BK
        splitk = triton.cdiv(D, KS)
        M_PAD = triton.cdiv(T, BM) * BM
        ws = torch.empty((splitk, M_PAD, NOUT), dtype=torch.float32, device=x.device) if splitk > 1 else y
        cnt = counters(n_tiles, x.device, "gdn_in_split") if splitk > 1 else y
        for _ in range(repeats("gdn_in")):
            _gdn2_in_proj_mma_kernel[(triton.cdiv(NOUT, BN), triton.cdiv(T, BM), splitk)](
                x, x.stride(0), weight, y, y.stride(0), conv_state, conv_state.stride(0), conv_state.stride(1),
                conv_state.stride(2), conv_weight, state_idx, T, n_real, ws, cnt, D=D, NOUT=NOUT, NCONV=NCONV, BM=BM,
                BN=BN, BK=BK, STAGES=stages, KS=KS, SPLITK=splitk, M_PAD=M_PAD, PDL=pdl("gdn_in"), launch_pdl=pdl("gdn_in"),
                num_warps=4)
        return y
    if T:
        BN = 4 if T <= 4 else 8
        assert NCONV % BN == 0 and x.stride(1) == 1 and n_real <= min(T, state_idx.shape[0])
        TB = min(T, 16)
        for _ in range(repeats("gdn_in")):
          _gdn2_in_proj_kernel[(triton.cdiv(NOUT, BN), triton.cdiv(T, TB))](
            x, x.stride(0), weight, y, y.stride(0), conv_state, conv_state.stride(0), conv_state.stride(1),
            conv_state.stride(2), conv_weight, state_idx, T, n_real, D=D, DP2=triton.next_power_of_2(D), NOUT=NOUT,
            NCONV=NCONV, BN=BN, TB=TB, PDL=pdl("gdn_in"), launch_pdl=pdl("gdn_in"), num_warps=8)
    return y


def gdn2_recurrent_step(proj: torch.Tensor, f_b_weight: torch.Tensor, g_b_weight: torch.Tensor,
                        g_b_bias: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, ssm_state: torch.Tensor,
                        state_idx: torch.Tensor, n_real: int, num_heads: int,
                        head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """proj with SiLU(conv) in its q/k/v columns -> (o [T, KD] bf16, output gate [T, KD] bf16); updates ssm_state
    ([slots, H, K, K] fp32) in place for the rows < n_real with a real slot."""
    T = proj.shape[0]
    H, K = num_heads, head_dim
    KD = H * K
    core = torch.empty((T, KD), dtype=torch.bfloat16, device=proj.device)
    gate = torch.empty((T, KD), dtype=torch.bfloat16, device=proj.device)
    if T:
        assert ssm_state.stride(-1) == 1 and ssm_state.stride(-2) == K and ssm_state.stride(-3) == K * K
        assert proj.stride(1) == 1 and n_real <= min(T, state_idx.shape[0])
        BV = 32
        for _ in range(repeats("gdn_rec")):
          _gdn2_recurrent_kernel[(T, H, K // BV)](
            proj, proj.stride(0), f_b_weight, g_b_weight, g_b_bias, A_log, dt_bias, ssm_state, ssm_state.stride(0),
            state_idx, core, gate, n_real, K ** -0.5, K=K, KD=KD, BV=BV, PDL=pdl("gdn_rec"), launch_pdl=pdl("gdn_rec"),
            num_warps=8)
    return core, gate


# ---- short prefills (and mixed batches): varlen short convolution + sequential recurrence --------------------------
# fla's chunked kernels are the right tool for long prompts, but each call runs ~15 autotuned Triton launches with
# their Python overhead (~0.8 ms per layer of host time): a 128-token prompt's time to first token was host-bound.
# For batches of up to a few hundred prefill tokens these two kernels do the work in two launches, with the same
# per-token maths as the decode path (fla's recurrence, vLLM's convolution).

@triton.jit
def _gdn2_prep_varlen_kernel(proj_ptr, proj_stride,      # [T, 5 KD + 2 K] bf16 in_proj output (raw)
                             f_ptr,                      # [T, KD] bf16 f_b(f_a)
                             qh_ptr, kh_ptr, bk_ptr, dec_ptr,   # [T, KD] fp32 outs: q/|q| * scale, k/|k|, b k, exp(g)
                             v_ptr,                      # [T, KD] bf16 out: SiLU(conv(v))
                             cs_ptr, cs_s_slot, cs_s_dim, cs_s_tok,   # conv history [slots, 3 KD, 3] bf16
                             cw_ptr, alog_ptr, dtb_ptr,  # [3 KD, 4] fp32, [H] fp32, [KD] fp32
                             qsl_ptr, idx_ptr, init_ptr, n_seqs, bs_iters, scale,
                             K: tl.constexpr, KD: tl.constexpr):
    """Everything per token of the recurrence, in parallel: the short convolutions (vLLM's causal_conv1d with the
    history for each sequence's first taps), fla's q/k L2 norm and decay, the erase gate times k. grid (T, 3 H):
    program (t, g * H + h) does head h of group g in {q and decay, k and b k, v}."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    g = j // (KD // K)
    h = j % (KD // K)
    offs = tl.arange(0, K)
    rows = h * K + offs
    total = tl.load(qsl_ptr + n_seqs)
    if t < total:
        # the sequence containing token t: the last s < n_seqs with qsl[s] <= t (binary search)
        lo = tl.full([], 0, tl.int32)
        hi = tl.full([], 0, tl.int32) + n_seqs - 1
        for _ in range(bs_iters):
            mid = (lo + hi + 1) // 2
            go_up = tl.load(qsl_ptr + mid) <= t
            lo = tl.where(go_up, mid, lo)
            hi = tl.where(go_up, hi, mid - 1)
        start = tl.load(qsl_ptr + lo)
        pos = t - start
        slot = tl.load(idx_ptr + lo).to(tl.int64)
        init = tl.load(init_ptr + lo) != 0
        ch = g * KD + rows                                     # this head's channels of q, k or v
        hbase = cs_ptr + slot * cs_s_slot + ch.to(tl.int64) * cs_s_dim
        acc = tl.zeros([K], dtype=tl.float32)
        for jj in tl.static_range(4):
            p = pos - 3 + jj                                   # this tap's position in the sequence
            if jj < 3:
                from_hist = p < 0
                xin = tl.load(proj_ptr + (start + p).to(tl.int64) * proj_stride + ch, mask=p >= 0, other=0.0)
                hv = tl.load(hbase + (3 + p) * cs_s_tok, mask=from_hist & init, other=0.0)
                tap = tl.where(from_hist, hv, xin)
            else:
                tap = tl.load(proj_ptr + t.to(tl.int64) * proj_stride + ch)
            acc += tap * tl.load(cw_ptr + ch * 4 + jj)
        x = (acc / (1 + tl.exp(-acc))).to(tl.bfloat16).to(tl.float32)
        t64 = t.to(tl.int64)
        if g == 0:
            tl.store(qh_ptr + t64 * KD + rows, x / tl.sqrt(tl.sum(x * x) + 1e-6) * scale)
            f = tl.load(f_ptr + t64 * KD + rows).to(tl.float32)
            gk = -tl.exp(tl.load(alog_ptr + h)) * softplus(f + tl.load(dtb_ptr + rows))
            tl.store(dec_ptr + t64 * KD + rows, tl.exp(gk))
        elif g == 1:
            kn = x / tl.sqrt(tl.sum(x * x) + 1e-6)
            b = tl.sigmoid(tl.load(proj_ptr + t64 * proj_stride + 3 * KD + rows).to(tl.float32))
            tl.store(kh_ptr + t64 * KD + rows, kn)
            tl.store(bk_ptr + t64 * KD + rows, b.to(tl.bfloat16).to(tl.float32) * kn)
        else:
            tl.store(v_ptr + t64 * KD + rows, x.to(tl.bfloat16))


@triton.jit
def _gdn2_recurrent_varlen_kernel(
    qh_ptr, kh_ptr, bk_ptr, dec_ptr,  # [T, KD] fp32 (from the prep kernel)
    v_ptr,                            # [T, KD] bf16 SiLU(conv(v))
    proj_ptr, proj_stride,            # [T, 5 KD + 2 K] bf16 in_proj output (raw w columns; raw q k v for history)
    ssm_ptr, ssm_s_slot,              # recurrent state [slots, H, K, V] fp32
    cs_ptr, cs_s_slot, cs_s_dim, cs_s_tok,   # conv history [slots, 3 KD, 3] bf16 (rewritten after the prep kernel)
    qsl_ptr, idx_ptr, init_ptr,       # [N + 1] query start offsets, [N] state slots, [N] has-initial-state
    core_ptr,                         # [T, KD] bf16 out
    K: tl.constexpr, KD: tl.constexpr, BV: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1)
    iv = tl.program_id(2)
    start = tl.load(qsl_ptr + s)
    end = tl.load(qsl_ptr + s + 1)
    if end > start:
        slot = tl.load(idx_ptr + s).to(tl.int64)
        init = tl.load(init_ptr + s) != 0
        offs = tl.arange(0, K)
        offs_v = iv * BV + tl.arange(0, BV)
        rows = h * K + offs
        vrows = h * K + offs_v
        sp = ssm_ptr + slot * ssm_s_slot + h * K * K + offs[:, None] * K + offs_v[None, :]
        st = tl.load(sp, mask=(offs[:, None] >= 0) & init, other=0.0)
        # fla's recurrence; the per-token inputs do not depend on the state, so the loads are pipelined
        for t in tl.range(start, end, num_stages=3):
            t64 = t.to(tl.int64)
            q = tl.load(qh_ptr + t64 * KD + rows)
            k = tl.load(kh_ptr + t64 * KD + rows)
            bk = tl.load(bk_ptr + t64 * KD + rows)
            dec = tl.load(dec_ptr + t64 * KD + rows)
            v = tl.load(v_ptr + t64 * KD + vrows).to(tl.float32)
            w = tl.sigmoid(tl.load(proj_ptr + t64 * proj_stride + 4 * KD + vrows).to(tl.float32))
            w = w.to(tl.bfloat16).to(tl.float32)
            st *= dec[:, None]
            erase = tl.sum(st * bk[:, None], 0)
            v_new = w * v - erase
            st += k[:, None] * v_new[None, :]
            o = tl.sum(st * q[:, None], 0)
            tl.store(core_ptr + t64 * KD + vrows, o.to(tl.bfloat16))
        tl.store(sp, st)
        # the convolution history of this head's channels: the last 3 raw inputs of the sequence (older history
        # entries for sequences shorter than 3); q and k channels by the first value block, v by each block
        n = end - start
        for i in tl.static_range(3):
            p = n - 3 + i
            hsrc = 3 + p
            for g in tl.static_range(3):
                if g < 2:
                    ch = g * KD + rows
                    do = iv == 0
                    m_all = offs >= 0
                else:
                    ch = 2 * KD + vrows
                    do = iv >= 0
                    m_all = offs_v >= 0
                hb = cs_ptr + slot * cs_s_slot + ch.to(tl.int64) * cs_s_dim
                raw = tl.load(proj_ptr + (start + p).to(tl.int64) * proj_stride + ch, mask=m_all & (p >= 0) & do,
                              other=0.0)
                old = tl.load(hb + hsrc * cs_s_tok, mask=m_all & (p < 0) & init & do, other=0.0)
                tl.debug_barrier()
                tl.store(hb + i * cs_s_tok, tl.where(p >= 0, raw, old), mask=m_all & do)


def gdn2_prefill_short(proj: torch.Tensor, f: torch.Tensor, conv_state: torch.Tensor, conv_weight: torch.Tensor,
                       ssm_state: torch.Tensor, qsl: torch.Tensor, state_idx: torch.Tensor, has_init: torch.Tensor,
                       A_log: torch.Tensor, dt_bias: torch.Tensor, n_seqs: int, num_heads: int,
                       head_dim: int, core: torch.Tensor) -> None:
    """Short convolutions + GDN-2 recurrence for the first qsl[n_seqs] rows of a batch of sequences (prefill chunks
    and decodes), reading and updating the state and history pools; writes o into core[:qsl[n_seqs]]."""
    T = proj.shape[0]
    H, K = num_heads, head_dim
    KD = H * K
    dev = proj.device
    qh = torch.empty((T, KD), dtype=torch.float32, device=dev)
    kh = torch.empty((T, KD), dtype=torch.float32, device=dev)
    bk = torch.empty((T, KD), dtype=torch.float32, device=dev)
    dec = torch.empty((T, KD), dtype=torch.float32, device=dev)
    v = torch.empty((T, KD), dtype=torch.bfloat16, device=dev)
    _gdn2_prep_varlen_kernel[(T, 3 * H)](
        proj, proj.stride(0), f, qh, kh, bk, dec, v, conv_state, conv_state.stride(0), conv_state.stride(1),
        conv_state.stride(2), conv_weight, A_log, dt_bias, qsl, state_idx, has_init, n_seqs,
        max(n_seqs, 1).bit_length() + 1, K ** -0.5, K=K, KD=KD, num_warps=4)
    # one warp per program keeps the per-step reductions over K inside a warp; narrow value tiles give a lone
    # sequence enough programs (measured, benchmarks/bench_kernels.py gdn_varlen: 128 tokens 65.8 -> 45.1 us)
    BV, nw = VARLEN_CFG or ((8, 1) if n_seqs <= 2 else (16, 1))
    _gdn2_recurrent_varlen_kernel[(n_seqs, H, K // BV)](
        qh, kh, bk, dec, v, proj, proj.stride(0), ssm_state, ssm_state.stride(0), conv_state, conv_state.stride(0),
        conv_state.stride(1), conv_state.stride(2), qsl, state_idx, has_init, core, K=K, KD=KD, BV=BV, num_warps=nw)


# ---- long prefills: the inputs of fla's chunked kernels in one pass ---------------------------------------------------
# For the chunked path the short convolutions, fla's q/k L2 norm (bf16 output, as fla's l2norm_fwd stores it) and the
# erase/write gates run in one kernel that writes fla's [T, H, K] layout directly (instead of vLLM's convolution
# kernel, a permute copy, the gate kernel and fla's two L2-norm launches); a small kernel then rewrites each sequence's
# convolution history (after every token has read the old one).

@triton.jit
def _gdn2_chunk_prep_kernel(proj_ptr, proj_stride,      # [T, 5 KD + 2 K] bf16 in_proj output (raw)
                            q_ptr, k_ptr, v_ptr, b_ptr, w_ptr,   # [T, KD] bf16 outs
                            cs_ptr, cs_s_slot, cs_s_dim, cs_s_tok,   # conv history [slots, 3 KD, 3] bf16
                            cw_ptr,                     # [3 KD, 4] fp32
                            qsl_ptr, idx_ptr, init_ptr, n_seqs, bs_iters,
                            K: tl.constexpr, KD: tl.constexpr):
    t = tl.program_id(0)
    j = tl.program_id(1)
    g = j // (KD // K)                                       # 0 q, 1 k, 2 v, 3 b, 4 w
    h = j % (KD // K)
    rows = h * K + tl.arange(0, K)
    t64 = t.to(tl.int64)
    total = tl.load(qsl_ptr + n_seqs)
    if t < total:
        if g < 3:
            lo = tl.full([], 0, tl.int32)
            hi = tl.full([], 0, tl.int32) + n_seqs - 1
            for _ in range(bs_iters):
                mid = (lo + hi + 1) // 2
                go_up = tl.load(qsl_ptr + mid) <= t
                lo = tl.where(go_up, mid, lo)
                hi = tl.where(go_up, hi, mid - 1)
            start = tl.load(qsl_ptr + lo)
            pos = t - start
            slot = tl.load(idx_ptr + lo).to(tl.int64)
            init = tl.load(init_ptr + lo) != 0
            ch = g * KD + rows
            hbase = cs_ptr + slot * cs_s_slot + ch.to(tl.int64) * cs_s_dim
            acc = tl.zeros([K], dtype=tl.float32)
            for jj in tl.static_range(4):
                p = pos - 3 + jj
                if jj < 3:
                    from_hist = p < 0
                    xin = tl.load(proj_ptr + (start + p).to(tl.int64) * proj_stride + ch, mask=p >= 0, other=0.0)
                    hv = tl.load(hbase + (3 + p) * cs_s_tok, mask=from_hist & init, other=0.0)
                    tap = tl.where(from_hist, hv, xin)
                else:
                    tap = tl.load(proj_ptr + t64 * proj_stride + ch)
                acc += tap * tl.load(cw_ptr + ch * 4 + jj)
            x = (acc / (1 + tl.exp(-acc))).to(tl.bfloat16).to(tl.float32)
            if g == 0:
                tl.store(q_ptr + t64 * KD + rows, (x * (1 / tl.sqrt(tl.sum(x * x) + 1e-6))).to(tl.bfloat16))
            elif g == 1:
                tl.store(k_ptr + t64 * KD + rows, (x * (1 / tl.sqrt(tl.sum(x * x) + 1e-6))).to(tl.bfloat16))
            else:
                tl.store(v_ptr + t64 * KD + rows, x.to(tl.bfloat16))
        else:
            raw = tl.load(proj_ptr + t64 * proj_stride + g * KD + rows).to(tl.float32)
            out = tl.where(g == 3, b_ptr, w_ptr)
            tl.store(out + t64 * KD + rows, tl.sigmoid(raw).to(tl.bfloat16))


@triton.jit
def _gdn2_conv_history_kernel(proj_ptr, proj_stride, cs_ptr, cs_s_slot, cs_s_dim, cs_s_tok, qsl_ptr, idx_ptr,
                              init_ptr, NCONV: tl.constexpr, BC: tl.constexpr):
    """each sequence's convolution history <- its last 3 raw inputs (older entries for sequences shorter than 3)"""
    s = tl.program_id(0)
    ch = tl.program_id(1) * BC + tl.arange(0, BC)
    mc = ch < NCONV
    start = tl.load(qsl_ptr + s)
    n = tl.load(qsl_ptr + s + 1) - start
    if n > 0:
        slot = tl.load(idx_ptr + s).to(tl.int64)
        init = tl.load(init_ptr + s) != 0
        hb = cs_ptr + slot * cs_s_slot + ch.to(tl.int64) * cs_s_dim
        v0 = tl.zeros([BC], dtype=tl.bfloat16)
        v1 = tl.zeros([BC], dtype=tl.bfloat16)
        v2 = tl.zeros([BC], dtype=tl.bfloat16)
        for i in tl.static_range(3):
            p = n - 3 + i
            raw = tl.load(proj_ptr + (start + p).to(tl.int64) * proj_stride + ch, mask=mc & (p >= 0), other=0.0)
            old = tl.load(hb + (3 + p) * cs_s_tok, mask=mc & (p < 0) & init, other=0.0)
            val = tl.where(p >= 0, raw, old)
            if i == 0:
                v0 = val
            elif i == 1:
                v1 = val
            else:
                v2 = val
        tl.debug_barrier()
        tl.store(hb, v0, mask=mc)
        tl.store(hb + cs_s_tok, v1, mask=mc)
        tl.store(hb + 2 * cs_s_tok, v2, mask=mc)


def gdn2_chunk_prep(proj: torch.Tensor, conv_state: torch.Tensor, conv_weight: torch.Tensor, qsl: torch.Tensor,
                    state_idx: torch.Tensor, has_init: torch.Tensor, n_seqs: int, num_heads: int, head_dim: int):
    """-> (q, k, v, b, w) [T, KD] bf16 for fla's chunked kernels (q, k L2-normalised); rewrites the histories."""
    T = proj.shape[0]
    H, K = num_heads, head_dim
    KD = H * K
    outs = [torch.empty((T, KD), dtype=torch.bfloat16, device=proj.device) for _ in range(5)]
    if T:
        _gdn2_chunk_prep_kernel[(T, 5 * H)](
            proj, proj.stride(0), *outs, conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
            conv_weight, qsl, state_idx, has_init, n_seqs, max(n_seqs, 1).bit_length() + 1, K=K, KD=KD, num_warps=4)
        BC = 512
        NCONV = conv_weight.shape[0]
        _gdn2_conv_history_kernel[(n_seqs, triton.cdiv(NCONV, BC))](
            proj, proj.stride(0), conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), qsl,
            state_idx, has_init, NCONV=NCONV, BC=BC, num_warps=4)
    return outs
