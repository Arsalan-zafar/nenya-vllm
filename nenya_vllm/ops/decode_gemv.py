# SPDX-License-Identifier: Apache-2.0
# GEMV / skinny GEMM for decode-sized batches (a few tokens), replacing cuBLAS on Nenya's decode path.
#
# At batch 1 a cuBLAS GEMV costs ~2.5-3.5 us even for a 0.4 MB matrix and cannot take the neighbouring elementwise
# work; this kernel (one program per BN output rows, the whole reduction dimension in one tile) measured 1.1-1.7 us on
# the same shapes on an RTX 5090, and 9.5 us vs 12 us for GDN-2's 15 MB in_proj (benchmarks/bench_kernels.py gemv).
# It also runs under programmatic dependent launch (PDL): each program loads its weight tile -- which no earlier
# kernel writes -- before waiting for the previous kernel, so the weight stream overlaps the previous kernel's tail.
#
# Maths: fp32 products and fp32 accumulation over the whole reduction, one rounding of the output (cuBLAS: fp32
# accumulation, bf16 output). Optional prologues compute the input from its producers in fp32 without the
# intermediate bf16 roundings of the unfused ops (so the input is closer to the fp32 model's, never further):
#   NONE           x                                        (bf16 or fp32 input)
#   SILU_MUL       silu(a) * u              in = [a | u]    (dense SwiGLU's down projection)
#   SIGMOID_GATE   a * sigmoid(g)           in = a, aux = g (MLA's gated attention output)
#   GATED_RMSNORM  rmsnorm_G(a) * w * sigmoid(g) per group of G channels (fla's FusedRMSNormGated, GDN-2's output)
#   GATHER         table[rows[k // G], k % G]  in = rows [T, K / G] int64, aux = table [*, G]  (Engram's lookup)

import os

import torch

from vllm.triton_utils import tl, triton

NONE, SILU_MUL, SIGMOID_GATE, GATED_RMSNORM, GATHER = 0, 1, 2, 3, 4
# internal: an fp32 input (a prologue's output, computed once) fed to the tensor cores as bf16 hi + residual
F32 = 5


def _pdl_default() -> bool:
    if os.environ.get("NENYA_PDL", "1") != "1":
        return False
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


PDL = _pdl_default()
# debugging aid: kernels named here (comma-separated) are launched twice, so the change in step time is their
# in-place cost under PDL, which profilers cannot attribute (a PDL kernel starts early and waits)
DEBUG_REPEAT = set(filter(None, os.environ.get("NENYA_DEBUG_REPEAT", "").split(",")))


def repeats(name: str) -> int:
    return 2 if name in DEBUG_REPEAT else 1


# debugging aid: kernels named here (comma-separated, e.g. "gdn_in,mhc") launch without PDL
PDL_OFF = set(filter(None, os.environ.get("NENYA_PDL_OFF", "").split(",")))


def pdl(name: str) -> bool:
    return PDL and name not in PDL_OFF


# token counts up to which the decode paths use these kernels instead of cuBLAS (NENYA_GEMV_T overrides); the LM
# head's own limit (its 100K-row matrix) is LM_HEAD_MAX_T
GEMV_MAX_T = int(os.environ.get("NENYA_GEMV_T", "32"))
LM_HEAD_MAX_T = int(os.environ.get("NENYA_LM_HEAD_T", "16"))

_COUNTERS = {}


def counters(n: int, device, name: str) -> torch.Tensor:
    """Zeroed int32 counters for a kernel family's cross-program reductions ("last program of a group finishes the
    work"): every launch leaves them at zero. Allocated by the first (eager) forward, before any CUDA graph capture."""
    key = (name, str(device))
    buf = _COUNTERS.get(key)
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 4096), dtype=torch.int32, device=device)
        _COUNTERS[key] = buf
    return buf


@triton.jit
def _prologue(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, t, offs_k, mask_k, eps,
              K: tl.constexpr, KP2: tl.constexpr, G: tl.constexpr, PROLOGUE: tl.constexpr):
    """The fp32 input row of token t, [KP2] (zero past K)."""
    t64 = t.to(tl.int64)
    if PROLOGUE == 1:      # SILU_MUL
        a = tl.load(x_ptr + t64 * x_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + t64 * x_stride + K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        x = a * tl.sigmoid(a) * u
    elif PROLOGUE == 2:    # SIGMOID_GATE
        a = tl.load(x_ptr + t64 * x_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g = tl.load(aux_ptr + t64 * aux_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        x = a * tl.sigmoid(g)
    elif PROLOGUE == 3:    # GATED_RMSNORM (fla: x * rstd * w, then * sigmoid(g); rstd = 1/sqrt(mean(x^2) + eps))
        a = tl.load(x_ptr + t64 * x_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g = tl.load(aux_ptr + t64 * aux_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        w = tl.load(nw_ptr + offs_k % G, mask=mask_k, other=0.0).to(tl.float32)
        a2 = tl.reshape(a, [KP2 // G, G])
        rstd = 1 / tl.sqrt(tl.sum(a2 * a2, axis=1) / G + eps)
        x = tl.reshape(a2 * rstd[:, None], [KP2]) * w * tl.sigmoid(g)
    elif PROLOGUE == 4:    # GATHER: the concatenated table rows of token t
        r = tl.load(x_ptr + t64 * x_stride + offs_k // G, mask=mask_k, other=0)
        x = tl.load(aux_ptr + r * aux_stride + offs_k % G, mask=mask_k, other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + t64 * x_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    return x


@triton.jit
def _decode_gemv_kernel(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, w_ptr, w_stride, bias_ptr, y_ptr, y_stride,
                        T, N, N_VALID, eps,
                        K: tl.constexpr, KP2: tl.constexpr, BN: tl.constexpr, TB: tl.constexpr, G: tl.constexpr,
                        PROLOGUE: tl.constexpr, HAS_BIAS: tl.constexpr, PDL: tl.constexpr):
    pid_n = tl.program_id(0)
    t0 = tl.program_id(1) * TB
    rows = pid_n * BN + tl.arange(0, BN)
    mask_n = rows < N
    offs_k = tl.arange(0, KP2)
    mask_k = offs_k < K
    # the weight tile first: no kernel writes it, so under PDL this overlaps the previous kernel
    w = tl.load(w_ptr + rows.to(tl.int64)[:, None] * w_stride + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :], other=0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + rows, mask=mask_n, other=0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    wf = w.to(tl.float32)
    for i in tl.static_range(TB):
        t = t0 + i
        if t < T:
            x = _prologue(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, t, offs_k, mask_k, eps, K, KP2, G, PROLOGUE)
            acc = tl.sum(wf * x[None, :], axis=1)
            if HAS_BIAS:
                acc += bias
            acc = tl.where(rows < N_VALID, acc, float("-inf"))
            tl.store(y_ptr + t.to(tl.int64) * y_stride + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _x_tile(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, offs_m, mask_m, offs_k, mask_k, eps,
            K: tl.constexpr, G: tl.constexpr, PROLOGUE: tl.constexpr):
    """[BM, BK] fp32 input tile (rows offs_m, columns offs_k) with _prologue's transforms; GATED_RMSNORM needs the
    tile to be exactly one group (BK == G)."""
    m64 = offs_m.to(tl.int64)[:, None]
    mk = mask_m[:, None] & mask_k[None, :]
    if PROLOGUE == 1:      # SILU_MUL
        a = tl.load(x_ptr + m64 * x_stride + offs_k[None, :], mask=mk, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + m64 * x_stride + K + offs_k[None, :], mask=mk, other=0.0).to(tl.float32)
        x = a * tl.sigmoid(a) * u
    elif PROLOGUE == 2:    # SIGMOID_GATE
        a = tl.load(x_ptr + m64 * x_stride + offs_k[None, :], mask=mk, other=0.0).to(tl.float32)
        g = tl.load(aux_ptr + m64 * aux_stride + offs_k[None, :], mask=mk, other=0.0).to(tl.float32)
        x = a * tl.sigmoid(g)
    elif PROLOGUE == 3:    # GATED_RMSNORM over this tile (one group)
        a = tl.load(x_ptr + m64 * x_stride + offs_k[None, :], mask=mk, other=0.0).to(tl.float32)
        g = tl.load(aux_ptr + m64 * aux_stride + offs_k[None, :], mask=mk, other=0.0).to(tl.float32)
        w = tl.load(nw_ptr + offs_k % G, mask=mask_k, other=0.0).to(tl.float32)
        rstd = 1 / tl.sqrt(tl.sum(a * a, axis=1) / G + eps)
        x = a * rstd[:, None] * w[None, :] * tl.sigmoid(g)
    elif PROLOGUE == 4:    # GATHER
        r = tl.load(x_ptr + m64 * x_stride + (offs_k // G)[None, :], mask=mk, other=0)
        x = tl.load(aux_ptr + r * aux_stride + (offs_k % G)[None, :], mask=mk, other=0.0).to(tl.float32)
    elif PROLOGUE == 5:    # F32: a precomputed fp32 input
        x = tl.load(x_ptr + m64 * x_stride + offs_k[None, :], mask=mk, other=0.0)
    else:
        x = tl.load(x_ptr + m64 * x_stride + offs_k[None, :], mask=mk, other=0.0).to(tl.float32)
    return x


@triton.jit
def _prologue_rows_kernel(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, out_ptr, eps,
                          K: tl.constexpr, KP2: tl.constexpr, G: tl.constexpr, PROLOGUE: tl.constexpr,
                          PDL: tl.constexpr):
    """A prologue's input rows, once per token ([T, K]: fp32, or bf16 for GATHER's exact table rows), for the
    tensor-core path at larger token counts, where recomputing it in every output-column block costs more than the
    product."""
    t = tl.program_id(0)
    offs_k = tl.arange(0, KP2)
    mask_k = offs_k < K
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = _prologue(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, t, offs_k, mask_k, eps, K, KP2, G, PROLOGUE)
    tl.store(out_ptr + t.to(tl.int64) * K + offs_k, x.to(out_ptr.dtype.element_ty), mask=mask_k)


@triton.jit
def _decode_gemm_kernel(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, w_ptr, w_stride, bias_ptr, y_ptr, y_stride,
                        ws_ptr, cnt_ptr, T, N, N_VALID, eps,
                        K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
                        KS: tl.constexpr, SPLITK: tl.constexpr, M_PAD: tl.constexpr,
                        PROLOGUE: tl.constexpr, HAS_BIAS: tl.constexpr, PDL: tl.constexpr, STAGES: tl.constexpr):
    """The same product on the tensor cores for 5..128 tokens: [BM tokens, BK] x [BK, BN] tiles, fp32 accumulation.
    A prologue's fp32 input goes in as two bf16 MMAs, its bf16 rounding and the rounding's residual (x = hi + lo to
    ~16 bits), so the transformed input keeps the GEMV path's precision instead of the unfused ops' bf16 rounding.
    Split-K (SPLITK > 1, for outputs too narrow to fill the GPU): program (n, m, s) sums the KS-wide K slice s into
    an fp32 partial; the tile's last program to finish adds the partials in slice order (deterministic)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)
    k_lo = pid_s * KS
    k_hi = tl.minimum(k_lo + KS, K)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < T
    wrow = w_ptr + offs_n.to(tl.int64)[:, None] * w_stride
    # the first weight chunk before the PDL wait
    offs_k0 = k_lo + tl.arange(0, BK)
    w0 = tl.load(wrow + offs_k0[None, :], mask=mask_n[:, None] & (offs_k0 < k_hi)[None, :], other=0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    SPLIT: tl.constexpr = (PROLOGUE == 1) or (PROLOGUE == 2) or (PROLOGUE == 3) or (PROLOGUE == 5)
    x0 = _x_tile(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, offs_m, mask_m, offs_k0, offs_k0 < k_hi, eps, K, G,
                 PROLOGUE)
    x0h = x0.to(tl.bfloat16)
    acc = tl.dot(x0h, tl.trans(w0))
    if SPLIT:
        acc = tl.dot((x0 - x0h.to(tl.float32)).to(tl.bfloat16), tl.trans(w0), acc)
    for k0 in tl.range(k_lo + BK, k_hi, BK, num_stages=STAGES):
        offs_k = k0 + tl.arange(0, BK)
        mask_k = offs_k < k_hi
        w = tl.load(wrow + offs_k[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        x = _x_tile(x_ptr, x_stride, aux_ptr, aux_stride, nw_ptr, offs_m, mask_m, offs_k, mask_k, eps, K, G,
                    PROLOGUE)
        xh = x.to(tl.bfloat16)
        acc = tl.dot(xh, tl.trans(w), acc)
        if SPLIT:
            acc = tl.dot((x - xh.to(tl.float32)).to(tl.bfloat16), tl.trans(w), acc)
    last = True
    if SPLITK > 1:
        # workspace [SPLITK, M_PAD, N] fp32; one counter per output tile, left at zero by the tile's last program
        wsp = ws_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
        tl.store(wsp + pid_s * (M_PAD * N), acc, mask=mask_n[None, :])
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(0) + pid_n
        last = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel") == SPLITK - 1
        if last:
            acc = tl.load(wsp, mask=mask_n[None, :], other=0.0, cache_modifier=".cg")
            for s in tl.static_range(1, SPLITK):
                acc += tl.load(wsp + s * (M_PAD * N), mask=mask_n[None, :], other=0.0, cache_modifier=".cg")
            tl.store(cnt_ptr + tile, 0)
    if last:
        if HAS_BIAS:
            acc += bias[None, :]
        acc = tl.where((offs_n < N_VALID)[None, :], acc, float("-inf"))
        tl.store(y_ptr + offs_m.to(tl.int64)[:, None] * y_stride + offs_n[None, :], acc.to(y_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


# the tensor-core path from this many tokens on (NENYA_GEMM_T overrides)
GEMM_MIN_T = int(os.environ.get("NENYA_GEMM_T", "5"))
# split-K of the tensor-core path: 0 = the measured table (_gemm_config), N = N slices, -1 = enough slices for
# ~SPLITK_PROGRAMS programs
SPLITK_DEFAULT = int(os.environ.get("NENYA_SPLITK", "0"))
SPLITK_PROGRAMS = int(os.environ.get("NENYA_SPLITK_PROGRAMS", "340"))
# from this many tokens a prologue runs once in its own kernel instead of in every output-column block
PROLOGUE_ROWS_T = int(os.environ.get("NENYA_PROLOGUE_ROWS_T", "8"))


def _gemm_config(N: int, K: int, T: int, prologue: int) -> tuple[int, int, int, int, int]:
    """(BN, BK, num_warps, num_stages, split-K slices) of the tensor-core path, measured per decode shape on an RTX
    5090 (benchmarks/bench_kernels.py gemm_split; prologue inputs computed once by _prologue_rows_kernel from
    PROLOGUE_ROWS_T tokens). Narrow outputs (N <= 1152: o_proj, out_proj, w_out, MLP down) split K to fill the GPU;
    wide ones (N >= 4096) do not need to."""
    if T <= 16:
        if N <= 1152:
            return 32, 128, 4, 2, 8
        if N <= 3072:
            return 16, 64, 4, 3, 8
        return (16, 128, 4, 2, 4) if N <= 8192 else (32, 64, 4, 3, 1)
    if T <= 32:
        if N <= 1152:
            return 32, 64, 4, 3, 4
        if N <= 3072:
            return 32, 64, 4, 3, 3
        return 32, 64, 4, 3, 1
    if N <= 1152:
        # fp32 inputs (a prologue's output: two MMAs per tile) want fewer slices than bf16 ones
        return (32, 64, 4, 3, 4) if prologue == F32 else (32, 128, 4, 2, 8)
    if N <= 3072:
        return 32, 64, 4, 3, 2
    return 32, 64, 4, 3, 1


def _config(N: int, K: int, T: int) -> tuple[int, int]:
    """(BN, num_warps): measured on an RTX 5090 for K in 128..1536 (benchmarks/bench_kernels.py gemv)."""
    kp2 = triton.next_power_of_2(K)
    if kp2 <= 256:
        return 16, 4
    bn = 4 if T <= 4 else 8
    return max(1, min(bn, 8192 // kp2)), 8


def decode_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *, prologue: int = NONE,
                  aux: torch.Tensor | None = None, norm_weight: torch.Tensor | None = None, eps: float = 0.0,
                  group: int = 1, out: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
                  n_valid: int | None = None, config: tuple | None = None) -> torch.Tensor:
    """y = prologue(x) @ weight^T (+ bias) for a few rows. x: [T, K] ([T, 2K] for SILU_MUL, [T, K / G] row ids for
    GATHER), rows may be strided (unit last stride); weight [N, K] row-major; returns [T, N] in out_dtype (default:
    weight's dtype). Output columns >= n_valid are -inf (the LM head's untrained padding rows)."""
    T = x.shape[0]
    N, K = weight.shape
    assert weight.stride(1) == 1 and x.stride(-1) == 1
    if out is None:
        out = torch.empty((T, N), dtype=out_dtype or weight.dtype, device=x.device)
    if T == 0:
        return out
    kp2 = triton.next_power_of_2(K)
    bn, nw = _config(N, K, T)
    tb = min(T, 16)
    if aux is None:
        aux = x
    if norm_weight is None:
        norm_weight = weight
    if prologue == GATED_RMSNORM:
        assert K % group == 0 and kp2 % group == 0
    if T >= GEMM_MIN_T:
        if prologue not in (NONE, F32) and T >= PROLOGUE_ROWS_T:
            # the transformed input once ([T, K]), then the plain product (fp32 input: hi + residual MMAs)
            xs = torch.empty((T, K), dtype=torch.bfloat16 if prologue == GATHER else torch.float32, device=x.device)
            for _ in range(repeats("gemv_pro")):
                _prologue_rows_kernel[(T,)](x, x.stride(0), aux, aux.stride(0), norm_weight, xs, eps, K=K, KP2=kp2,
                                            G=group, PROLOGUE=prologue, PDL=pdl("prologue"), launch_pdl=pdl("prologue"), num_warps=4)
            x = aux = xs
            prologue = NONE if prologue == GATHER else F32
        BM = 16 if T <= 16 else 32 if T <= 32 else 64
        cfg = config or _gemm_config(N, K, T, prologue)
        BN, BK, nw, stages = cfg[:4]
        if prologue == GATED_RMSNORM:
            BK = group
        if BM * BK >= 64 * 128:
            stages = min(stages, 2)
        n_tiles = triton.cdiv(N, BN) * triton.cdiv(T, BM)
        splitk = cfg[4] if len(cfg) > 4 else 1
        if SPLITK_DEFAULT:      # override (NENYA_SPLITK): a fixed slice count, or -1 for ~SPLITK_PROGRAMS programs
            splitk = SPLITK_DEFAULT
        if splitk < 0:
            splitk = max(1, min(triton.cdiv(K, BK), round(SPLITK_PROGRAMS / n_tiles)))
        KS = triton.cdiv(triton.cdiv(K, BK), splitk) * BK
        splitk = triton.cdiv(K, KS)
        M_PAD = triton.cdiv(T, BM) * BM
        ws = torch.empty((splitk, M_PAD, N), dtype=torch.float32, device=x.device) if splitk > 1 else out
        cnt = counters(n_tiles, x.device, "decode_gemm") if splitk > 1 else out
        for _ in range(repeats("gemv_%d_%d" % (N, K)) * repeats("gemv")):
            _decode_gemm_kernel[(triton.cdiv(N, BN), triton.cdiv(T, BM), splitk)](
                x, x.stride(0), aux, aux.stride(0), norm_weight, weight, weight.stride(0),
                bias if bias is not None else weight, out, out.stride(0), ws, cnt, T, N,
                N if n_valid is None else n_valid, eps, K=K, BM=BM, BN=BN, BK=BK, G=group, KS=KS, SPLITK=splitk,
                M_PAD=M_PAD, PROLOGUE=prologue, HAS_BIAS=bias is not None, PDL=pdl("gemm"), STAGES=stages, launch_pdl=pdl("gemm"),
                num_warps=nw)
        return out
    for _ in range(repeats("gemv_%d_%d" % (N, K)) * repeats("gemv")):
        _decode_gemv_kernel[(triton.cdiv(N, bn), triton.cdiv(T, tb))](
            x, x.stride(0), aux, aux.stride(0), norm_weight, weight, weight.stride(0),
            bias if bias is not None else weight, out, out.stride(0), T, N, N if n_valid is None else n_valid, eps,
            K=K, KP2=kp2, BN=bn, TB=tb, G=group, PROLOGUE=prologue, HAS_BIAS=bias is not None, PDL=pdl("gemv"),
            launch_pdl=pdl("gemv"), num_warps=nw)
    return out
