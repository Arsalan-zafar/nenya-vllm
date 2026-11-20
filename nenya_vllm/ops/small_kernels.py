# SPDX-License-Identifier: Apache-2.0
# Small fused Triton kernels that replace chains of PyTorch elementwise ops (each launch costs a few microseconds of
# GPU time inside a CUDA graph, and a decode step runs ~1,500 of them). Same maths and rounding points as the
# PyTorch code they replace.

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm_fp32w_kernel(x_ptr, w_ptr, out_ptr, x_stride, out_stride, eps, D: tl.constexpr,
                           BLOCK_D: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * (1.0 / tl.sqrt(var + eps)) * w
    tl.store(out_ptr + row * out_stride + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def rms_norm_fp32w(x: torch.Tensor, weight: torch.Tensor, eps: float, out_dtype: torch.dtype) -> torch.Tensor:
    """torch.nn.RMSNorm with an fp32 weight under autocast (computed in fp32), cast once to out_dtype.
    x: [..., D] with a unit last-dim stride (rows may be strided)."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D) if x.dim() != 2 else x
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    out = torch.empty((x2.shape[0], D), dtype=out_dtype, device=x.device)
    if x2.shape[0]:
        _rms_norm_fp32w_kernel[(x2.shape[0],)](x2, weight, out, x2.stride(0), out.stride(0), eps, D=D,
                                               BLOCK_D=triton.next_power_of_2(D), num_warps=4)
    return out.view(*x.shape[:-1], D)


@triton.jit
def _gdn2_gates_kernel(proj_ptr, b_ptr, w_ptr, proj_stride, B_OFF: tl.constexpr, W_OFF: tl.constexpr,
                       KD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < KD
    bv = tl.load(proj_ptr + row * proj_stride + B_OFF + offs, mask=mask, other=0.0).to(tl.float32)
    wv = tl.load(proj_ptr + row * proj_stride + W_OFF + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(b_ptr + row * KD + offs, tl.sigmoid(bv).to(b_ptr.dtype.element_ty), mask=mask)
    tl.store(w_ptr + row * KD + offs, tl.sigmoid(wv).to(w_ptr.dtype.element_ty), mask=mask)


def gdn2_gates(proj: torch.Tensor, b_off: int, w_off: int, kd: int) -> tuple[torch.Tensor, torch.Tensor]:
    """sigmoid of the erase (b) and write (w) gate pre-activations, read from the in-projection output and written
    as two contiguous [T, kd] tensors in proj's dtype (torch.sigmoid on bf16 rounds the same way)."""
    T = proj.shape[0]
    b = torch.empty((T, kd), dtype=proj.dtype, device=proj.device)
    w = torch.empty((T, kd), dtype=proj.dtype, device=proj.device)
    if T:
        block = 512
        _gdn2_gates_kernel[(T, triton.cdiv(kd, block))](proj, b, w, proj.stride(0), B_OFF=b_off, W_OFF=w_off,
                                                       KD=kd, BLOCK=block, num_warps=4)
    return b, w


@triton.jit
def _moe_route_kernel(logits_ptr, bias_ptr, weights_ptr, ids_ptr, logits_stride, E: tl.constexpr,
                      K: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E
    logits = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=0.0).to(tl.float32)
    # Rows that are CUDA-graph padding can carry garbage (NaN/inf) activations: keep every chosen id a valid
    # expert and every weight finite for them. Real rows never hit these guards.
    logits = tl.where((logits == logits) & (tl.abs(logits) < 3.0e38), logits, -1.0e30)
    biased = tl.where(mask, logits + tl.load(bias_ptr + offs, mask=mask, other=0.0), float("-inf"))
    koffs = tl.arange(0, BLOCK_K)
    chosen_logit = tl.full([BLOCK_K], -1.0e30, tl.float32)
    chosen_id = tl.zeros([BLOCK_K], dtype=tl.int32)
    for k in tl.static_range(K):
        m = tl.max(biased, axis=0)
        # first index reaching the max (torch.topk keeps the lower index on exact ties in practice)
        idx = tl.min(tl.where(mask & (biased == m), offs, E - 1), axis=0)
        lg = tl.sum(tl.where(offs == idx, logits, 0.0), axis=0)
        chosen_logit = tl.where(koffs == k, lg, chosen_logit)
        chosen_id = tl.where(koffs == k, idx.to(tl.int32), chosen_id)
        biased = tl.where(offs == idx, float("-inf"), biased)
    kmask = koffs < K
    mx = tl.max(tl.where(kmask, chosen_logit, -1.0e30), axis=0)
    ex = tl.where(kmask, tl.exp(chosen_logit - mx), 0.0)
    tl.store(weights_ptr + row * K + koffs, ex / tl.sum(ex, axis=0), mask=kmask)
    tl.store(ids_ptr + row * K + koffs, chosen_id, mask=kmask)


def moe_route(logits: torch.Tensor, bias: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Nenya's routing in one kernel: top-k of (logits + bias), weights = softmax of the chosen UNBIASED logits.
    logits: [T, E] fp32. Returns (weights fp32 [T, k], ids int32 [T, k])."""
    T, E = logits.shape
    weights = torch.empty((T, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((T, k), dtype=torch.int32, device=logits.device)
    if T:
        _moe_route_kernel[(T,)](logits, bias, weights, ids, logits.stride(0), E=E, K=k,
                                BLOCK_E=triton.next_power_of_2(E), BLOCK_K=triton.next_power_of_2(k),
                                num_warps=4)
    return weights, ids


@triton.jit
def _embed_norm_streams_kernel(ids_ptr, emb_ptr, emb_stride, w_ptr, out_ptr, eps, N: tl.constexpr, D: tl.constexpr,
                               BLOCK_D: tl.constexpr, PDL: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    tok = tl.load(ids_ptr + t).to(tl.int64)
    x = tl.load(emb_ptr + tok * emb_stride + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    y = x * (1.0 / tl.sqrt(var + eps)) * w
    for i in tl.static_range(N):
        tl.store(out_ptr + (t * N + i) * D + offs, y, mask=mask)


def embed_norm_streams(input_ids: torch.Tensor, emb_weight: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                       n: int) -> torch.Tensor:
    """rms_norm_fp32w(F.embedding(ids, emb_weight)) copied into n residual streams: [T, n, D] fp32, one kernel
    (the same maths as _rms_norm_fp32w_kernel)."""
    from nenya_vllm.ops.decode_gemv import PDL, pdl

    T = input_ids.shape[0]
    D = emb_weight.shape[1]
    out = torch.empty((T, n, D), dtype=torch.float32, device=emb_weight.device)
    if T:
        _embed_norm_streams_kernel[(T,)](input_ids, emb_weight, emb_weight.stride(0), norm_weight, out, eps, N=n,
                                         D=D, BLOCK_D=triton.next_power_of_2(D), PDL=pdl("embed"), launch_pdl=pdl("embed"),
                                         num_warps=4)
    return out
