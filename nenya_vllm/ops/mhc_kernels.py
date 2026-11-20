# SPDX-License-Identifier: Apache-2.0
# Nenya's manifold-constrained hyper-connections (mHC) as two fused Triton kernels per sublayer.
#
# The maths and dtype flow are the release's (modeling_nenya.NenyaMHC under bf16 autocast):
#   m      = bf16(theta(bf16(flat))) * rsqrt(mean(flat^2) + 1e-20)        flat = the n*d residual streams
#   pre    = sigmoid(a_pre * m[:n] + b_pre)                -> x = sum_i pre_i * x_i  -> RMSNorm (fp32 weight)
#   post   = 2 * sigmoid(a_post * m[n:2n] + b_post)
#   H      = Sinkhorn(a_res * m[2n:] + b_res, 20 iterations: normalise columns, then rows)
#   x_new_i = bf16( sum_j H_ij x_j + post_i * layer_out )
# The Sinkhorn runs in the log domain in fp32, which is training's own op (the HF release uses an fp64
# multiplicative form that lands within 1.4e-6 of it). theta(flat) is a cuBLAS GEMM outside the kernels.

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mhc_pre_kernel(
    x_ptr,          # [T, N, D] residual streams (bf16, or fp32 before the first sublayer)
    theta_ptr,      # [T, NM] bf16 theta(flat), NM = 2N + N*N
    alpha_ptr,      # [3] fp32: alpha_pre, alpha_post, alpha_res
    beta_ptr,       # [NM] fp32: beta_pre (N), beta_post (N), beta_res (N*N, row-major [i, j])
    norm_w_ptr,     # [D] fp32
    out_f32_ptr,    # [T, D] normalised layer input, fp32
    out_bf16_ptr,   # [T, D] the same in bf16 (what the matmuls of the sublayer read)
    coef_ptr,       # [T, N + N*N] fp32: post gates, then the Sinkhorn matrix H (row-major)
    norm_eps,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ITERS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    NM: tl.constexpr = 2 * N + N * N
    offs_n = tl.arange(0, N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    x = tl.load(x_ptr + t * N * D + offs_n[:, None] * D + offs_d[None, :], mask=mask_d[None, :],
                other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(tl.sum(x * x, axis=1), axis=0) / (N * D) + 1e-20)

    a_pre = tl.load(alpha_ptr)
    a_post = tl.load(alpha_ptr + 1)
    a_res = tl.load(alpha_ptr + 2)

    m_pre = tl.load(theta_ptr + t * NM + offs_n).to(tl.float32) * r
    h_pre = tl.sigmoid(a_pre * m_pre + tl.load(beta_ptr + offs_n))
    xin = tl.sum(h_pre[:, None] * x, axis=0)                                   # [BLOCK_D]
    var = tl.sum(xin * xin, axis=0) / D
    w = tl.load(norm_w_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    xn = xin * (1.0 / tl.sqrt(var + norm_eps)) * w
    tl.store(out_f32_ptr + t * D + offs_d, xn, mask=mask_d)
    tl.store(out_bf16_ptr + t * D + offs_d, xn.to(tl.bfloat16), mask=mask_d)

    m_post = tl.load(theta_ptr + t * NM + N + offs_n).to(tl.float32) * r
    h_post = 2.0 * tl.sigmoid(a_post * m_post + tl.load(beta_ptr + N + offs_n))
    tl.store(coef_ptr + t * (N + N * N) + offs_n, h_post)

    offs_i = tl.arange(0, N)[:, None]
    offs_j = tl.arange(0, N)[None, :]
    m_res = tl.load(theta_ptr + t * NM + 2 * N + offs_i * N + offs_j).to(tl.float32) * r
    z = a_res * m_res + tl.load(beta_ptr + 2 * N + offs_i * N + offs_j)
    for _ in tl.static_range(ITERS):
        # columns (torch dim=-2): logsumexp over i
        cmax = tl.max(z, axis=0)
        z = z - (cmax[None, :] + tl.log(tl.sum(tl.exp(z - cmax[None, :]), axis=0))[None, :])
        # rows (dim=-1): logsumexp over j
        rmax = tl.max(z, axis=1)
        z = z - (rmax[:, None] + tl.log(tl.sum(tl.exp(z - rmax[:, None]), axis=1))[:, None])
    tl.store(coef_ptr + t * (N + N * N) + N + offs_i * N + offs_j, tl.exp(z))


@triton.jit
def _mhc_post_kernel(
    x_ptr,          # [T, N, D] residual streams before this sublayer
    layer_ptr,      # [T, D] sublayer output (bf16)
    coef_ptr,       # [T, N + N*N] fp32 from _mhc_pre_kernel
    out_ptr,        # [T, N, D] bf16 new residual streams
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1)
    offs_n = tl.arange(0, N)
    offs_d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    base = coef_ptr + t * (N + N * N)
    h_post = tl.load(base + offs_n)
    layer = tl.load(layer_ptr + t * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    acc = h_post[:, None] * layer[None, :]
    for j in tl.static_range(N):
        h_col = tl.load(base + N + offs_n * N + j)                               # H[:, j]
        x_j = tl.load(x_ptr + t * N * D + j * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        acc += h_col[:, None] * x_j[None, :]
    tl.store(out_ptr + t * N * D + offs_n[:, None] * D + offs_d[None, :], acc.to(tl.bfloat16),
             mask=mask_d[None, :])


def mhc_pre(x_res: torch.Tensor, theta_out: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            norm_weight: torch.Tensor, norm_eps: float, iters: int):
    """x_res [T, N, D]; theta_out [T, 2N+N*N] bf16 -> (x_norm fp32 [T, D], x_norm bf16 [T, D], coef fp32)."""
    T, N, D = x_res.shape
    x_res = x_res.contiguous()
    theta_out = theta_out.contiguous()
    out_f32 = torch.empty((T, D), dtype=torch.float32, device=x_res.device)
    out_bf16 = torch.empty((T, D), dtype=torch.bfloat16, device=x_res.device)
    coef = torch.empty((T, N + N * N), dtype=torch.float32, device=x_res.device)
    if T:
        _mhc_pre_kernel[(T,)](x_res, theta_out, alpha, beta, norm_weight, out_f32, out_bf16, coef, norm_eps,
                              N=N, D=D, BLOCK_D=triton.next_power_of_2(D), ITERS=iters, num_warps=4)
    return out_f32, out_bf16, coef


def mhc_post(x_res: torch.Tensor, layer_out: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """New residual streams [T, N, D] bf16."""
    T, N, D = x_res.shape
    out = torch.empty((T, N, D), dtype=torch.bfloat16, device=x_res.device)
    if T:
        block_d = 256
        _mhc_post_kernel[(T, triton.cdiv(D, block_d))](x_res.contiguous(), layer_out.contiguous(), coef, out,
                                                       N=N, D=D, BLOCK_D=block_d, num_warps=4)
    return out
