# SPDX-License-Identifier: Apache-2.0
"""Nenya's mHC (manifold-constrained hyper-connections): 4 residual streams, mixed around every sublayer.

Parameter names and shapes are the release's (`mHC_att` / `mHC_ffn` of every block), so the checkpoint loads
directly. `pre` reads the streams into the sublayer input (and RMS-normalises it with the block's norm1/norm2),
`post` mixes the streams with the Sinkhorn matrix and writes the sublayer output into them.

Two implementations with the same maths: fused Triton kernels (default) and a PyTorch reference
(NENYA_MHC_IMPL=torch), which also offers the HF release's fp64 multiplicative Sinkhorn for parity tests
(NENYA_SINKHORN=mul).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMPL = os.environ.get("NENYA_MHC_IMPL", "triton")
_SINKHORN = os.environ.get("NENYA_SINKHORN", "log")


def sinkhorn(logits: torch.Tensor, iters: int, mode: str = "log") -> torch.Tensor:
    """[..., n, n] fp32 -> doubly stochastic, columns normalised first and rows last (the release's order)."""
    if mode == "mul":
        z = logits.double()
        p = (z - z.amax(dim=(-2, -1), keepdim=True)).exp()
        for _ in range(iters):
            p = p / p.sum(dim=-2, keepdim=True)
            p = p / p.sum(dim=-1, keepdim=True)
        return p.to(logits.dtype)
    z = logits.float()
    for _ in range(iters):
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
    return z.exp()


def rms_norm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """torch.nn.RMSNorm under bf16 autocast with an fp32 weight: computed and returned in fp32."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


class NenyaMHC(nn.Module):
    def __init__(self, num_streams: int, hidden_size: int, sinkhorn_iters: int) -> None:
        super().__init__()
        n, d = num_streams, hidden_size
        self.n, self.d, self.iters = n, d, sinkhorn_iters
        f32 = torch.float32
        self.alpha_pre = nn.Parameter(torch.zeros((), dtype=f32), requires_grad=False)
        self.beta_pre = nn.Parameter(torch.zeros(n, dtype=f32), requires_grad=False)
        self.alpha_post = nn.Parameter(torch.zeros((), dtype=f32), requires_grad=False)
        self.beta_post = nn.Parameter(torch.zeros(n, dtype=f32), requires_grad=False)
        self.alpha_res = nn.Parameter(torch.zeros((), dtype=f32), requires_grad=False)
        self.beta_res = nn.Parameter(torch.zeros(n * n, dtype=f32), requires_grad=False)
        # checkpoint name `theta.weight` (bf16 [2n + n*n, n*d])
        self.theta = nn.Linear(n * d, 2 * n + n * n, bias=False, dtype=torch.bfloat16)
        self.theta.weight.requires_grad_(False)
        # packed copies for the kernels, filled by finalize() after loading
        self.register_buffer("alpha", torch.zeros(3, dtype=f32), persistent=False)
        self.register_buffer("beta", torch.zeros(2 * n + n * n, dtype=f32), persistent=False)

    @torch.no_grad()
    def finalize(self) -> None:
        self.alpha.copy_(torch.stack([self.alpha_pre, self.alpha_post, self.alpha_res]).float())
        self.beta.copy_(torch.cat([self.beta_pre, self.beta_post, self.beta_res]).float())

    def theta_out(self, x_res: torch.Tensor) -> torch.Tensor:
        # autocast's F.linear: bf16 input, bf16 weight, bf16 output
        return F.linear(x_res.reshape(x_res.shape[0], -1).to(torch.bfloat16), self.theta.weight)

    def pre(self, x_res: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float):
        """x_res [T, n, d] -> (normalised sublayer input fp32 [T, d], same in bf16, coefficients for post)."""
        th = self.theta_out(x_res)
        if _IMPL == "torch":
            return self._pre_torch(x_res, th, norm_weight, norm_eps)
        from nenya_vllm.ops.mhc_kernels import mhc_pre

        return mhc_pre(x_res, th, self.alpha, self.beta, norm_weight, norm_eps, self.iters)

    def post(self, x_res: torch.Tensor, layer_out: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        if _IMPL == "torch":
            return self._post_torch(x_res, layer_out, coef)
        from nenya_vllm.ops.mhc_kernels import mhc_post

        return mhc_post(x_res, layer_out, coef)

    # ---- PyTorch reference: the release's ops in the release's dtypes ----
    def _pre_torch(self, x_res, th, norm_weight, norm_eps):
        T, n, d = x_res.shape
        flat = x_res.reshape(T, n * d)
        r = torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + 1e-20)
        m = th.float() * r
        read_m, write_m, mix_m = m[:, :n], m[:, n:2 * n], m[:, 2 * n:]
        h_pre = torch.sigmoid(self.alpha_pre * read_m + self.beta_pre)
        x = (h_pre.unsqueeze(-1) * x_res.float()).sum(-2)
        xn = rms_norm_fp32(x, norm_weight, norm_eps)
        h_post = 2.0 * torch.sigmoid(self.alpha_post * write_m + self.beta_post)
        h_res = sinkhorn((self.alpha_res * mix_m + self.beta_res).reshape(T, n, n), self.iters, _SINKHORN)
        coef = torch.cat([h_post, h_res.reshape(T, n * n)], dim=-1)
        return xn, xn.to(torch.bfloat16), coef

    def _post_torch(self, x_res, layer_out, coef):
        T, n, d = x_res.shape
        h_post = coef[:, :n]
        h_res = coef[:, n:].reshape(T, n, n)
        write = h_post.unsqueeze(-1) * layer_out.float().unsqueeze(-2)
        mixed = (h_res.unsqueeze(-1) * x_res.float().unsqueeze(-3)).sum(-2)
        return (mixed + write).to(torch.bfloat16)
