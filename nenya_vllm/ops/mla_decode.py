# SPDX-License-Identifier: Apache-2.0
# Nenya's MLA decode step in five kernels instead of ~17 (layers/mla.py routes decode-only batches here):
#   1. [W_KVD | W_query | W_gate] GEMV                                                      (ops/decode_gemv.py)
#   2. prep: kv_norm (fp32 RMSNorm) + the latent row into the paged cache (vLLM's concat_and_cache_mla), and the
#      query absorbed into the latent space per head (q_nope W_UK) next to its shared-key part
#   3. vLLM's split-KV decode attention, stage 1 (TRITON_MLA's kernel, unchanged)
#   4. stage 2 (merge of the splits, vLLM's formula) + the value up-projection per head (W_UV)
#   5. out_proj with the output gate attn * sigmoid(g) computed in its input prologue, + bias  (ops/decode_gemv.py)
# Same maths and rounding points as vLLM's MLA decode path: the absorbed query and the merged attention output are
# rounded to bf16 where vLLM's bmm / stage-2 store round them; the value up-projection's output too.

import torch

from vllm.triton_utils import tl, triton

from nenya_vllm.ops.decode_gemv import PDL, pdl, repeats


@triton.jit
def _mla_prep_kernel(kqg_ptr, kqg_stride,                 # [T, (L + R) + N (P + R) + ...] bf16 GEMV output
                     kvn_w_ptr, eps,                      # [L] fp32 kv_norm weight
                     cache_ptr, c_s_block, c_s_entry,     # paged latent cache [blocks, PAGE, L + R] bf16
                     slot_ptr,                            # [T] int64 slot mapping (< 0: padding, not written)
                     wuk_ptr, wuk_s_h, wuk_s_p, wuk_s_l,  # W_UK_T [N, P, L] bf16 (any strides)
                     q_ptr,                               # [T, N, L + R] bf16 out: absorbed query
                     PAGE: tl.constexpr, L: tl.constexpr, LP2: tl.constexpr, R: tl.constexpr, N: tl.constexpr,
                     P: tl.constexpr, PP2: tl.constexpr, LC: tl.constexpr, PDL: tl.constexpr):
    # grid (T, N * L / LC + 1): programs j < N * L / LC absorb LC latent dims of one head; the last one does the
    # kv_norm and the cache row
    t = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    NLC: tl.constexpr = L // LC
    offs_r = tl.arange(0, R)
    row = kqg_ptr + t * kqg_stride
    if j < N * NLC:
        # ql_nope = q_nope @ W_UK_T[h] (vLLM's bmm: fp32 sums, bf16 output). The tile is read as [LC, P]: in vLLM's
        # W_UK_T view the P axis is the contiguous one (stride 1), the L axis strided
        h = j // NLC
        offs_l = (j % NLC) * LC + tl.arange(0, LC)
        offs_p = tl.arange(0, PP2)
        mp = offs_p < P
        w = tl.load(wuk_ptr + h * wuk_s_h + offs_l[:, None] * wuk_s_l + offs_p[None, :] * wuk_s_p,
                    mask=mp[None, :], other=0.0)
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        qrow = row + (L + R) + h * (P + R)
        qn = tl.load(qrow + offs_p, mask=mp, other=0.0).to(tl.float32)
        acc = tl.sum(w.to(tl.float32) * qn[None, :], axis=1)
        qout = q_ptr + (t * N + h) * (L + R)
        tl.store(qout + offs_l, acc.to(tl.bfloat16))
        if j % NLC == 0:
            tl.store(qout + L + offs_r, tl.load(qrow + P + offs_r))
    else:
        # kv_norm (torch.nn.RMSNorm with an fp32 weight under autocast, stored bf16) + the cache row
        offs_k = tl.arange(0, LP2)
        mk = offs_k < L
        wn = tl.load(kvn_w_ptr + offs_k, mask=mk, other=0.0)
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        slot = tl.load(slot_ptr + t)
        if slot >= 0:
            kv = tl.load(row + offs_k, mask=mk, other=0.0).to(tl.float32)
            var = tl.sum(kv * kv, axis=0) / L
            y = kv * (1.0 / tl.sqrt(var + eps)) * wn
            dst = cache_ptr + (slot // PAGE) * c_s_block + (slot % PAGE) * c_s_entry
            tl.store(dst + offs_k, y.to(tl.bfloat16), mask=mk)
            tl.store(dst + L + offs_r, tl.load(row + L + offs_r))


@triton.jit
def _mla_stage2_vup_kernel(mid_ptr, s_mb, s_mh, s_ms,           # split partials [B, N, S, L + 1] fp32 (stage 1)
                           seqlen_ptr,                          # [B] int32
                           wuv_ptr, wuv_s_h, wuv_s_l, wuv_s_v,  # W_UV [N, L, V] bf16 (any strides)
                           out_ptr, out_stride,                 # [B, N V] bf16
                           NUM_KV_SPLITS: tl.constexpr, L: tl.constexpr, LP2: tl.constexpr, V: tl.constexpr,
                           VC: tl.constexpr, N: tl.constexpr, PDL: tl.constexpr):
    # grid (B, N, V / VC): each program merges the head's splits (cheap, [L] values) and up-projects VC of its V
    # outputs -- a [L, VC] weight tile (the whole [L, V] one would spill registers)
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_v = tl.program_id(2) * VC + tl.arange(0, VC)
    offs_l = tl.arange(0, LP2)
    ml = offs_l < L
    w = tl.load(wuv_ptr + h * wuv_s_h + offs_l[:, None] * wuv_s_l + offs_v[None, :] * wuv_s_v, mask=ml[:, None],
                other=0.0)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # vLLM's _fwd_kernel_stage2: merge the splits by their log-sum-exp
    seq_len = tl.load(seqlen_ptr + b)
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([LP2], dtype=tl.float32)
    offs = b.to(tl.int64) * s_mb + h * s_mh + offs_l
    offs_lse = b.to(tl.int64) * s_mb + h * s_mh + L
    for split in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
        start = kv_len_per_split * split
        end = tl.minimum(start + kv_len_per_split, seq_len)
        if end > start:
            tv = tl.load(mid_ptr + offs + split * s_ms, mask=ml, other=0.0)
            tlogic = tl.load(mid_ptr + offs_lse + split * s_ms)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max
    o = (acc / e_sum).to(tl.bfloat16).to(tl.float32)        # stage 2 stores bf16
    # value up-projection (vLLM's bmm with W_UV: fp32 sums, bf16 output)
    y = tl.sum(w.to(tl.float32) * o[:, None], axis=0)
    tl.store(out_ptr + b.to(tl.int64) * out_stride + h * V + offs_v, y.to(tl.bfloat16))


def mla_decode_prep(kqg: torch.Tensor, kv_norm_weight: torch.Tensor, eps: float, kv_cache: torch.Tensor,
                    slot_mapping: torch.Tensor, W_UK_T: torch.Tensor, latent: int, rope: int, heads: int,
                    nope: int) -> torch.Tensor:
    """kqg [T, ...] -> absorbed query [T, heads, latent + rope] bf16; writes each token's normalised latent row and
    shared key into kv_cache ([blocks, page, latent + rope]) at its slot."""
    T = kqg.shape[0]
    q = torch.empty((T, heads, latent + rope), dtype=torch.bfloat16, device=kqg.device)
    if T:
        assert kv_cache.dim() == 3 and kv_cache.stride(2) == 1 and kqg.stride(1) == 1
        for _ in range(repeats("mla_prep")):
            LC = 64
            assert latent % LC == 0
            _mla_prep_kernel[(T, heads * (latent // LC) + 1)](
                kqg, kqg.stride(0), kv_norm_weight, eps, kv_cache, kv_cache.stride(0), kv_cache.stride(1),
                slot_mapping, W_UK_T, W_UK_T.stride(0), W_UK_T.stride(1), W_UK_T.stride(2), q,
                PAGE=kv_cache.shape[1], L=latent, LP2=triton.next_power_of_2(latent), R=rope, N=heads, P=nope,
                PP2=triton.next_power_of_2(nope), LC=LC, PDL=pdl("mla_prep"), launch_pdl=pdl("mla_prep"), num_warps=4)
    return q


def mla_stage2_vup(attn_logits: torch.Tensor, seq_lens: torch.Tensor, W_UV: torch.Tensor, out: torch.Tensor,
                   latent: int, heads: int, v_dim: int) -> None:
    """out[:B] [B, heads * v_dim] bf16 = per head bf16(bf16(merged attention) @ W_UV[h])."""
    B = seq_lens.shape[0]
    if B:
        for _ in range(repeats("mla_s2")):
            VC = 16
            assert v_dim % VC == 0
            _mla_stage2_vup_kernel[(B, heads, v_dim // VC)](
                attn_logits, attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2), seq_lens,
                W_UV, W_UV.stride(0), W_UV.stride(1), W_UV.stride(2), out, out.stride(0),
                NUM_KV_SPLITS=attn_logits.shape[2], L=latent, LP2=triton.next_power_of_2(latent), V=v_dim, VC=VC,
                N=heads, PDL=pdl("mla_s2"), launch_pdl=pdl("mla_s2"), num_warps=4)
