# SPDX-License-Identifier: Apache-2.0
"""Nenya's Multi-head Latent Attention on vLLM's MLA layer.

Release module (modeling_nenya.NenyaMLA) -> vLLM:
    W_KVD  [352, 960]  -> w_kqg rows 0:352      (output = [latent 320 | shared key 32], DeepSeek's order)
    kv_norm (fp32)     -> RMSNorm on the latent, computed in fp32 (autocast), stored bf16 in the KV cache
    W_query [1344,960] -> w_kqg rows 352:1696   (per head [80 nope | 32 "rope" part, which is NOT rotated])
    W_gate  [960, 960] -> w_kqg rows 1696:2656  (output * sigmoid(gate))
    W_KU, W_VU         -> kv_b_proj            (interleaved per head: [k_nope_h (80); v_h (80)] x 12)
    out_proj (bias)    -> o_proj
The three input projections read the same input, so they are one matrix (one GEMM). No positional encoding, so no
rotary embedding. Decoding runs vLLM's TRITON_MLA backend (absorbed attention over the 352-wide latent cache) --
for decode-only batches of up to GEMV_MAX_T tokens through ops/mla_decode.py's five kernels around its stage-1
kernel; prefill runs FlashAttention with V zero-padded to the QK width (mla_prefill.py).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import CacheConfig
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

from nenya_vllm.ops.small_kernels import rms_norm_fp32w
from nenya_vllm.mla_prefill import NenyaFlashAttnPrefillBackend

# Nenya's latent KV row: 320 latent + 32 shared key.
NENYA_MLA_HEAD_SIZE = 352
FUSED = os.environ.get("NENYA_FUSED_MLA", os.environ.get("NENYA_FUSED", "1")) == "1"


def _allow_nenya_mla_head_size() -> None:
    """vLLM's MLA metadata validates the latent width against MLACommonBackend's list (320, 512, 576) whatever
    backend runs it. Nenya always runs TRITON_MLA, whose kernels take any width, so 352 is added to that list
    (in the processes that build a Nenya model only: this module is imported there)."""
    from vllm.model_executor.layers.attention.mla_attention import MLACommonBackend

    original = MLACommonBackend.get_supported_head_sizes.__func__
    if getattr(original, "_nenya_patched", False):
        return

    def get_supported_head_sizes(cls) -> list[int]:
        sizes = list(original(cls))
        return sizes if (not sizes or NENYA_MLA_HEAD_SIZE in sizes) else sizes + [NENYA_MLA_HEAD_SIZE]

    get_supported_head_sizes._nenya_patched = True
    MLACommonBackend.get_supported_head_sizes = classmethod(get_supported_head_sizes)


def _route_nenya_context_gather() -> None:
    """Chunked prefill gathers a request's earlier latent rows from the paged cache with
    ops.gather_and_maybe_dequant_cache, a CUDA kernel compiled for 320/512/576-wide rows only. For Nenya's 352-wide
    unquantized cache the "dequant" is a plain copy, which ops.cp_gather_cache does for any row width (same
    block-table / cu_seq_lens / seq_starts semantics; its row count is the request count)."""
    import vllm._custom_ops as ops

    original = ops.gather_and_maybe_dequant_cache
    if getattr(original, "_nenya_patched", False):
        return

    def gather_and_maybe_dequant_cache(src_cache, dst, block_table, cu_seq_lens, token_to_seq, num_tokens,
                                       kv_cache_dtype, scale, seq_starts=None):
        if dst.size(-1) == NENYA_MLA_HEAD_SIZE and kv_cache_dtype in ("auto", "bfloat16", "float16"):
            ops.cp_gather_cache(src_cache, dst[:num_tokens], block_table, cu_seq_lens,
                                cu_seq_lens.numel() - 1, seq_starts)
            return
        original(src_cache, dst, block_table, cu_seq_lens, token_to_seq, num_tokens, kv_cache_dtype, scale,
                 seq_starts)

    gather_and_maybe_dequant_cache._nenya_patched = True
    ops.gather_and_maybe_dequant_cache = gather_and_maybe_dequant_cache


_allow_nenya_mla_head_size()
_route_nenya_context_gather()


class NenyaMLA(nn.Module):
    def __init__(self, config, cache_config: CacheConfig | None, quant_config: QuantizationConfig | None,
                 prefix: str) -> None:
        super().__init__()
        d = config.hidden_size
        self.num_heads = H = config.num_attention_heads
        self.head_dim = hd = config.hidden_size // H                 # 80
        self.shared_key_dim = kr = config.mla_shared_key_dim          # 32
        self.latent_dim = L = config.mla_latent_dim                   # 320
        self.kv_norm_eps = config.mla_kv_norm_eps
        bias = config.attention_bias

        if bias:
            raise NotImplementedError("Nenya's MLA has no input-projection biases (config.attention_bias=False)")
        # [W_KVD | W_query | W_gate]: one matrix for the three projections of the same input
        self.splits = (L + kr, H * (hd + kr), H * hd)
        self.w_kqg = nn.Parameter(torch.zeros(sum(self.splits), d, dtype=torch.bfloat16), requires_grad=False)
        self.kv_norm_weight = nn.Parameter(torch.ones(L, dtype=torch.float32), requires_grad=False)
        self.kv_b_proj = ColumnParallelLinear(L, H * (hd + hd), bias=False, quant_config=quant_config,
                                              prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(H * hd, d, bias=True, quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend

        self.mla_attn = MLAAttention(
            num_heads=H,
            scale=(hd + kr) ** -0.5,        # SDPA's default for the trained q/k width (112)
            qk_nope_head_dim=hd,
            qk_rope_head_dim=kr,
            v_head_dim=hd,
            q_lora_rank=None,
            kv_lora_rank=L,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_backend=TritonMLABackend,
            prefill_backend_cls=NenyaFlashAttnPrefillBackend,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, d] bf16 (the normalised sublayer input). Returns [T, d] bf16."""
        if FUSED:
            from nenya_vllm.ops.decode_gemv import GEMV_MAX_T

            if x.shape[0] <= GEMV_MAX_T:
                md = self._decode_metadata()
                if md is not None:
                    return self._decode_fused(x, md)
        kqg = F.linear(x, self.w_kqg)
        kv_lora, q, gate = kqg.split(self.splits, dim=-1)
        kv_c, k_pe = kv_lora.split([self.latent_dim, self.shared_key_dim], dim=-1)
        kv_c_normed = rms_norm_fp32w(kv_c, self.kv_norm_weight, self.kv_norm_eps, x.dtype)
        k_pe = k_pe.unsqueeze(1)
        q = q.contiguous().view(-1, self.num_heads, self.head_dim + self.shared_key_dim)
        attn_out = self.mla_attn(q, kv_c_normed, k_pe, output_shape=(x.shape[0], self.num_heads * self.head_dim))
        attn_out = attn_out * torch.sigmoid(gate)
        out, _ = self.o_proj(attn_out)
        return out

    def _decode_metadata(self):
        """This layer's MLA metadata if the batch is decode-only (one token per request) and this code runs eagerly
        or under a FULL CUDA graph, else None. Under PIECEWISE graphs (mixed and prefill batches) this module's code
        is captured once per size with whatever batch the capture used, and replayed for any batch of that size: a
        branch on the metadata here would be baked into the graph (a decode-path graph replayed on a prefill batch
        gives garbage), so those take the regular path, whose metadata-dependent work runs inside vLLM's attention
        op, outside the graph. FULL graphs are captured and replayed for uniform decode batches only."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        if ctx.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        attn_metadata = ctx.attn_metadata
        if not isinstance(attn_metadata, dict):
            return None
        md = attn_metadata.get(self.mla_attn.layer_name)
        if md is None or md.num_prefills or not md.num_decodes or md.num_decode_tokens != md.num_decodes:
            return None
        return md

    def _decode_fused(self, x: torch.Tensor, md) -> torch.Tensor:
        """The decode step in five kernels (ops/mla_decode.py): projections GEMV, prep (kv_norm, cache row,
        absorbed query), vLLM's stage-1 split-KV attention, stage 2 + value up-projection, gated out_proj."""
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
        from vllm.v1.attention.ops.triton_decode_attention import _decode_grouped_att_m_fwd

        from nenya_vllm.ops.decode_gemv import SIGMOID_GATE, decode_linear
        from nenya_vllm.ops.mla_decode import mla_decode_prep, mla_stage2_vup

        T = x.shape[0]
        H, hd, kr, L = self.num_heads, self.head_dim, self.shared_key_dim, self.latent_dim
        layer = self.mla_attn
        kqg = decode_linear(x, self.w_kqg)
        slot_mapping = get_forward_context().slot_mapping[layer.layer_name]
        kv_cache = layer.kv_cache
        q = mla_decode_prep(kqg, self.kv_norm_weight, self.kv_norm_eps, kv_cache, slot_mapping, layer.W_UK_T, L, kr,
                            H, hd)
        B = md.num_decode_tokens
        dec = md.decode
        splits = _compute_num_kv_splits(md.max_seq_len, layer.impl._sm_count)
        attn_logits = torch.empty((B, H, splits, L + 1), dtype=torch.float32, device=x.device)
        k_buffer = kv_cache.unsqueeze(2)
        _decode_grouped_att_m_fwd(q[:B], k_buffer, k_buffer[..., :L], attn_logits, dec.block_table, dec.seq_lens,
                                  splits, layer.impl.scale, kv_cache.shape[1], 0.0, layer._k_scale, layer._k_scale,
                                  is_mla=True)
        attn = torch.empty((T, H * hd), dtype=x.dtype, device=x.device)
        if B < T:
            attn[B:].zero_()
        mla_stage2_vup(attn_logits, dec.seq_lens, layer.W_UV, attn, L, H, hd)
        return decode_linear(attn, self.o_proj.weight, self.o_proj.bias, prologue=SIGMOID_GATE,
                             aux=kqg[:, self.splits[0] + self.splits[1]:])
