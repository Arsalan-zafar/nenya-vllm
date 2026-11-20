# SPDX-License-Identifier: Apache-2.0
"""FlashAttention MLA prefill for Nenya's head shape.

vLLM's FlashAttention MLA prefill backend accepts only the DeepSeek/GLM (qk_nope, rope, v) shapes. Nenya's MLA is
(80, 32, 80): per-head key/value width 80 plus a 32-wide key part shared by all heads (no RoPE). This subclass
adds that shape and always zero-pads V to the QK width (112): FA2 needs equal widths, FA3's different-width path
only covers QK 128/192, and the padding columns are sliced off by the caller, so the result is exact.
"""

from vllm.v1.attention.backends.mla.prefill.base import MLADimensions
from vllm.v1.attention.backends.mla.prefill.flash_attn import FlashAttnPrefillBackend

NENYA_MLA_DIMENSIONS = MLADimensions(qk_nope_head_dim=80, qk_rope_head_dim=32, v_head_dim=80)


class NenyaFlashAttnPrefillBackend(FlashAttnPrefillBackend):
    @classmethod
    def supports_mla_dimensions(cls, mla_dimensions: MLADimensions) -> bool:
        return mla_dimensions == NENYA_MLA_DIMENSIONS or super().supports_mla_dimensions(mla_dimensions)

    def __init__(self, num_heads, scale, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                 vllm_config) -> None:
        super().__init__(num_heads=num_heads, scale=scale, kv_lora_rank=kv_lora_rank,
                         qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim,
                         v_head_dim=v_head_dim, vllm_config=vllm_config)
        dims = MLADimensions(qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim,
                             v_head_dim=v_head_dim)
        if dims == NENYA_MLA_DIMENSIONS:
            self.requires_v_padding = True
