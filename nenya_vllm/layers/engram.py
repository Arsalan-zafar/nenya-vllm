# SPDX-License-Identifier: Apache-2.0
"""Nenya's Engram (hashed n-gram memory) as a vLLM stateful (short-conv type) layer.

Release module: modeling_nenya.NenyaEngram + NenyaModel.engram_rows. Per token t:
    rows   = hash of the compressed ids of the 2- and 3-grams ending at t (16 tables of 60-wide rows)
    e      = concat of the 16 looked-up rows                                 [960]
    k_s    = W_K[s] e + bias, v = W_V e + bias  (s = the 4 residual streams)
    gate_s = sigmoid(sign(z) sqrt(max(|z|, 1e-6))),  z = <RMSNorm(x_s), RMSNorm(k_s)> / sqrt(960)
    c_s    = RMSNorm(gate_s * v)
    y_s    = SiLU(dilated causal depthwise conv of c_s, taps t, t-3, t-6, t-9)
    x_s   += bf16(y_s + gate_s * v)
The gate, norms and convolution run in fp32 like the release under autocast (its RMSNorms have fp32 weights and
return fp32); the conv history is kept in fp32 too.

Per-request state: the conv inputs of the last 9 positions over the 4 x 960 stream channels (fp32). The two token
ids before a request's chunk, which the hash needs, are not state: the model runner reads them from its
device-resident token history every step (model_state.NenyaModelState), as vLLM does for Qwen4Exp's n-gram layer.
"""

import math
import os
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator, is_conv_state_dim_first
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)

from nenya_vllm.layers.mhc import rms_norm_fp32

# torch.nn.RMSNorm(eps=None) on fp32 data: fp32 machine epsilon
_ENGRAM_NORM_EPS = torch.finfo(torch.float32).eps
# decode-only batches: rows, gathered-input GEMV, one combine kernel (ops/engram_decode.py)
FUSED = os.environ.get("NENYA_FUSED_ENGRAM", os.environ.get("NENYA_FUSED", "1")) == "1"


def engram_state_shape(config) -> tuple[tuple[int, int]]:
    dilation = max(config.engram_ngram_orders)
    state_len = (config.engram_conv_kernel_dim - 1) * dilation
    channels = config.mhc_num_streams * config.hidden_size
    return MambaStateShapeCalculator.short_conv_state_shape(tp_world_size=1, intermediate_size=channels,
                                                            conv_kernel=state_len + 1)


ENGRAM_STATE_DTYPE = (torch.float32,)


class NenyaEngram(nn.Module, MambaBase):
    def __init__(self, config, vllm_config: VllmConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.config = config
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("Nenya's Engram layer does not support speculative decoding yet")
        self.n = config.mhc_num_streams
        self.d = config.hidden_size
        self.channels = self.n * self.d
        self.heads = config.engram_num_heads
        self.orders = list(config.engram_ngram_orders)
        self.ctx_len = max(self.orders) - 1
        self.dilation = max(self.orders)
        self.kernel = config.engram_conv_kernel_dim
        self.state_len = (self.kernel - 1) * self.dilation
        self.eot_token_id = config.eot_token_id
        sizes = config.engram_table_sizes[str(layer_id)] if config.engram_table_sizes else None
        if sizes is None:
            raise ValueError("Nenya's config must carry engram_table_sizes (the release config does)")
        num_rows = sum(int(s) for row in sizes for s in row)
        num_tables = len(self.orders) * self.heads
        self.head_dim = config.engram_dim // num_tables

        # hashing buffers, shipped in the checkpoint (int64, never cast)
        self.register_buffer("table_sizes", torch.zeros(len(self.orders), self.heads, dtype=torch.long))
        self.register_buffer("table_offsets", torch.zeros(num_tables, dtype=torch.long))
        self.register_buffer("multipliers", torch.zeros(max(self.orders), dtype=torch.long))
        self.tables = nn.Parameter(torch.zeros(num_rows, self.head_dim, dtype=torch.bfloat16), requires_grad=False)
        # W_K (4 x [960 -> 960]) and W_V ([960 -> 960]) with biases, one GEMM
        self.kv_proj = MergedColumnParallelLinear(config.engram_dim, [self.d] * self.n + [self.d], bias=True,
                                                  quant_config=None, prefix=f"{prefix}.kv_proj")
        f32 = torch.float32
        self.norm_h = nn.Parameter(torch.ones(self.d, dtype=f32), requires_grad=False)
        self.norm_k = nn.Parameter(torch.ones(self.d, dtype=f32), requires_grad=False)
        self.norm_v = nn.Parameter(torch.ones(self.d, dtype=f32), requires_grad=False)
        # the 4 per-stream depthwise convs as one [4*960, kernel] fp32 tap matrix (stream-major channels)
        self.conv_weight = nn.Parameter(torch.zeros(self.channels, self.kernel, dtype=f32), requires_grad=False)

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),)
        self.token_map: torch.Tensor | None = None      # set by the model (shared engram_lookup buffer)
        self.pad_id: int = int(config.engram_pad_id) if config.engram_pad_id is not None else -1

    # ---- MambaBase ----
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self):
        return PleShortConvAttentionBackend

    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        return engram_state_shape(self.config)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return ENGRAM_STATE_DTYPE

    # ---- forward ----
    def forward(self, x_res: torch.Tensor, input_ids: torch.Tensor, query_start_loc: torch.Tensor,
                ngram_context: torch.Tensor, output: torch.Tensor, layer_out: torch.Tensor | None = None,
                coef: torch.Tensor | None = None) -> None:
        """x_res [T, n*d]: the residual streams; with layer_out and coef, the streams before the previous sublayer's
        mHC post, which is applied first (inside the decode kernel)."""
        torch.ops.vllm.nenya_engram_forward(x_res, input_ids, query_start_loc, ngram_context, output, self.prefix,
                                            layer_out, coef)

    def rows(self, input_ids, query_start_loc, ngram_context) -> torch.Tensor:
        from nenya_vllm.ops.ngram_rows import nenya_ngram_rows

        return nenya_ngram_rows(input_ids, query_start_loc, ngram_context, self.token_map, self.multipliers,
                                self.table_sizes.reshape(-1), self.table_offsets, self.eot_token_id, self.pad_id,
                                self.heads)

    def _forward_impl(self, x_res, input_ids, query_start_loc, ngram_context, output, layer_out=None,
                      coef=None) -> None:
        T = x_res.shape[0]
        n, d = self.n, self.d
        attn_metadata = get_forward_context().attn_metadata
        md = attn_metadata.get(self.prefix) if isinstance(attn_metadata, dict) else None
        if FUSED and md is not None and md.num_prefills == 0 and md.num_decodes > 0:
            from nenya_vllm.ops.decode_gemv import GEMV_MAX_T

            if T <= GEMV_MAX_T and md.spec_sequence_masks is None:
                self._decode_fused(x_res, input_ids, query_start_loc, ngram_context, output, layer_out, coef, md)
                return
        if FUSED:
            # gathered-row GEMV (or the lookup + cuBLAS for many tokens), one kernel for the post + gate + norms, vLLM's
            # dilated convolution, one kernel for the residual add
            from nenya_vllm.ops.decode_gemv import GATHER, GEMV_MAX_T, decode_linear
            from nenya_vllm.ops.engram_decode import engram_combine, engram_gate

            rows = self.rows(input_ids[:T], query_start_loc, ngram_context)      # [T, 16]
            if T <= GEMV_MAX_T:
                kv = decode_linear(rows, self.kv_proj.weight, self.kv_proj.bias, prologue=GATHER, aux=self.tables,
                                   group=self.head_dim)
            else:
                kv, _ = self.kv_proj(F.embedding(rows, self.tables).reshape(T, -1))
            c, gate, xs = engram_gate(x_res.view(T, n, d), layer_out, coef, kv, self.norm_h, self.norm_k,
                                      self.norm_v, _ENGRAM_NORM_EPS)
            y = torch.zeros((T, n * d), dtype=torch.float32, device=x_res.device)
            if md is not None:
                self._conv(c, y, md)
            engram_combine(xs, y, gate, kv, output[:T].view(T, n, d))
            return
        if layer_out is not None:
            from nenya_vllm.ops.mhc_kernels import mhc_post

            x_res = mhc_post(x_res.view(T, n, d), layer_out, coef).reshape(T, n * d)
        rows = self.rows(input_ids[:T], query_start_loc, ngram_context)          # [T, 16]
        e = F.embedding(rows, self.tables).reshape(T, -1)                        # [T, 960] bf16
        kv, _ = self.kv_proj(e)                                                  # [T, 4*960 + 960] bf16
        key = kv[:, :n * d].view(T, n, d)
        value = kv[:, n * d:]
        xr = x_res.view(T, n, d)

        z = (rms_norm_fp32(xr, self.norm_h, _ENGRAM_NORM_EPS)
             * rms_norm_fp32(key, self.norm_k, _ENGRAM_NORM_EPS)).sum(-1, keepdim=True) / math.sqrt(d)
        z = z.abs().clamp_min(1e-6).sqrt() * z.sign()
        v_gated = torch.sigmoid(z) * value.float().unsqueeze(1)                  # [T, n, d] fp32
        c = rms_norm_fp32(v_gated, self.norm_v, _ENGRAM_NORM_EPS).reshape(T, n * d).contiguous()

        y = torch.zeros((T, n * d), dtype=torch.float32, device=x_res.device)
        if md is not None:
            self._conv(c, y, md)
        mem = (y.view(T, n, d) + v_gated).to(x_res.dtype)
        output[:T].copy_((xr + mem).reshape(T, n * d))

    def _decode_fused(self, x_res, input_ids, query_start_loc, ngram_context, output, layer_out, coef, md) -> None:
        """Decode-only batch: hash rows, the [W_K | W_V] GEMV with the table rows gathered in its prologue, and one
        kernel for the previous post + gate + norms + dilated conv + residual add (ops/engram_decode.py)."""
        from nenya_vllm.ops.decode_gemv import GATHER, decode_linear
        from nenya_vllm.ops.engram_decode import engram_decode_combine

        T = x_res.shape[0]
        n, d = self.n, self.d
        rows = self.rows(input_ids[:T], query_start_loc, ngram_context)          # [T, 16] global table rows
        kv = decode_linear(rows, self.kv_proj.weight, self.kv_proj.bias, prologue=GATHER, aux=self.tables,
                           group=self.head_dim)
        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_state = conv_state[..., -self.state_len:]
        idx = md.state_indices_tensor
        engram_decode_combine(x_res.view(T, n, d), layer_out, coef, kv, self.norm_h, self.norm_k, self.norm_v,
                              self.conv_weight, conv_state, idx, md.has_initial_states_d, min(T, idx.shape[0]),
                              _ENGRAM_NORM_EPS, self.dilation, out=output[:T].view(T, n, d))

    def _conv(self, c: torch.Tensor, y: torch.Tensor, md: PleShortConvAttentionMetadata) -> None:
        """SiLU(dilated causal conv) of c into y, reading and updating the per-request conv history."""
        from nenya_vllm.ops.dilated_conv import ple_conv

        if md.spec_sequence_masks is not None:
            raise NotImplementedError("speculative decoding is not supported by Nenya's Engram layer")
        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)          # canonical [slot, channel, window]
        conv_state = conv_state[..., -self.state_len:]
        state_indices = md.state_indices_tensor
        n_d, n_p = md.num_decodes, md.num_prefills
        d_tok, p_tok = md.num_decode_tokens, md.num_prefill_tokens
        if n_p > 0:
            if n_d > 0:
                ple_conv(inputs=c[:d_tok], residual=y[:d_tok], conv_state=conv_state, conv_weights=self.conv_weight,
                         state_indices=state_indices[:n_d], outer_residual=None, mode="decode",
                         dilation=self.dilation, has_initial_states=md.has_initial_states_d)
            ple_conv(inputs=c[d_tok:d_tok + p_tok], residual=y[d_tok:d_tok + p_tok], conv_state=conv_state,
                     conv_weights=self.conv_weight, state_indices=state_indices[n_d:n_d + n_p], outer_residual=None,
                     mode="prefill", dilation=self.dilation, query_start_loc=md.query_start_loc_p,
                     has_initial_states=md.has_initial_states_p)
        elif n_d > 0:
            rows = min(c.shape[0], state_indices.shape[0])
            ple_conv(inputs=c[:rows], residual=y[:rows], conv_state=conv_state, conv_weights=self.conv_weight,
                     state_indices=state_indices[:rows], outer_residual=None, mode="decode",
                     dilation=self.dilation, has_initial_states=md.has_initial_states_d)


@eager_break_during_capture
def nenya_engram_forward(x_res: torch.Tensor, input_ids: torch.Tensor, query_start_loc: torch.Tensor,
                         ngram_context: torch.Tensor, output: torch.Tensor, layer_name: str,
                         layer_out: torch.Tensor | None = None, coef: torch.Tensor | None = None) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_impl(x_res, input_ids, query_start_loc, ngram_context, output, layer_out, coef)


def _nenya_engram_forward_fake(x_res, input_ids, query_start_loc, ngram_context, output, layer_name,
                               layer_out=None, coef=None) -> None:
    return None


direct_register_custom_op(
    op_name="nenya_engram_forward",
    op_func=nenya_engram_forward,
    mutates_args=["output"],
    fake_impl=_nenya_engram_forward_fake,
)
