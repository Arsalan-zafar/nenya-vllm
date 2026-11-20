# SPDX-License-Identifier: Apache-2.0
"""Nenya's Gated DeltaNet-2 token mixer as a vLLM stateful (mamba-type) layer.

The recurrence (fla 0.5.2, the version Nenya was trained with):
    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T,   o_t = S_t^T (q_t / sqrt(K))
with L2-normalised q and k, a channel-wise log-decay g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias), and
channel-wise erase (b, key axis) and write (w, value axis) gates. Prefill runs fla's chunked kernel over the
batch's variable-length sequences with their cached states; decoding runs fla's recurrent kernel, which reads and
updates vLLM's state pool in place through the state indices.

Per-request state (vLLM mamba cache): the last 3 inputs of the q/k/v short convolutions (bf16) and the recurrent
state [12, 128, 128] in fp32 (training's precision; --mamba-ssm-cache-dtype bfloat16 halves it).

Like vLLM's OLMo-Hybrid GDN layer, the whole token mixer (projections, convolution, recurrence, output norm and
projection) is one custom op and a break point for breakable CUDA graphs -- except under PIECEWISE graphs (mixed
and prefill batches), where only the convolutions + recurrence break out (nenya_gdn2_core) and the projections,
output norm and projection are captured with the rest of the graph segment.
"""

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

# decode-only batches: in_proj -> one fused step kernel (ops/gdn2_decode.py) -> o_proj with the gated norm fused in
FUSED = os.environ.get("NENYA_FUSED_GDN", os.environ.get("NENYA_FUSED", "1")) == "1"
# batches with prefill tokens: up to this many tokens the varlen convolution + sequential recurrence (2 kernels),
# above fla's chunked kernels (fast on the GPU, ~0.8 ms of host time per layer per call)
SHORT_PREFILL_T = int(os.environ.get("NENYA_SHORT_PREFILL_T", "1024"))
# long prefills: the chunked kernels' inputs from one prep kernel (ops/gdn2_decode.gdn2_chunk_prep)
CHUNK_PREP = os.environ.get("NENYA_GDN_CHUNK_PREP", "1") == "1"
# decode-only batches up to this many tokens take _decode_fused; larger ones the general path (cuBLAS f_b / g_b, the
# gate kernel, fla's recurrent kernel), measured 3-5% faster from 64 sequences up on an RTX 5090: the fused recurrent
# kernel recomputes f_b and g_b in each of its 4 value-block programs per head (NENYA_GDN_DECODE_T overrides)
DECODE_FUSED_MAX_T = int(os.environ.get("NENYA_GDN_DECODE_T", "32"))
# under PIECEWISE CUDA graphs (mixed and prefill batches) only the convolutions + recurrence are an eager break; the
# projections, gated norm and o_proj are captured (NENYA_GDN_SPLIT=0: the whole layer is one eager op)
SPLIT_PIECEWISE = os.environ.get("NENYA_GDN_SPLIT", "1") == "1"


def gdn2_state_shape(num_heads: int, head_dim: int, conv_kernel: int) -> tuple[tuple[int, int], tuple[int, int, int]]:
    conv_dim = 3 * num_heads * head_dim
    return (MambaStateShapeCalculator._orient_conv_shape(conv_dim, conv_kernel - 1),
            (num_heads, head_dim, head_dim))


def gdn2_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    # conv state in the model dtype, recurrent state fp32 unless --mamba-ssm-cache-dtype says otherwise
    return MambaStateDtypeCalculator.kda_state_dtype(vllm_config.model_config.dtype,
                                                     vllm_config.cache_config.mamba_cache_dtype,
                                                     vllm_config.cache_config.mamba_ssm_cache_dtype)


class NenyaGatedDeltaNet2(nn.Module, MambaBase):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("Nenya's GDN-2 layer does not support speculative decoding yet")
        d = config.hidden_size
        self.num_heads = H = config.num_attention_heads
        self.head_dim = K = config.linear_head_dim
        self.key_dim = self.value_dim = kd = H * K
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.norm_eps = config.rms_norm_eps

        # q, k, v, b (erase), w (write), f_a (decay, low rank), g_a (output gate, low rank): one GEMM
        self.in_proj = MergedColumnParallelLinear(d, [kd, kd, kd, kd, kd, K, K], bias=False,
                                                  quant_config=quant_config, prefix=f"{prefix}.in_proj")
        self.f_b_proj = ReplicatedLinear(K, kd, bias=False, quant_config=quant_config, prefix=f"{prefix}.f_b_proj")
        self.g_b_proj = ReplicatedLinear(K, kd, bias=True, quant_config=quant_config, prefix=f"{prefix}.g_b_proj")
        self.o_proj = RowParallelLinear(kd, d, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        f32 = torch.float32
        # q/k/v short-convolution taps, fused [q; k; v] (fp32 as in the release)
        self.conv_weight = nn.Parameter(torch.zeros(3 * kd, self.conv_kernel_size, dtype=f32), requires_grad=False)
        self.A_log = nn.Parameter(torch.zeros(H, dtype=f32), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.zeros(kd, dtype=f32), requires_grad=False)
        self.o_norm_weight = nn.Parameter(torch.ones(K, dtype=f32), requires_grad=False)
        self._state_dtype = gdn2_state_dtype(vllm_config)
        self._max_num_seqs = vllm_config.scheduler_config.max_num_seqs

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    # ---- MambaBase ----
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        return gdn2_state_shape(self.num_heads, self.head_dim, self.conv_kernel_size)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return self._state_dtype

    # ---- forward ----
    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        if FUSED and SPLIT_PIECEWISE:
            from vllm.config import CUDAGraphMode

            if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                self._forward_piecewise(hidden_states, output)
                return
        torch.ops.vllm.nenya_gdn2_forward(hidden_states, output, self.prefix)

    def _forward_piecewise(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        """Mixed and prefill batches under PIECEWISE CUDA graphs: the projections, the gated norm and o_proj run in
        the captured graph segments; only the metadata-dependent convolutions + recurrence break out to an eager op
        (nenya_gdn2_core). The same ops as _forward_impl's general path, minus ~6 eager launches per layer and call
        (the host time that bounded short prefills)."""
        from fla.modules.fused_norm_gate import rms_norm_gated

        T = hidden_states.shape[0]
        H, K, kd = self.num_heads, self.head_dim, self.key_dim
        proj, _ = self.in_proj(hidden_states)          # [T, q k v b w f_a g_a]
        f, _ = self.f_b_proj(proj[:, 5 * kd:5 * kd + K])
        gate, _ = self.g_b_proj(proj[:, 5 * kd + K:])
        core = torch.empty((T, H, K), dtype=hidden_states.dtype, device=hidden_states.device)
        torch.ops.vllm.nenya_gdn2_core(proj, f, core, self.prefix)
        on = rms_norm_gated(core.view(T * H, K), gate.view(T * H, K), self.o_norm_weight, None,
                            activation="sigmoid", eps=self.norm_eps)
        torch.mm(on.view(T, H * K), self.o_proj.weight.t(), out=output[:T])

    def _forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        from fla.modules.fused_norm_gate import rms_norm_gated

        T = hidden_states.shape[0]
        H, K, kd = self.num_heads, self.head_dim, self.key_dim
        attn_metadata = get_forward_context().attn_metadata
        md = attn_metadata.get(self.prefix) if isinstance(attn_metadata, dict) else None
        if FUSED and md is not None and md.num_prefills == 0 and md.num_decodes > 0 and T <= DECODE_FUSED_MAX_T:
            self._decode_fused(hidden_states, output, md)
            return
        proj, _ = self.in_proj(hidden_states)          # [T, q k v b w f_a g_a]
        f, _ = self.f_b_proj(proj[:, 5 * kd:5 * kd + K])
        gate, _ = self.g_b_proj(proj[:, 5 * kd + K:])

        core = torch.empty((T, H, K), dtype=hidden_states.dtype, device=hidden_states.device)
        self._core_dispatch(proj, f, core)
        # fla's FusedRMSNormGated(activation='sigmoid'), the kernel training ran: rmsnorm(o) * w * sigmoid(g)
        on = rms_norm_gated(core.view(T * H, K), gate.view(T * H, K), self.o_norm_weight, None,
                            activation="sigmoid", eps=self.norm_eps)
        torch.mm(on.view(T, H * K), self.o_proj.weight.t(), out=output[:T])

    def _core_dispatch(self, proj: torch.Tensor, f: torch.Tensor, core: torch.Tensor) -> None:
        """The metadata-dependent part: short convolutions + recurrence for the batch's real rows into core
        ([T, H, K]; padding rows zero), reading and updating the state pools."""
        T = proj.shape[0]
        H, K = self.num_heads, self.head_dim
        attn_metadata = get_forward_context().attn_metadata
        md = attn_metadata.get(self.prefix) if isinstance(attn_metadata, dict) else None
        n = 0
        if md is not None:
            assert isinstance(md, GDNAttentionMetadata)
            if md.num_prefills > 0:
                # real tokens only: the chunked kernel leaves rows past the last sequence unwritten
                n = md.num_prefill_tokens + md.num_decode_tokens
            elif md.num_decodes > 0:
                # decode-only batches may be padded for CUDA graphs; padded rows map to the null state slot
                n = min(md.num_actual_tokens, md.non_spec_state_indices_tensor.shape[0])
            if n and FUSED and md.num_prefills > 0 and n <= SHORT_PREFILL_T and md.spec_sequence_masks is None:
                from nenya_vllm.ops.gdn2_decode import gdn2_prefill_short

                conv_state, ssm_state = self.kv_cache
                if not is_conv_state_dim_first():
                    conv_state = conv_state.transpose(-1, -2)
                qsl = md.non_spec_query_start_loc
                gdn2_prefill_short(proj, f, conv_state, self.conv_weight, ssm_state, qsl,
                                   md.non_spec_state_indices_tensor, md.has_initial_state, self.A_log, self.dt_bias,
                                   qsl.shape[0] - 1, H, K, core.view(T, H * K))
            elif n:
                self._core(proj[:n], f[:n], md, core)
        elif T and not torch.cuda.is_current_stream_capturing():
            self._profile_core(proj, f)
        if n < T:
            core[n:].zero_()

    def _decode_fused(self, x: torch.Tensor, output: torch.Tensor, md: GDNAttentionMetadata) -> None:
        """Decode-only batch: in_proj with the short convolutions in its epilogue, the recurrence with f_b, g_b and
        the gates in its prologue (ops/gdn2_decode.py), o_proj with fla's gated RMSNorm computed in its input
        prologue -- 3 kernels instead of 9. Above GEMV_MAX_T tokens in_proj and o_proj stay on cuBLAS (with vLLM's
        convolution kernel and fla's gated norm around them)."""
        from fla.modules.fused_norm_gate import rms_norm_gated

        from nenya_vllm.ops.decode_gemv import GATED_RMSNORM, GEMV_MAX_T, decode_linear
        from nenya_vllm.ops.gdn2_decode import gdn2_in_proj, gdn2_recurrent_step

        if md.spec_sequence_masks is not None:
            raise NotImplementedError("speculative decoding is not supported by Nenya's GDN-2 layer")
        T = x.shape[0]
        H, K = self.num_heads, self.head_dim
        idx = md.non_spec_state_indices_tensor
        # decode batches may be padded for CUDA graphs; padded rows map to the null state slot (skipped)
        n = min(md.num_actual_tokens, idx.shape[0], T)
        conv_state, ssm_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        small = T <= GEMV_MAX_T
        if small:
            proj = gdn2_in_proj(x, self.in_proj.weight, conv_state, self.conv_weight, idx, n)
        else:
            proj, _ = self.in_proj(x)
            # SiLU(conv) written over the q/k/v columns (out=x), the layout the recurrent kernel reads
            causal_conv1d_update(proj[:n, :3 * self.key_dim], conv_state, self.conv_weight, None, "silu",
                                 conv_state_indices=idx[:n], validate_data=False)
        core, gate = gdn2_recurrent_step(proj, self.f_b_proj.weight, self.g_b_proj.weight, self.g_b_proj.bias,
                                         self.A_log, self.dt_bias, ssm_state, idx, n, H, K)
        if small:
            decode_linear(core, self.o_proj.weight, prologue=GATED_RMSNORM, aux=gate, norm_weight=self.o_norm_weight,
                          eps=self.norm_eps, group=K, out=output[:T])
        else:
            on = rms_norm_gated(core.view(T * H, K), gate.view(T * H, K), self.o_norm_weight, None,
                                activation="sigmoid", eps=self.norm_eps)
            torch.mm(on.view(T, H * K), self.o_proj.weight.t(), out=output[:T])

    def _core(self, proj: torch.Tensor, f: torch.Tensor, md: GDNAttentionMetadata, core: torch.Tensor) -> None:
        """Short convolutions + GDN-2 recurrence for the n real rows of the batch, reading and updating the state
        pool; writes o into core[:n]."""
        from fla.ops.gdn2 import chunk_gdn2
        from fla.ops.gdn2.fused_recurrent import fused_recurrent_gdn2_fwd

        from nenya_vllm.ops.small_kernels import gdn2_gates

        if md.spec_sequence_masks is not None:
            raise NotImplementedError("speculative decoding is not supported by Nenya's GDN-2 layer")
        n = proj.shape[0]
        H, K, kd = self.num_heads, self.head_dim, self.key_dim
        conv_state, ssm_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        mixed_qkv = proj[:, :3 * kd]
        # erase / write gates: sigmoid in bf16 (the release's autocast dtype), one kernel for both (the fused chunked
        # path computes them in its prep kernel)
        if not (FUSED and CHUNK_PREP and md.num_prefills > 0):
            b, w = gdn2_gates(proj, 3 * kd, 4 * kd, kd)
        # the decay gate -exp(A_log) * softplus(f + dt_bias) is computed in fp32 inside fla's kernels
        gate_kw = dict(use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias)

        if md.num_prefills > 0:
            # every non-speculative request (decodes too, as 1-token sequences) through the chunked path
            idx = md.non_spec_state_indices_tensor
            init = torch.where(md.has_initial_state[:, None, None, None], ssm_state[idx].float(), 0.0)
            if FUSED and CHUNK_PREP:
                # the convolutions, L2 norms and gates in one kernel, in fla's layout (gdn2_chunk_prep)
                from nenya_vllm.ops.gdn2_decode import gdn2_chunk_prep

                qsl = md.non_spec_query_start_loc
                q, k, v, b, w = gdn2_chunk_prep(proj, conv_state, self.conv_weight, qsl, idx, md.has_initial_state,
                                                qsl.shape[0] - 1, H, K)
                o, final = chunk_gdn2(q.view(1, n, H, K), k.view(1, n, H, K), v.view(1, n, H, K),
                                      f.view(1, n, H, K), b.view(1, n, H, K), w.view(1, n, H, K), scale=K ** -0.5,
                                      initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=False,
                                      cu_seqlens=qsl, **gate_kw)
            else:
                x = causal_conv1d_fn(mixed_qkv.transpose(0, 1), self.conv_weight, None, conv_states=conv_state,
                                     query_start_loc=md.non_spec_query_start_loc,
                                     cache_indices=md.non_spec_state_indices_tensor,
                                     has_initial_state=md.has_initial_state, activation="silu", metadata=md)
                qkv = x.reshape(3, H, K, n).permute(0, 3, 1, 2).contiguous()     # [3, n, H, K]
                o, final = chunk_gdn2(qkv[0].unsqueeze(0), qkv[1].unsqueeze(0), qkv[2].unsqueeze(0),
                                      f.view(1, n, H, K), b.view(1, n, H, K), w.view(1, n, H, K), scale=K ** -0.5,
                                      initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=True,
                                      cu_seqlens=md.non_spec_query_start_loc, **gate_kw)
            ssm_state[idx] = final.to(ssm_state.dtype)
            core[:n].copy_(o[0])
        else:
            idx = md.non_spec_state_indices_tensor[:n]
            x = causal_conv1d_update(mixed_qkv, conv_state, self.conv_weight, None, "silu",
                                     conv_state_indices=idx, validate_data=False)
            qkv = x.reshape(n, 3, H, K).permute(1, 0, 2, 3).contiguous()     # [3, n, H, K]
            fused_recurrent_gdn2_fwd(qkv[0].unsqueeze(1), qkv[1].unsqueeze(1), qkv[2].unsqueeze(1),
                                     f.view(n, 1, H, K), b.view(n, 1, H, K), w.view(n, 1, H, K),
                                     initial_state=ssm_state, scale=K ** -0.5, inplace_final_state=True,
                                     ssm_state_indices=idx, use_qk_l2norm_in_kernel=True,
                                     out=core[:n].view(n, 1, H, K), **gate_kw)


    def _profile_core(self, proj: torch.Tensor, f: torch.Tensor) -> None:
        """Forward passes without metadata for this layer (vLLM's memory profiling run and its kernel warmup) skip
        the recurrence, so its transient buffers would be missing from the peak memory vLLM measures to size the
        KV cache: the per-sequence fp32 initial and final states (0.8 MB each per sequence) and fla's per-chunk
        states (0.8 MB per 64-token chunk). Measured on an RTX 5090 (max_num_seqs 256): the profiled peak goes
        from 1.35 to 1.91 GiB with this, and without it a 256-request batch ran out of memory at
        --gpu-memory-utilization 0.95. Run the same kernels here on the profiling batch, split into as many sequences
        as vLLM's dummy run uses, with zero inputs and throwaway states; nothing is written to the state pool and the
        output is unused. This also autotunes fla's kernels at startup instead of on the first request."""
        from fla.ops.gdn2 import chunk_gdn2

        from nenya_vllm.ops.small_kernels import gdn2_gates

        T = proj.shape[0]
        H, K, kd = self.num_heads, self.head_dim, self.key_dim
        N = min(T, self._max_num_seqs)
        # vLLM's _dummy_run spreads the T tokens evenly over N requests
        lens = torch.full((N,), T // N, dtype=torch.int32)
        if T % N:
            lens[N - T % N:] += 1
        cu = torch.zeros(N + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(lens, 0)
        b, w = gdn2_gates(proj, 3 * kd, 4 * kd, kd)
        x = torch.zeros((3 * kd, T), dtype=proj.dtype, device=proj.device)     # the short convolutions' output
        qkv = x.reshape(3, H, K, T).permute(0, 3, 1, 2).contiguous()
        init = torch.zeros((N, H, K, K), dtype=torch.float32, device=proj.device)
        chunk_gdn2(qkv[0].unsqueeze(0), qkv[1].unsqueeze(0), qkv[2].unsqueeze(0), f.view(1, T, H, K),
                   b.view(1, T, H, K), w.view(1, T, H, K), scale=K ** -0.5, initial_state=init,
                   output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu.to(proj.device),
                   use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias)


@eager_break_during_capture
def nenya_gdn2_forward(hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_impl(hidden_states, output)


def _nenya_gdn2_forward_fake(hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    return None


@eager_break_during_capture
def nenya_gdn2_core(proj: torch.Tensor, f: torch.Tensor, core: torch.Tensor, layer_name: str) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._core_dispatch(proj, f, core)


def _nenya_gdn2_core_fake(proj: torch.Tensor, f: torch.Tensor, core: torch.Tensor, layer_name: str) -> None:
    return None


direct_register_custom_op(
    op_name="nenya_gdn2_forward",
    op_func=nenya_gdn2_forward,
    mutates_args=["output"],
    fake_impl=_nenya_gdn2_forward_fake,
)


direct_register_custom_op(
    op_name="nenya_gdn2_core",
    op_func=nenya_gdn2_core,
    mutates_args=["proj", "core"],
    fake_impl=_nenya_gdn2_core_fake,
)
