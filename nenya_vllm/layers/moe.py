# SPDX-License-Identifier: Apache-2.0
"""Nenya's feed-forward layers: dense SwiGLU (blocks 0-1) and the latent MoE (blocks 2-31).

MoE (release: modeling_nenya.NenyaMoE): 120 routed SwiGLU experts (384 -> 2x384 -> 384) inside a 960 <-> 384
latent (expert_down / expert_up), top-6, plus 2 always-on shared experts fused into one SwiGLU 960 -> 768 -> 960.
Routing: router logits in fp32 from the fp32 normalised input, the quantile-balancer bias added for the choice only,
weights = softmax over the chosen experts' UNBIASED logits.

The matrices that read the same input are stored merged (the Metal engine's layout):
    w_in  = [expert_down | shared fc1 | shared fc2]   [384 + 768 + 768, 960]
    w_out = [expert_up | shared fc3]                   [960, 384 + 768]  (over the concatenated input)
so a layer is: route -> one GEMM in -> routed experts (latent) + shared SiLU-mul into one buffer -> one GEMM out.
Decode-sized batches (NENYA_MOE_SMALL_T, default 8 tokens) run ops/moe_small.py's kernels (4 launches: the w_in
GEMV with the router and top-k, the experts' up and down, the w_out GEMV); larger ones vLLM's grouped fused-MoE
kernels between cuBLAS GEMMs. NENYA_FUSED=0 selects the previous structure (FusedMoE with the
latent transforms and shared experts inside, the router as a separate GEMM), kept for A/B comparisons.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

ROUTE_IMPL = os.environ.get("NENYA_ROUTE_IMPL", "triton")
FUSED = os.environ.get("NENYA_FUSED", "1") == "1"
SMALL_T = int(os.environ.get("NENYA_MOE_SMALL_T", "8"))
# up to this many tokens the w_in GEMV + router + top-k kernel feeds the routed experts with its routing
MID_T = int(os.environ.get("NENYA_MOE_MID_T", "32"))
# between SMALL_T and MID_T tokens: the experts grouped by expert in Triton (ops/moe_small.moe_experts_grouped, fp32
# outputs into w_out) instead of vLLM's grouped CUTLASS kernels given the routing (NENYA_MOE_GROUPED=0)
GROUPED = os.environ.get("NENYA_MOE_GROUPED", "1") == "1"


class NenyaMLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)). Checkpoint names fc1 (gate), fc2 (up), fc3 (down)."""

    def __init__(self, hidden_size: int, intermediate_size: int, quant_config: QuantizationConfig | None,
                 prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False,
                                                       quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from nenya_vllm.ops.decode_gemv import GEMV_MAX_T, SILU_MUL, decode_linear

        if FUSED and x.shape[0] <= GEMV_MAX_T:
            # decode: two GEMVs, the SiLU-mul fused into the second one's input (fp32, no intermediate rounding)
            return decode_linear(decode_linear(x, self.gate_up_proj.weight), self.down_proj.weight,
                                 prologue=SILU_MUL)
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
        return out


class NenyaMoE(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None, prefix: str) -> None:
        super().__init__()
        d = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.latent = latent = config.moe_latent_size
        hidden = config.moe_intermediate_size
        self.shared_hidden = hs = hidden * config.num_shared_experts
        self.fused = FUSED
        f32, bf16 = torch.float32, torch.bfloat16
        # router weight and balancer bias stay fp32 (release `_keep_in_fp32_modules`)
        self.gate_weight = nn.Parameter(torch.zeros(self.num_experts, d, dtype=f32), requires_grad=False)
        self.balancer_bias = nn.Parameter(torch.zeros(self.num_experts, dtype=f32), requires_grad=False)
        if self.fused:
            self.w_in = nn.Parameter(torch.zeros(latent + 2 * hs, d, dtype=bf16), requires_grad=False)
            self.w_out = nn.Parameter(torch.zeros(d, latent + hs, dtype=bf16), requires_grad=False)
            self.experts = FusedMoEFactory(
                num_experts=self.num_experts, top_k=self.top_k, hidden_size=latent, intermediate_size=hidden,
                renormalize=False, quant_config=quant_config, prefix=f"{prefix}.experts",
                custom_routing_function=self.route, activation="silu", router_logits_dtype=f32)
        else:
            self.expert_down = ReplicatedLinear(d, latent, bias=False, quant_config=quant_config,
                                                prefix=f"{prefix}.expert_down")
            self.expert_up = ReplicatedLinear(latent, d, bias=False, quant_config=quant_config,
                                              prefix=f"{prefix}.expert_up")
            self.shared_experts = NenyaMLP(d, hs, quant_config, prefix=f"{prefix}.shared_experts")
            self.experts = FusedMoEFactory(
                num_experts=self.num_experts, top_k=self.top_k, hidden_size=latent, intermediate_size=hidden,
                renormalize=False, quant_config=quant_config, prefix=f"{prefix}.experts",
                custom_routing_function=self.route, activation="silu", shared_experts=self.shared_experts,
                routed_input_transform=self.expert_down, routed_output_transform=self.expert_up,
                router_logits_dtype=f32)
        self._gate_first = None
        self._routing = None            # routing already computed by moe_in_route, handed to the grouped experts

    def route(self, hidden_states: torch.Tensor, gating_output: torch.Tensor, topk: int,
              renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if self._routing is not None and self._routing[1].shape[0] == gating_output.shape[0]:
            return self._routing
        if ROUTE_IMPL == "torch":
            logits = gating_output.float()
            biased = logits + self.balancer_bias
            # the release's selection (top k+1, keep k): identical up to ties at the boundary
            ids = torch.topk(biased, topk + 1, dim=-1).indices[..., :topk]
            weights = torch.softmax(torch.gather(logits, -1, ids), dim=-1)
            return weights, ids.to(torch.int32)
        from nenya_vllm.ops.small_kernels import moe_route

        return moe_route(gating_output.float(), self.balancer_bias, topk)

    def _expert_weights(self):
        """(w13, w2, gate_first) of the grouped layer, after vLLM's post-load processing (the FlashInfer CUTLASS
        backend stores w13 as [up; gate])."""
        mod = self.experts
        w13 = getattr(mod, "w13_weight", None)
        if w13 is None:
            mod = mod.routed_experts
            w13 = mod.w13_weight
        if self._gate_first is None:
            backend = getattr(getattr(mod, "quant_method", None), "unquantized_backend", None)
            self._gate_first = "FLASHINFER_CUTLASS" not in str(backend)
        return w13, mod.w2_weight, self._gate_first

    def forward(self, x_f32: torch.Tensor, x_bf16: torch.Tensor) -> torch.Tensor:
        """x_f32: the fp32 normalised input (router); x_bf16: the same in bf16 (experts)."""
        if not self.fused:
            router_logits = F.linear(x_f32, self.gate_weight)
            return self.experts(hidden_states=x_bf16, router_logits=router_logits)
        from nenya_vllm.ops.decode_gemv import decode_linear
        from nenya_vllm.ops.moe_small import moe_experts_small, moe_in_route, shared_silu_mul

        T = x_bf16.shape[0]
        lat, hs = self.latent, self.shared_hidden
        buf = torch.empty((T, lat + hs), dtype=x_bf16.dtype, device=x_bf16.device)
        if T <= MID_T:
            # [w_in GEMV + fp32 router + top-k] in one kernel, then the routed experts: per (token, expert) pair for
            # a few tokens, grouped by expert above (ops/moe_small.py), or vLLM's grouped kernels given that
            # routing; w_out as one GEMV / tensor-core product
            lat_in, sh, logits, weights, ids = moe_in_route(x_f32, self.gate_weight, self.balancer_bias, self.w_in,
                                                            self.top_k, lat)
            if T <= SMALL_T:
                w13, w2, gate_first = self._expert_weights()
                moe_experts_small(lat_in, sh, ids, weights, w13, w2, gate_first, buf)
            elif GROUPED:
                from nenya_vllm.ops.decode_gemv import F32
                from nenya_vllm.ops.moe_small import moe_experts_grouped

                w13, w2, gate_first = self._expert_weights()
                xs = moe_experts_grouped(lat_in, sh, ids, weights, w13, w2, gate_first)
                return decode_linear(xs, self.w_out, prologue=F32)
            else:
                self._routing = (weights, ids)
                try:
                    buf[:, :lat] = self.experts(hidden_states=lat_in, router_logits=logits)
                finally:
                    self._routing = None
                shared_silu_mul(sh, lat, hs, buf)
            return decode_linear(buf, self.w_out)
        mid = F.linear(x_bf16, self.w_in)                         # [T, lat + 2 hs]
        router_logits = F.linear(x_f32, self.gate_weight)
        buf[:, :lat] = self.experts(hidden_states=mid[:, :lat].contiguous(), router_logits=router_logits)
        shared_silu_mul(mid[:, lat:], lat, hs, buf)
        return F.linear(buf, self.w_out)
