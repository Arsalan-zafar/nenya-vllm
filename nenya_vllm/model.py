# SPDX-License-Identifier: Apache-2.0
"""Nenya-2.5B-A0.5B for vLLM.

A port of release/HF/nenya/modeling_nenya.py (itself a port of the training script) onto vLLM's layers:
    embedding -> RMSNorm (fp32) -> 4 mHC streams
    32 blocks: [Engram before blocks 1 and 14] -> mHC pre -> MLA (9 blocks) | GDN-2 (23) -> mHC post
                                                 -> mHC pre -> dense SwiGLU (blocks 0-1) | latent MoE -> mHC post
    sum of streams -> RMSNorm -> tied head (rows past the tokenizer's 100,278 ids masked)
Dtype flow follows the release under bf16 autocast: matmuls in bf16, norms/gates/router/Sinkhorn in fp32, the
residual streams stored in bf16 after every sublayer (fp32 before the first one).

Not supported (asserted): tensor/pipeline parallelism, speculative decoding, prompt embeddings, several documents
in one request (<|endoftext|> inside a prompt: the release masks across it, this port does not).
"""

import os
from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from nenya_vllm.layers.engram import ENGRAM_STATE_DTYPE, NenyaEngram, engram_state_shape
from nenya_vllm.layers.gdn2 import NenyaGatedDeltaNet2, gdn2_state_dtype, gdn2_state_shape
from nenya_vllm.layers.mhc import NenyaMHC
from nenya_vllm.layers.mla import NenyaMLA
from nenya_vllm.layers.moe import NenyaMLP, NenyaMoE
from nenya_vllm.ops.small_kernels import rms_norm_fp32w

logger = init_logger(__name__)

# fused kernels (mHC boundary, small-batch MoE, ...); 0 = the original per-op path, kept for A/B comparisons
FUSED = os.environ.get("NENYA_FUSED", "1") == "1"
# cuBLAS bf16 GEMMs with fp32 split-K reductions: PyTorch's default lets cuBLAS sum split-K partials in bf16. Turning
# that off brought vLLM's log-probs ~5.5% closer to the fp32 model (mean |dlogprob| 0.02292 -> 0.02174 on the 16K-token
# parity set, 0.02489 -> 0.02360 on the 1K one) at no measured speed cost (RTX 5090); NENYA_FP32_REDUCE=0 restores it.
FP32_REDUCE = os.environ.get("NENYA_FP32_REDUCE", "1") == "1"
# debugging aid (eager mode only: it syncs): report the first sublayer output with non-finite values
DEBUG_NAN = os.environ.get("NENYA_DEBUG_NAN", "0") == "1"


def _check(tag: str, t: torch.Tensor) -> None:
    if DEBUG_NAN and not torch.cuda.is_current_stream_capturing():
        bad = ~torch.isfinite(t.float())
        if bad.any():
            rows = bad.reshape(t.shape[0], -1).any(-1).nonzero().flatten().tolist()
            logger.warning("NENYA_DEBUG_NAN: non-finite values in %s (T=%d), rows %s", tag, t.shape[0], rows[:16])


class _Norm(nn.Module):
    """Holds an fp32 RMSNorm weight under the release's name (`<prefix>.weight`)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)


class NenyaBlock(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig, layer_idx: int, prefix: str) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mla = config.layer_types[layer_idx] == "full_attention"
        self.is_moe = layer_idx >= config.num_dense_layers
        self.norm_eps = config.rms_norm_eps
        d = config.hidden_size
        if self.is_mla:
            self.att_latent = NenyaMLA(config, vllm_config.cache_config, vllm_config.quant_config,
                                       prefix=f"{prefix}.att_latent")
        else:
            self.gda_att = NenyaGatedDeltaNet2(config, vllm_config, prefix=f"{prefix}.gda_att")
        if self.is_moe:
            self.ff = NenyaMoE(config, vllm_config.quant_config, prefix=f"{prefix}.ff")
        else:
            self.ff = NenyaMLP(d, config.intermediate_size, vllm_config.quant_config, prefix=f"{prefix}.ff")
        self.norm1 = _Norm(d)
        self.norm2 = _Norm(d)
        self.mHC_att = NenyaMHC(config.mhc_num_streams, d, config.mhc_sinkhorn_iters)
        self.mHC_ffn = NenyaMHC(config.mhc_num_streams, d, config.mhc_sinkhorn_iters)

    def mix(self, x16: torch.Tensor) -> torch.Tensor:
        """The token mixer (MLA or GDN-2) on the normalised bf16 input."""
        if self.is_mla:
            return self.att_latent(x16)
        out = torch.empty_like(x16)
        self.gda_att(x16, out)
        return out

    def forward(self, x_res: torch.Tensor) -> torch.Tensor:
        """x_res: [T, n, d] residual streams -> [T, n, d] bf16."""
        x32, x16, coef = self.mHC_att.pre(x_res, self.norm1.weight, self.norm_eps)
        if self.is_mla:
            out = self.att_latent(x16)
        else:
            out = torch.empty_like(x16)
            self.gda_att(x16, out)
        x_res = self.mHC_att.post(x_res, out, coef)
        x32, x16, coef = self.mHC_ffn.pre(x_res, self.norm2.weight, self.norm_eps)
        out = self.ff(x32, x16) if self.is_moe else self.ff(x16)
        return self.mHC_ffn.post(x_res, out, coef)


class NenyaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        d = config.hidden_size
        self.n_streams = config.mhc_num_streams
        self.norm_eps = config.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, d, prefix=f"{prefix}.embed_tokens")
        self.emb_norm = _Norm(d)
        self.layers = nn.ModuleList([NenyaBlock(config, vllm_config, i, prefix=f"{prefix}.layers.{i}")
                                     for i in range(config.num_hidden_layers)])
        self.engrams = nn.ModuleDict({str(lid): NenyaEngram(config, vllm_config, lid, prefix=f"{prefix}.engrams.{lid}")
                                      for lid in config.engram_layers})
        self.norm = _Norm(d)
        # token id -> compressed id (shipped in the checkpoint), read by every Engram's hash
        self.register_buffer("engram_lookup", torch.zeros(config.vocab_size, dtype=torch.long))
        for eng in self.engrams.values():
            eng.token_map = self.engram_lookup

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None, inputs_embeds: torch.Tensor | None = None,
                query_start_loc: torch.Tensor | None = None,
                ngram_context: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("Nenya needs input_ids: Engram hashes the token ids (prompt embeddings are unsupported)")
        if query_start_loc is None or ngram_context is None:
            raise RuntimeError("Nenya's model state did not provide the Engram n-gram context")
        T = input_ids.shape[0]
        n, d = self.n_streams, self.config.hidden_size
        if FUSED:
            from nenya_vllm.ops.small_kernels import embed_norm_streams

            # embedding lookup + RMSNorm (fp32) + the n identical streams in one kernel
            x_res = embed_norm_streams(input_ids, self.embed_tokens.weight, self.emb_norm.weight, self.norm_eps, n)
            return self._forward_fused(x_res, input_ids, query_start_loc, ngram_context)
        x = rms_norm_fp32w(self.embed_tokens(input_ids), self.emb_norm.weight, self.norm_eps, torch.float32)
        x_res = x.unsqueeze(1).expand(T, n, d).contiguous()
        for i, layer in enumerate(self.layers):
            eng = self.engrams[str(i)] if str(i) in self.engrams else None
            if eng is not None:
                x_res = x_res.to(torch.bfloat16)
                out = torch.empty((T, n * d), dtype=torch.bfloat16, device=x_res.device)
                eng(x_res.reshape(T, n * d), input_ids, query_start_loc, ngram_context, out)
                x_res = out.view(T, n, d)
            x_res = layer(x_res)
        h = x_res.sum(dim=1)
        return rms_norm_fp32w(h, self.norm.weight, self.norm_eps, torch.bfloat16)

    def _forward_fused(self, x_res, input_ids, query_start_loc, ngram_context) -> torch.Tensor:
        """Same computation with one mHC boundary kernel between sublayers (post of one + pre of the next); around
        an Engram the boundary splits into post -> Engram -> pre, and the last post carries the final norm."""
        from nenya_vllm.ops.mhc_boundary import mhc_boundary, mhc_post_final

        T = x_res.shape[0]
        n, d, eps = self.n_streams, self.config.hidden_size, self.norm_eps
        layers = self.layers
        x_res, x32, x16, coef = mhc_boundary(x_res, None, None, layers[0].mHC_att, layers[0].norm1.weight, eps)
        _check("boundary 0 pre", x32)
        for i, layer in enumerate(layers):
            out = layer.mix(x16)
            _check(f"layer {i} mixer ({'MLA' if layer.is_mla else 'GDN-2'})", out)
            x_res, x32, x16, coef = mhc_boundary(x_res, out, coef, layer.mHC_ffn, layer.norm2.weight, eps)
            _check(f"layer {i} boundary att->ffn", x32)
            out = layer.ff(x32, x16) if layer.is_moe else layer.ff(x16)
            _check(f"layer {i} ffn ({'MoE' if layer.is_moe else 'dense'})", out)
            if i + 1 == len(layers):
                return mhc_post_final(x_res, out, coef, self.norm.weight, eps)
            nxt = layers[i + 1]
            eng = self.engrams[str(i + 1)] if str(i + 1) in self.engrams else None
            if eng is None:
                x_res, x32, x16, coef = mhc_boundary(x_res, out, coef, nxt.mHC_att, nxt.norm1.weight, eps)
            else:
                # the Engram applies this post itself (in its decode kernel; before its own ops otherwise)
                mem = torch.empty((T, n * d), dtype=torch.bfloat16, device=x_res.device)
                eng(x_res.reshape(T, n * d), input_ids, query_start_loc, ngram_context, mem, layer_out=out,
                    coef=coef)
                x_res, x32, x16, coef = mhc_boundary(mem.view(T, n, d), None, None, nxt.mHC_att,
                                                     nxt.norm1.weight, eps)


class NenyaForCausalLM(nn.Module, HasInnerState, IsHybrid):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1 or get_pp_group().world_size != 1:
            raise NotImplementedError("Nenya runs with tensor_parallel_size=1 and pipeline_parallel_size=1; "
                                      "scale out with data-parallel replicas")
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        if FP32_REDUCE:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.model = NenyaModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.num_real_tokens = int(config.tokenizer_vocab_size)

    @staticmethod
    def get_model_state_cls():
        from nenya_vllm.model_state import NenyaModelState

        return NenyaModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None, inputs_embeds: torch.Tensor | None = None,
                **kwargs: object) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds,
                          query_start_loc=kwargs.get("query_start_loc"), ngram_context=kwargs.get("ngram_context"))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if FUSED:
            from nenya_vllm.ops.decode_gemv import LM_HEAD_MAX_T, decode_linear

            if hidden_states.shape[0] <= LM_HEAD_MAX_T and self.logits_processor.scale == 1.0 and \
                    self.logits_processor.soft_cap is None:
                # decode: the tied head as one GEMV (bf16 output like cuBLAS) with the padding rows masked in it
                org = self.logits_processor.org_vocab_size
                return decode_linear(hidden_states, self.lm_head.weight[:org], n_valid=self.num_real_tokens)
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.num_real_tokens < logits.shape[-1]:
            # padding rows: never trained, no text maps to them
            logits[..., self.num_real_tokens:] = float("-inf")
        return logits

    # ---- hybrid state declarations (sizes for the KV-cache manager) ----
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        cfg = vllm_config.model_config.hf_text_config
        return gdn2_state_shape(cfg.num_attention_heads, cfg.linear_head_dim, cfg.linear_conv_kernel_dim)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return gdn2_state_dtype(vllm_config)

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig):
        from vllm.v1.kv_cache_interface import MambaSpec

        cfg = vllm_config.model_config.hf_text_config
        return (
            MambaSpec(shapes=cls.get_mamba_state_shape_from_config(vllm_config),
                      dtypes=cls.get_mamba_state_dtype_from_config(vllm_config), block_size=-1),
            MambaSpec(shapes=engram_state_shape(cfg), dtypes=ENGRAM_STATE_DTYPE, block_size=-1, tp_replicated=True),
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(cls, mamba_types):
        funcs = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: MambaStateCopyFuncCalculator.short_conv_state_copy_func(),
        }
        missing = set(mamba_types) - funcs.keys()
        assert not missing, f"missing state copy funcs for {missing}"
        return {t: funcs[t] for t in mamba_types}

    # ---- weights ----
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return NenyaWeightLoader(self).load(weights)


class NenyaWeightLoader:
    """Maps the release checkpoint (1,297 tensors) onto the vLLM modules."""

    GDN_SHARDS = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "b_proj": 3, "w_proj": 4, "f_proj.0": 5, "g_proj.0": 6}
    CONV_SHARDS = {"q_conv1d": 0, "k_conv1d": 1, "v_conv1d": 2}

    def __init__(self, model: NenyaForCausalLM) -> None:
        self.model = model
        self.config = model.config
        self.params = dict(model.named_parameters())
        # the checkpoint's integer buffers: the compressed-id map and each Engram's hashing tables
        buffer_names = ["model.engram_lookup"]
        for lid in self.config.engram_layers:
            buffer_names += [f"model.engrams.{lid}.{b}" for b in ("table_sizes", "table_offsets", "multipliers")]
        all_buffers = dict(model.named_buffers())
        self.buffers = {name: all_buffers[name] for name in buffer_names}
        self.loaded: set[str] = set()

    def _param(self, name: str) -> nn.Parameter:
        if name not in self.params:
            raise KeyError(f"no parameter {name} for a checkpoint tensor")
        return self.params[name]

    def _load(self, name: str, tensor: torch.Tensor, *shard) -> None:
        param = self._param(name)
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, tensor, *shard)
        self.loaded.add(name)

    def _copy(self, name: str, tensor: torch.Tensor, rows: slice | None = None, cols: slice | None = None) -> None:
        param = self._param(name)
        target = param.data if rows is None else param.data[rows]
        if cols is not None:
            target = target[:, cols]
        if target.shape != tensor.shape:
            raise ValueError(f"{name}{'' if rows is None else rows}: shape {tuple(target.shape)} vs "
                             f"checkpoint {tuple(tensor.shape)}")
        target.copy_(tensor.to(target.dtype))
        self.loaded.add(name)

    def _buffer(self, name: str, tensor: torch.Tensor) -> None:
        buf = self.buffers[name]
        if buf.shape != tensor.shape:
            raise ValueError(f"buffer {name}: shape {tuple(buf.shape)} vs checkpoint {tuple(tensor.shape)}")
        buf.copy_(tensor.to(buf.dtype))

    def load(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        cfg = self.config
        kd = cfg.num_attention_heads * cfg.linear_head_dim
        d = cfg.hidden_size
        mla_pending: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        buffers_seen = set()
        for name, w in weights:
            if name == "lm_head.weight" and cfg.tie_word_embeddings:
                continue
            if name in self.buffers:
                self._buffer(name, w)
                buffers_seen.add(name)
                continue
            if self._load_layer(name, w, kd, d, mla_pending):
                continue
            # everything else keeps the release's name
            self._load(name, w)

        # MLA: W_KU and W_VU interleaved per head into kv_b_proj
        H, hd = cfg.num_attention_heads, d // cfg.num_attention_heads
        for prefix, parts in mla_pending.items():
            if set(parts) != {"W_KU", "W_VU"}:
                raise ValueError(f"{prefix}: incomplete MLA up-projections {sorted(parts)}")
            ku = parts["W_KU"].view(H, hd, -1)
            vu = parts["W_VU"].view(H, hd, -1)
            self._load(f"{prefix}.kv_b_proj.weight", torch.cat([ku, vu], dim=1).reshape(H * 2 * hd, -1))

        missing_buffers = set(self.buffers) - buffers_seen
        if missing_buffers:
            raise ValueError(f"checkpoint lacks buffers {sorted(missing_buffers)}")
        self._finalize()
        return self.loaded

    def _load_layer(self, name: str, w: torch.Tensor, kd: int, d: int, mla_pending) -> bool:
        parts = name.split(".")
        # Engram: model.engrams.L.<...>
        if len(parts) > 3 and parts[1] == "engrams":
            base = ".".join(parts[:3])
            rest = ".".join(parts[3:])
            if rest == "tables.weight":
                self._copy(f"{base}.tables", w)
            elif rest.startswith("W_K."):
                j, kind = int(parts[4]), parts[5]
                self._load(f"{base}.kv_proj.{kind}", w, j)
            elif rest.startswith("W_V."):
                self._load(f"{base}.kv_proj.{parts[4]}", w, self.config.mhc_num_streams)
            elif rest in ("norm_h.weight", "norm_k.weight", "norm_v.weight"):
                self._copy(f"{base}.{parts[3]}", w)
            elif rest.startswith("conv."):
                j = int(parts[4])
                self._copy(f"{base}.conv_weight", w.reshape(d, -1), slice(j * d, (j + 1) * d))
            else:
                return False
            return True
        if len(parts) < 5 or parts[1] != "layers":
            return False
        base = ".".join(parts[:3])           # model.layers.N
        sub = parts[3]
        rest = ".".join(parts[4:])
        if sub == "att_latent":
            mapping = {"out_proj.weight": "o_proj.weight", "out_proj.bias": "o_proj.bias",
                       "kv_norm.weight": "kv_norm_weight"}
            # [W_KVD | W_query | W_gate] -> one matrix
            kqg_rows = {"W_KVD.weight": 0, "W_query.weight": 1, "W_gate.weight": 2}
            if rest in ("W_KU.weight", "W_VU.weight"):
                mla_pending[f"{base}.att_latent"][rest.split(".")[0]] = w
            elif rest in kqg_rows:
                splits = self.model.model.layers[int(parts[2])].att_latent.splits
                start = sum(splits[:kqg_rows[rest]])
                self._copy(f"{base}.att_latent.w_kqg", w, rows=slice(start, start + splits[kqg_rows[rest]]))
            elif rest in mapping:
                self._load(f"{base}.att_latent.{mapping[rest]}", w)
            else:
                raise KeyError(f"unexpected MLA tensor {name}")
            return True
        if sub == "gda_att":
            stem = rest.rsplit(".", 1)[0]
            if stem in self.GDN_SHARDS and rest.endswith(".weight"):
                self._load(f"{base}.gda_att.in_proj.weight", w, self.GDN_SHARDS[stem])
            elif stem in self.CONV_SHARDS:
                j = self.CONV_SHARDS[stem]
                self._copy(f"{base}.gda_att.conv_weight", w.reshape(kd, -1), slice(j * kd, (j + 1) * kd))
            elif rest == "f_proj.1.weight":
                self._load(f"{base}.gda_att.f_b_proj.weight", w)
            elif rest in ("g_proj.1.weight", "g_proj.1.bias"):
                self._load(f"{base}.gda_att.g_b_proj.{parts[-1]}", w)
            elif rest == "o_proj.weight":
                self._load(f"{base}.gda_att.o_proj.weight", w)
            elif rest == "o_norm.weight":
                self._copy(f"{base}.gda_att.o_norm_weight", w)
            elif rest in ("A_log", "dt_bias"):
                self._copy(f"{base}.gda_att.{rest}", w)
            else:
                raise KeyError(f"unexpected GDN-2 tensor {name}")
            return True
        if sub == "ff":
            if rest in ("fc1.weight", "fc2.weight"):                     # dense SwiGLU
                self._load(f"{base}.ff.gate_up_proj.weight", w, 0 if rest.startswith("fc1") else 1)
            elif rest == "fc3.weight":
                self._load(f"{base}.ff.down_proj.weight", w)
            elif rest in ("fc1_shared.weight", "fc2_shared.weight", "fc3_shared.weight", "expert_down.weight",
                          "expert_up.weight") and f"{base}.ff.w_in" in self.params:
                # merged layout (layers/moe.py): w_in = [expert_down | fc1 | fc2], w_out = [expert_up | fc3]
                lat, hs = self.config.moe_latent_size, self.config.moe_intermediate_size * self.config.num_shared_experts
                if rest == "expert_down.weight":
                    self._copy(f"{base}.ff.w_in", w, rows=slice(0, lat))
                elif rest == "fc1_shared.weight":
                    self._copy(f"{base}.ff.w_in", w, rows=slice(lat, lat + hs))
                elif rest == "fc2_shared.weight":
                    self._copy(f"{base}.ff.w_in", w, rows=slice(lat + hs, lat + 2 * hs))
                elif rest == "expert_up.weight":
                    self._copy(f"{base}.ff.w_out", w, cols=slice(0, lat))
                else:
                    self._copy(f"{base}.ff.w_out", w, cols=slice(lat, lat + hs))
            elif rest in ("fc1_shared.weight", "fc2_shared.weight"):
                self._load(f"{base}.ff.shared_experts.gate_up_proj.weight", w,
                           0 if rest.startswith("fc1") else 1)
            elif rest == "fc3_shared.weight":
                self._load(f"{base}.ff.shared_experts.down_proj.weight", w)
            elif rest == "gate.weight":
                self._copy(f"{base}.ff.gate_weight", w)
            elif rest == "balancer.bias":
                self._copy(f"{base}.ff.balancer_bias", w)
            elif rest in ("expert_down.weight", "expert_up.weight"):
                self._load(f"{base}.ff.{rest}", w)
            elif rest == "w_gate_up":
                self._load_experts(base, w, "w13")
            elif rest == "w_down":
                self._load_experts(base, w, "w2")
            else:
                raise KeyError(f"unexpected FFN tensor {name}")
            return True
        return False                          # norm1/norm2, mHC: release names

    def _load_experts(self, base: str, w: torch.Tensor, stem: str) -> None:
        # release layout is [E, in, out] (x @ w); vLLM's experts are [E, out, in], w13 = [gate; up]
        name = f"{base}.ff.experts.routed_experts.{stem}_weight"
        if name not in self.params:                  # routed-only FusedMoE (the merged layout)
            name = f"{base}.ff.experts.{stem}_weight"
        param = self._param(name)
        num_experts = w.shape[0]
        for e in range(num_experts):
            if stem == "w13":
                half = w.shape[2] // 2
                param.weight_loader(param, w[e, :, :half].t(), f"{stem}_weight", "w1", e)
                param.weight_loader(param, w[e, :, half:].t(), f"{stem}_weight", "w3", e)
            else:
                param.weight_loader(param, w[e].t(), f"{stem}_weight", "w2", e)
        self.loaded.add(name)

    @torch.no_grad()
    def _finalize(self) -> None:
        model = self.model.model
        for layer in model.layers:
            layer.mHC_att.finalize()
            layer.mHC_ffn.finalize()
        lookup = model.engram_lookup
        for key, eng in model.engrams.items():
            # the n-gram products must fit in int64, as the training code guaranteed
            if int(eng.multipliers.max()) * int(lookup.max()) >= 2 ** 63:
                raise ValueError(f"engram {key}: hash multipliers would overflow int64")
            sizes = eng.table_sizes.reshape(-1)
            if int(eng.table_offsets[-1] + sizes[-1]) != eng.tables.shape[0]:
                raise ValueError(f"engram {key}: table offsets/sizes do not match the table height")
