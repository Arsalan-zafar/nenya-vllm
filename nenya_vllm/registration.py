# SPDX-License-Identifier: Apache-2.0
"""Everything the plugin registers with vLLM. Runs once per process, before any model config is read.

Besides the model and its config, Nenya needs four hooks into vLLM 0.33 internals (all done here, no fork):

1. MLA detection. vLLM enables its latent-attention path (latent KV cache, MLA backends, hybrid page sizing) only
   for a hard-coded list of model types; a model-arch convertor for "nenya" switches it on.
2. MLA prefill head shape. vLLM's FlashAttention MLA prefill backend accepts a whitelist of (qk_nope, rope, v)
   shapes; Nenya's (80, 32, 80) is added by overriding the FLASH_ATTN prefill backend class (mla_prefill.py).
3. Graph splitting. Nenya's stateful layers run as custom ops (gdn2.py, engram.py) that torch.compile must not
   trace through; they are added to the compilation config's splitting ops.
4. Breakable CUDA graphs by default: the stateful layers run eagerly between captured graph segments and nothing
   goes through Inductor (Inductor's reordering of the projections around a recurrent layer compounds through the
   state; see vLLM's own OLMo-Hybrid GDN layer for the same observation).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ARCHITECTURE = "NenyaForCausalLM"
MODEL_TYPE = "nenya"
SPLITTING_OPS = ("vllm::nenya_gdn2_forward", "vllm::nenya_engram_forward")

_registered = False


def register_all() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    _register_config()
    _register_model()
    _register_mla_arch_convertor()
    _register_mla_prefill_backend()
    _register_splitting_ops()
    _default_to_breakable_cudagraph()
    _register_tool_parser()


def _register_config() -> None:
    from transformers import AutoConfig

    from .config import NenyaConfig

    AutoConfig.register(MODEL_TYPE, NenyaConfig, exist_ok=True)
    # vLLM's own registry: a registered model type is loaded with this class and is no longer treated as remote
    # code, even when the server was started with --trust-remote-code.
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY[MODEL_TYPE] = NenyaConfig


def _register_model() -> None:
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        # a string, so the model module (and CUDA) is only imported where the model is built
        ModelRegistry.register_model(ARCHITECTURE, "nenya_vllm.model:NenyaForCausalLM")


def _ensure_mla_aliases(cfg) -> None:
    """A config loaded from somewhere else (e.g. the release's remote code) lacks the DeepSeek-style names."""
    aliases = {
        "kv_lora_rank": "mla_latent_dim",
        "qk_rope_head_dim": "mla_shared_key_dim",
    }
    for alias, source in aliases.items():
        if getattr(cfg, alias, None) is None and hasattr(cfg, source):
            setattr(cfg, alias, getattr(cfg, source))
    head = cfg.hidden_size // cfg.num_attention_heads
    for alias in ("qk_nope_head_dim", "v_head_dim"):
        if getattr(cfg, alias, None) is None:
            setattr(cfg, alias, head)
    if not hasattr(cfg, "q_lora_rank"):
        cfg.q_lora_rank = None


def _register_mla_arch_convertor() -> None:
    from vllm.transformers_utils import model_arch_config_convertor as convertors

    class NenyaModelArchConfigConvertor(convertors.ModelArchConfigConvertorBase):
        def __init__(self, hf_config, hf_text_config, revision=None):
            super().__init__(hf_config, hf_text_config, revision)
            _ensure_mla_aliases(hf_text_config)

        def is_deepseek_mla(self) -> bool:
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

    convertors.MODEL_ARCH_CONFIG_CONVERTORS[MODEL_TYPE] = NenyaModelArchConfigConvertor


def _register_mla_prefill_backend() -> None:
    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
        register_mla_prefill_backend,
    )

    # Overrides the class behind FLASH_ATTN for this process: the subclass only widens the dimension whitelist and
    # forces the zero-padded-V path, everything else is vLLM's FlashAttention prefill.
    register_mla_prefill_backend(MLAPrefillBackendEnum.FLASH_ATTN, "nenya_vllm.mla_prefill.NenyaFlashAttnPrefillBackend")


def _register_splitting_ops() -> None:
    from vllm.config.compilation import CompilationConfig

    ops = CompilationConfig._attention_ops
    for op in SPLITTING_OPS:
        if op not in ops:
            ops.append(op)


def _default_to_breakable_cudagraph() -> None:
    import vllm.config.vllm as vllm_config_module

    archs = vllm_config_module.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES
    if ARCHITECTURE not in archs:
        vllm_config_module.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = frozenset(archs | {ARCHITECTURE})
        vllm_config_module.default_breakable_cudagraph_architectures.cache_clear()


def _register_tool_parser() -> None:
    try:
        from vllm.tool_parsers import ToolParserManager
    except ImportError:  # pragma: no cover - entrypoints not installed
        return
    if "nenya" not in ToolParserManager.list_registered():
        ToolParserManager.register_lazy_module("nenya", "nenya_vllm.tool_parser", "NenyaToolParser")
