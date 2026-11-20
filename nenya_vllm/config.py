# SPDX-License-Identifier: Apache-2.0
# Configuration of Nenya-2.5B-A0.5B for vLLM.
#
# The same fields and defaults as the release's configuration_nenya.py (release/HF/nenya/), so the Hub repo's
# config.json loads unchanged, plus the DeepSeek-style MLA names vLLM reads to size the latent KV cache and to
# pick MLA backends (kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, q_lora_rank). The plugin
# registers this class for model_type "nenya", so no remote code from the Hub runs on the server.

from transformers import PretrainedConfig


def default_layer_types(num_hidden_layers):
    # The trained layout: MLA at blocks 0, 3, 7, 11, ... and the last block, GDN-2 everywhere else.
    return ["full_attention" if i == 0 or i == num_hidden_layers - 1 or i % 4 == 3 else "linear_attention"
            for i in range(num_hidden_layers)]


class NenyaConfig(PretrainedConfig):
    model_type = "nenya"
    keys_to_ignore_at_inference = ["past_key_values"]

    # MLA has no query low-rank projection (W_query maps the hidden state straight to the heads).
    q_lora_rank = None

    def __init__(
        self,
        vocab_size=100352,
        hidden_size=960,
        num_hidden_layers=32,
        num_attention_heads=12,
        layer_types=None,
        num_dense_layers=2,
        intermediate_size=2560,
        num_experts=120,
        num_experts_per_tok=6,
        num_shared_experts=2,
        moe_intermediate_size=384,
        moe_latent_size=384,
        mla_latent_dim=320,
        mla_shared_key_dim=32,
        mla_kv_norm_eps=1e-6,
        attention_bias=False,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
        mhc_num_streams=4,
        mhc_sinkhorn_iters=20,
        engram_layers=None,
        engram_ngram_orders=None,
        engram_num_heads=8,
        engram_dim=960,
        engram_table_sizes=None,
        engram_pad_id=None,
        engram_pad_token=100257,
        engram_conv_kernel_dim=4,
        rms_norm_eps=1e-5,
        max_position_embeddings=16384,
        initializer_range=0.02,
        doc_masking=True,
        eot_token_id=100257,
        tokenizer_vocab_size=100278,
        use_cache=True,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=100257,
        pad_token_id=100277,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_types = list(layer_types) if layer_types is not None else default_layer_types(num_hidden_layers)
        self.num_dense_layers = num_dense_layers
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.mla_latent_dim = mla_latent_dim
        self.mla_shared_key_dim = mla_shared_key_dim
        self.mla_kv_norm_eps = mla_kv_norm_eps
        self.attention_bias = attention_bias
        self.linear_head_dim = linear_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.mhc_num_streams = mhc_num_streams
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.engram_layers = [int(x) for x in (engram_layers if engram_layers is not None else [1, 14])]
        self.engram_ngram_orders = sorted(int(x) for x in (engram_ngram_orders if engram_ngram_orders is not None
                                                           else [2, 3]))
        self.engram_num_heads = engram_num_heads
        self.engram_dim = engram_dim
        self.engram_table_sizes = ({str(k): [list(map(int, row)) for row in v] for k, v in engram_table_sizes.items()}
                                   if engram_table_sizes is not None else None)
        self.engram_pad_id = engram_pad_id
        self.engram_pad_token = engram_pad_token
        self.engram_conv_kernel_dim = engram_conv_kernel_dim
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.doc_masking = doc_masking
        self.eot_token_id = eot_token_id
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.use_cache = use_cache

        if hidden_size % num_attention_heads:
            raise ValueError(f"hidden_size {hidden_size} must split evenly across {num_attention_heads} heads")
        if len(self.layer_types) != num_hidden_layers:
            raise ValueError(f"layer_types has {len(self.layer_types)} entries for {num_hidden_layers} layers")
        bad = set(self.layer_types) - {"full_attention", "linear_attention"}
        if bad:
            raise ValueError(f"unknown layer types {sorted(bad)}")
        if not all(0 <= layer < num_hidden_layers for layer in self.engram_layers):
            raise ValueError(f"engram_layers {self.engram_layers} out of range for {num_hidden_layers} layers")
        n_tables = len(self.engram_ngram_orders) * engram_num_heads
        if engram_dim % n_tables:
            raise ValueError(f"engram_dim {engram_dim} must split evenly across {n_tables} tables")

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    # ---- derived sizes (as the release config) ----
    @property
    def mla_head_dim(self):
        # per-head width of the queries'/keys' non-shared part and of the values: 960 / 12 = 80
        return self.hidden_size // self.num_attention_heads

    @property
    def engram_head_dim(self):
        # width of one table row: 960 / (2 orders x 8 heads) = 60
        return self.engram_dim // (len(self.engram_ngram_orders) * self.engram_num_heads)

    # ---- DeepSeek-style MLA names read by vLLM ----
    @property
    def kv_lora_rank(self):
        return self.mla_latent_dim

    @property
    def qk_rope_head_dim(self):
        # the key part shared by all heads (Nenya has no positional encoding: it is NOT rotated)
        return self.mla_shared_key_dim

    @property
    def qk_nope_head_dim(self):
        return self.mla_head_dim

    @property
    def v_head_dim(self):
        return self.mla_head_dim

    @property
    def num_key_value_heads(self):
        return self.num_attention_heads
