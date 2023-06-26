from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_engine.pytorch import *

from lit_gpt.config import Config
from lit_gpt.model import apply_rope, GPT as BaseModel, RoPECache, KVCache
from lit_gpt.utils import find_multiple


ATTN_MASK_TYPE = "causal"  # use "padding" for inference


class GPT(BaseModel):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict({"h": nn.ModuleList(Block(config, i) for i in range(config.n_layer))})
        self.ln_f_lm_head = LayerNormLinear(
            config.n_embd, config.padded_vocab_size, bias=False, params_dtype=torch.get_default_dtype()
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (Linear, LayerNormLinear)):
            # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L10
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L12
            module.eps = 1e-5
        if isinstance(module, (LayerNormLinear, LayerNormMLP)):
            torch.nn.init.ones_(module.layer_norm_weight)
            torch.nn.init.zeros_(module.layer_norm_bias)
        if isinstance(module, LayerNormMLP):
            torch.nn.init.normal_(module.fc1_weight, mean=0.0, std=0.02)
            if module.fc1_bias is not None:
                torch.nn.init.zeros_(module.fc1_bias)
            torch.nn.init.normal_(module.fc2_weight, mean=0.0, std=0.02)
            if module.fc2_bias is not None:
                torch.nn.init.zeros_(module.fc2_bias)

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
        padding_multiple: int = 8,
    ) -> torch.Tensor:
        B, T = idx.size()
        use_kv_cache = input_pos is not None

        T_padded = find_multiple(T, padding_multiple)
        padding = T_padded - T
        if padding > 0:  # fp8 support
            idx = F.pad(idx, (0, padding))  # right padding
            if max_seq_length == T:
                max_seq_length = T_padded
            T = T_padded

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                T <= max_seq_length
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if (use_kv_cache or padding > 0) and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        cos, sin = self.rope_cache
        if use_kv_cache:
            cos = cos.index_select(0, input_pos)
            cos = torch.cat((cos, torch.zeros(padding, cos.size(1), device=cos.device)))
            sin = sin.index_select(0, input_pos)
            sin = torch.cat((sin, torch.zeros(padding, sin.size(1), device=sin.device)))
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
            mask = F.pad(mask, (0, 0, 0, padding))
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None

        # forward the model itself
        x = self.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, (cos, sin), max_seq_length, mask, padding=padding)
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, cos.size(-1))
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i], padding)

        logits = self.ln_f_lm_head(x)  # (b, t, vocab_size)

        logits = logits[:, : -padding or None]

        return logits

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        mapping = {
            "transformer.wte.weight": "wte.weight",
            "transformer.ln_f.weight": "ln_f_lm_head.layer_norm_weight",
            "transformer.ln_f.bias": "ln_f_lm_head.layer_norm_bias",
            "lm_head.weight": "ln_f_lm_head.weight",
        }
        for checkpoint_name, attribute_name in mapping.items():
            full_checkpoint_name = prefix + checkpoint_name
            if full_checkpoint_name in state_dict:
                full_attribute_name = prefix + attribute_name
                state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.norm_1_attn = LayerNormLinear(
            config.n_embd, shape, bias=config.bias, params_dtype=torch.get_default_dtype()
        )
        self.attn = DotProductAttention(
            num_attention_heads=config.n_head,
            kv_channels=config.head_size,
            attn_mask_type=ATTN_MASK_TYPE,
            layer_number=block_idx + 1,
        )
        # output projection
        self.proj = Linear(config.n_embd, config.n_embd, bias=config.bias, params_dtype=torch.get_default_dtype())
        if config.shared_attention_norm:
            raise NotImplementedError
        self.norm_2_mlp = LayerNormMLP(
            hidden_size=config.n_embd, ffn_hidden_size=4 * config.n_embd, params_dtype=torch.get_default_dtype()
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        padding: int = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.norm_1_attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        # each group has 1+ queries, 1 key, and 1 value (hence the + 2)
        qkv = qkv.view(B, T, self.config.n_query_groups, q_per_kv + 2, self.config.head_size).permute(0, 2, 3, 1, 4)
        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
        if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
            # for MHA this is a no-op
            k = k.repeat_interleave(q_per_kv, dim=2)
            v = v.repeat_interleave(q_per_kv, dim=2)
        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.view(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.view(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        n_elem = int(self.config.rotary_percentage * self.config.head_size)

        cos, sin = rope
        q_roped = apply_rope(q[..., :n_elem], cos, sin)
        k_roped = apply_rope(k[..., :n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if input_pos is not None and kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            padding_idx = cache_k.size(2) - 1  # send padding data to a padding index, doesn't matter which
            input_pos = torch.cat(
                (input_pos, torch.full((padding,), padding_idx, device=input_pos.device, dtype=input_pos.dtype))
            )
            k = cache_k.index_copy_(2, input_pos, k)
            v = cache_v.index_copy_(2, input_pos, v)
            kv_cache = k, v

        # flash attn requires (T, B, nh, hs)
        q = q.permute(2, 0, 1, 3)
        k = k.permute(2, 0, 1, 3)
        v = v.permute(2, 0, 1, 3)
        y = self.attn(q, k, v, attention_mask=(~mask if mask is not None else None))
        y = y.transpose(0, 1)

        # output projection
        h = self.proj(y)

        if self.config.parallel_residual:
            x = x + h + self.norm_2_mlp(x)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = x + h
            x = x + self.norm_2_mlp(x)
        return x, kv_cache

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        mapping = {
            "norm_1.weight": "norm_1_attn.layer_norm_weight",
            "norm_1.bias": "norm_1_attn.layer_norm_bias",
            "attn.attn.weight": "norm_1_attn.weight",
            "attn.attn.bias": "norm_1_attn.bias",
            "attn.proj.weight": "proj.weight",
            "attn.proj.bias": "proj.bias",
            "norm_2.weight": "norm_2_mlp.layer_norm_weight",
            "norm_2.bias": "norm_2_mlp.layer_norm_bias",
            "mlp.fc.weight": "norm_2_mlp.fc1_weight",
            "mlp.fc.bias": "norm_2_mlp.fc1_bias",
            "mlp.proj.weight": "norm_2_mlp.fc2_weight",
            "mlp.proj.bias": "norm_2_mlp.fc2_bias",
        }
        for checkpoint_name, attribute_name in mapping.items():
            full_checkpoint_name = prefix + checkpoint_name
            if full_checkpoint_name in state_dict:
                full_attribute_name = prefix + attribute_name
                state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
