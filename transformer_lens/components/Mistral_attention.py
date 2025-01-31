import math
from typing import Dict, Optional, Tuple, Union

import torch
import einops
import torch.nn as nn
from jaxtyping import Float, Int
import torch.nn.functional as F
from transformer_lens.utilities.attention import simple_attn_linear, complex_attn_linear
from transformer_lens.components.abstract_attention import AbstractAttention
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import logger
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MistralAttention(AbstractAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig],attn_type: str = "global", layer_id: Optional[int] = None):
        super().__init__(cfg, attn_type, layer_id)
        
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        if layer_id is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attn_type = attn_type

        self.W_K = nn.Parameter(
                torch.empty(
                    self.cfg.n_key_value_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=self.cfg.dtype,
                )
            )
        self.W_V = nn.Parameter(
                torch.empty(
                    self.cfg.n_key_value_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=self.cfg.dtype,
                )
            )
        self.b_K = nn.Parameter(torch.zeros(self.cfg.n_key_value_heads, self.cfg.d_head, dtype=self.cfg.dtype))
        self.b_V = nn.Parameter(torch.zeros(self.cfg.n_key_value_heads, self.cfg.d_head, dtype=self.cfg.dtype))
        self.o_proj = nn.Linear(self.cfg.n_heads * self.cfg.d_head, self.cfg.d_model, bias=False)

        self.layer_id=layer_id
        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, pos, head_index, d_model]

    def calculate_attention_scores(
        self,
        q: Float[torch.Tensor, "batch query_pos head_index d_head"],
        k: Float[torch.Tensor, "batch key_pos head_index d_head"],
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        q_ = einops.rearrange(
            q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
        )
        k_ = einops.rearrange(
            k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
        )
        attn_scores = torch.matmul(q_,k_) /math.sqrt(self.cfg.d_head) 
        # if self.cfg.attn_scores_soft_cap > 0:
        #     attn_scores = self.cfg.attn_scores_soft_cap * F.tanh(
        #         attn_scores / self.cfg.attn_scores_soft_cap
        #     )
        return attn_scores


    
    def forward(
        self,
        hidden_state,
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 kv_pos"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
        position_bias: Optional[Float[torch.Tensor, "1 head_index pos kv_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        shortformer_pos_embed is only used if self.cfg.positional_embedding_type == "shortformer", else defaults to None and is irrelevant. See HookedTransformerConfig for more details
        past_kv_cache_entry is an optional entry of past keys and values for this layer, only relevant if generating text. Defaults to None
        additive_attention_mask is an optional mask to add to the attention weights. Defaults to None.
        attention_mask is the attention mask for padded tokens. Defaults to None.
        """

        attn_fn = (
            complex_attn_linear
            if self.cfg.use_split_qkv_input or self.cfg.use_attn_in
            else simple_attn_linear
        )
        
        q = self.hook_q(attn_fn(hidden_state, self.W_Q, self.b_Q))
        k = self.hook_k(attn_fn(hidden_state, self.W_K, self.b_K))    
        v = self.hook_v(attn_fn(hidden_state, self.W_V, self.b_V))
        
        
        if past_kv_cache_entry is not None:
            # Appends the new keys and values to the cached values, and automatically updates the cache
            kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
            k, v = past_kv_cache_entry.append(k, v)
        else:
            # Not using a cache
            kv_cache_pos_offset = 0
        if self.cfg.positional_embedding_type == "rotary":
            q = self.hook_rot_q(self.apply_rotary(q, kv_cache_pos_offset, attention_mask))
            k = self.hook_rot_k(
                self.apply_rotary(k, 0, attention_mask)
            )  # keys are cached so no offset
        

        if self.cfg.dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)
        k=k.transpose(1, 2).contiguous()
        v=v.transpose(1, 2).contiguous()
        k=repeat_kv(k, self.cfg.n_heads//self.cfg.n_key_value_heads)
        v=repeat_kv(v, self.cfg.n_heads//self.cfg.n_key_value_heads)

        q_ = einops.rearrange(
            q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
        )
        attn_scores = torch.matmul(q_,k.transpose(2,3)) /math.sqrt(self.cfg.d_head)

        if self.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset, attention_mask
            )  # [batch, head_index, query_pos, key_pos]
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)  # [batch, head_index, query_pos, key_pos]
        pattern = pattern.to(self.cfg.dtype)
        pattern = pattern.to(v.device)

        z=torch.matmul(pattern,v)
        z=self.hook_z(z.transpose(1, 2).contiguous())
        z = z.reshape(z.shape[0], z.shape[1], self.cfg.d_head * self.cfg.n_heads).contiguous()
        # if not self.cfg.use_attn_result:
        # device = self.W_O.device
        w = einops.rearrange(self.W_O, "head_index d_head d_model -> (head_index d_head) d_model ").contiguous()
        w=w.transpose(0,1).contiguous()
        #einops.rearrange(self.W_O, "head_index d_head d_model -> (head_index d_head) d_model ").contiguous()
        self.o_proj.weight = nn.Parameter(w)
        out = self.o_proj(z)
               

        return out
