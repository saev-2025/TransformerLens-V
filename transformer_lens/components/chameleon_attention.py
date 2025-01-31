import math
from typing import Dict, Optional, Tuple, Union

import torch
import einops
import torch.nn as nn
from jaxtyping import Float, Int
import torch.nn.functional as F

from transformer_lens.components.abstract_attention import AbstractAttention
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

class ChameleonLayerNorm(nn.LayerNorm):
    """
    LayerNorm but computes stats only over the last dim because Chameleon applies gamma and beta
    from each shard separately to each head, instead of reducing. We can apply each head's own
    gamma/beta by repeat-interleaving weights from each shard, but the stats have to be computed
    in the last dimension. This module applies gamma/beta manually to fulfill this requirement.
    """

    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(hidden_size, *args, **kwargs)
        self.normalized_shape = (hidden_size[-1],)

    def forward(self, hidden_states):
        # print(f"Input hidden_states: {hidden_states.shape}, content: {hidden_states}")
        # print(f"Input weight: {self.weight.shape}, content: {self.weight}")
        # print(f"Input bias: {self.bias.shape}, content: {self.bias}")
        hidden_states = F.layer_norm(hidden_states, self.normalized_shape, None, None, eps=1e-5)
        hidden_states = hidden_states * self.weight + self.bias
        # print(f"Output hidden_states: {hidden_states.shape}, content: {hidden_states}")
        return hidden_states
    
class ChameleonAttention(AbstractAttention):
    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        attn_type: str = "global",
        layer_id: Optional[int] = None,
    ):
        super().__init__(cfg, attn_type, layer_id)
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg

        self.norm_Q = ChameleonLayerNorm((cfg.n_heads, cfg.d_head))
        self.norm_K = ChameleonLayerNorm((cfg.n_heads, cfg.d_head))

        self.W_K = nn.Parameter(
            torch.empty(
                self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=self.cfg.dtype
            )
        )
        self.W_V = nn.Parameter(
            torch.empty(
                self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=self.cfg.dtype
            )
        )
        self.b_K = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self.b_V = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        
    def calculate_qkv_matrices(
        self,
        query_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
        key_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
        ],
    ) -> Tuple[
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch kv_pos head_index d_head"],
        Float[torch.Tensor, "batch kv_pos head_index d_head"],
    ]:
        
        bsz, q_len, _ = query_input.shape
        
        _W_Q = einops.rearrange(self.W_Q, "head_index d_model d_head -> (head_index d_head) d_model")
        _b_Q = einops.rearrange(self.b_Q, "head_index d_head -> (head_index d_head)")
        q = self.hook_q((self.norm_Q(F.linear(query_input, _W_Q, _b_Q).reshape(-1, self.cfg.n_heads, self.cfg.d_head))).reshape(bsz, q_len, self.b_Q.shape[0], self.b_Q.shape[1]))
        
        _W_K = einops.rearrange(self.W_K, "head_index d_model d_head -> (head_index d_head) d_model")
        _b_K = einops.rearrange(self.b_K, "head_index d_head -> (head_index d_head)")
        k = self.hook_k((self.norm_K(F.linear(key_input, _W_K, _b_K).reshape(-1, self.cfg.n_heads, self.cfg.d_head))).reshape(bsz, q_len, self.b_K.shape[0], self.b_K.shape[1]))
        
        _W_V = einops.rearrange(self.W_V, "head_index d_model d_head -> (head_index d_head) d_model")
        _b_V = einops.rearrange(self.b_V, "head_index d_head -> (head_index d_head)")
        v = self.hook_v(F.linear(value_input, _W_V, _b_V).reshape(bsz, q_len, self.b_V.shape[0], self.b_V.shape[1]))
        return q, k, v