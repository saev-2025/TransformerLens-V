"""Hooked Transformer Transformer Block Component.

This module contains all the component :class:`TransformerBlock`.
"""

from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import (
    Attention,
    ChameleonAttention,
    GroupedQueryAttention,
    LayerNorm,
    LayerNormPre,
    RMSNorm,
    RMSNormPre,
    TransformerBlock,
)
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.factories.mlp_factory import MLPFactory
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
from transformer_lens.utils import repeat_along_head_dimension


# Transformer Block
class ChameleonBlock(TransformerBlock):

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig], block_index):
        super().__init__(cfg, block_index)
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        normalization_layer: Callable  # type: ignore
        normalization_layer_after: Callable  # type: ignore

        self.normalization_type = self.cfg.normalization_type

        if self.normalization_type == "LN":
            normalization_layer = LayerNorm
        elif self.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            normalization_layer = LayerNormPre
        elif self.normalization_type == "RMS":
            normalization_layer = RMSNorm
        elif self.normalization_type == "RMSPre":
            normalization_layer = RMSNormPre
        elif self.normalization_type is None:
            # This should just be the identity.
            # We need to make this a lambda so we can call it on the config, just like the others
            normalization_layer = lambda cfg: nn.Identity()
        else:
            raise ValueError(f"Invalid normalization_type passed in: {self.normalization_type}")

        if self.cfg.use_normalization_before_and_after:
            # If we use LN before and after, we do *not* fold in the weights to the LN
            # after, though we can fold for the one before.
            if self.normalization_type is None:
                normalization_layer_after = lambda cfg: nn.Identity()
            elif self.normalization_type.startswith("RMS"):
                normalization_layer_after = RMSNorm
            elif self.normalization_type.startswith("LayerNorm"):
                normalization_layer_after = LayerNorm

        self.ln1 = normalization_layer(cfg)
        if self.cfg.use_normalization_before_and_after:
            self.ln1_post = normalization_layer_after(cfg)
        if not self.cfg.attn_only:
            self.ln2 = normalization_layer(cfg)
            if self.cfg.use_normalization_before_and_after:
                self.ln2_post = normalization_layer_after(cfg)

        attention = ChameleonAttention
        if not self.cfg.use_local_attn:
            self.attn = attention(self.cfg, "global", block_index)
        else:
            if self.cfg.attn_types is None:
                raise ValueError("attn_types must be set when using local attention")
            attn_type = self.cfg.attn_types[block_index]
            self.attn = attention(self.cfg, attn_type, block_index)
        if not self.cfg.attn_only:
            self.mlp = MLPFactory.create_mlp(self.cfg)

        self.hook_attn_in = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]

        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]