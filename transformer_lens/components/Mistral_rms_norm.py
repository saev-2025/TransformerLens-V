"""Hooked Transformer RMS Norm Component.

This module contains all the component :class:`RMSNorm`.
"""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class MistralRMSNorm(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig], length: Optional[int] = None):
        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length, dtype=self.cfg.dtype))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance=x.pow(2).mean(-1, keepdim=True)

        x = self.hook_normalized(x *self.hook_scale(torch.rsqrt(variance+self.eps) )) # [batch, pos, length]
        return self.w*x.to(input_dtype)
