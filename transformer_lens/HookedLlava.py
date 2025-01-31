"""Hooked Transformer.

The Hooked Transformer is the core part of TransformerLens.

In common PyTorch model implementations (e.g. ones from HuggingFace) it's fairly easy to extract
model weights, but much harder to extract activations. TransformerLens aims to simplify this task by
attaching hooks to every notable activation within the model. This enables the inspection and/or
alteration of activations in individual components like attention heads and MLP layers, facilitating
a deeper understanding of the internal workings of transformers like GPT-2.
"""
import threading
import logging
import os
import pdb
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload
import math
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
from fancy_einsum import einsum
from jaxtyping import Float, Int
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from typing_extensions import Literal
import time
import transformer_lens.loading_from_pretrained as loading
import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import (
    Embed,
    LayerNorm,
    LayerNormPre,
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    MistralRMSNorm,
    TransformerBlock,
    MistralBlock,
    Unembed,
)
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.loading_from_pretrained import NON_HF_HOSTED_MODEL_NAMES

# Note - activation cache is used with run_with_cache, past_key_value_caching is used for
# generation.
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utilities import devices
from transformer_lens.utils import (
    USE_DEFAULT_VALUE,
    init_kaiming_normal_,
    init_kaiming_uniform_,
    init_xavier_normal_,
    init_xavier_uniform_,
    select_best_resolution,
)

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class Output(NamedTuple):
    """Output Named Tuple.

    Named tuple object for if we want to output both logits and loss.
    """

    logits: Float[torch.Tensor, "batch pos d_vocab"]
    loss: Loss

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`torch.LongTensor` or `np.ndarray` or `Tuple[int, int]`):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches

def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


class HookedLlava(HookedRootModule):
    """Hooked Transformer.

    Implements a full Transformer using the components :doc:`here <transformer_lens.components>`,
    with a :class:`transformer_lens.hook_points.HookPoint` on every interesting activation.

    TransformerLens comes loaded with >50 GPT-style models. Typically you initialise it with one of
    these via :meth:`from_pretrained`, although it can also be instantiated with randomly
    initialized weights via :meth:`__init__`.

    Once you've initialized the model, a common next step is to test it can do the task you're
    investigating. This can be done with :func:`transformer_lens.utils.test_prompt`.
    """

    ln_final: nn.Module

    def __init__(
        self,
        cfg: Union[HookedTransformerConfig, Dict],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        move_to_device: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
        stop_at_layer:Optional[int]=None,
    ):
        """Model initialization.

        Note that if you want to load the model from pretrained weights, you should use
        :meth:`from_pretrained` instead.

        Args:
            cfg: The config to use for the model.
            tokenizer: The tokenizer to use for the model. If not provided, it is inferred from
                `cfg.tokenizer_name` or initialized to `None`. If `None`, then the model cannot be
                passed strings, and d_vocab must be explicitly set.
            move_to_device: Whether to move the model to the device specified in cfg.
                device. Must be true if `n_devices` in the config is greater than 1, since the
                model's layers will be split across multiple devices.
            default_padding_side: Which side to pad on.
        """
        super().__init__()
        if isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a "
                "pretrained model, use HookedTransformer.from_pretrained() instead."
            )
        self.padding_side = "left"
        self.pad_token_id=-1
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.cfg.stop_at_layer=stop_at_layer
        
        tokenizer_start_time = time.time()
        if tokenizer is not None:
            self.set_tokenizer(tokenizer, default_padding_side=default_padding_side)
        elif self.cfg.tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            if self.cfg.tokenizer_name in NON_HF_HOSTED_MODEL_NAMES:
                logging.warning(
                    "%s tokenizer not loaded. Please load manually.",
                    self.cfg.tokenizer_name,
                )
            else:
                # Hugging Face defaults to use_fast to True
                use_fast = True
                # Phi model's fast tokenizer does not support adding a BOS token, use_fast
                # should be False
                if "phi" in self.cfg.tokenizer_name.lower():
                    use_fast = False
                huggingface_token = os.environ.get("HF_TOKEN", None)
                self.set_tokenizer(
                    AutoTokenizer.from_pretrained(
                        self.cfg.tokenizer_name,
                        add_bos_token=True,
                        trust_remote_code=self.cfg.trust_remote_code,
                        use_fast=use_fast,
                        token=huggingface_token,
                    ),
                    default_padding_side=default_padding_side,
                )
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and
            # will pass in tokens directly. In this case, we don't need a tokenizer.
            assert self.cfg.d_vocab != -1, "Must provide a tokenizer if d_vocab is not provided"
            self.tokenizer = None
            if default_padding_side != "right":
                logging.warning(
                    "default_padding_side is explictly given but ignored because tokenizer is not set."
                )

        tokenizer_time = time.time() - tokenizer_start_time
        print(f"Tokenizer setup time: {tokenizer_time:.2f}s")
        
        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]
        self.dtype = self.cfg.dtype
        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.cfg)
            self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]

        if self.cfg.use_hook_tokens:
            self.hook_tokens = HookPoint()  # [batch, pos]

        self.blocks = nn.ModuleList(
            [MistralBlock(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )

        if self.cfg.normalization_type == "RMS":
            self.ln_final = RMSNorm(self.cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln_final = RMSNormPre(self.cfg)
        elif self.cfg.normalization_type == "LN":
            if self.cfg.final_rms:
                self.ln_final = RMSNorm(self.cfg)
            else:
                self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            if self.cfg.final_rms:
                self.ln_final = RMSNormPre(self.cfg)
            else:
                self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type == "MistralRMSNorm":
            self.ln_final = MistralRMSNorm(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning("Invalid normalization_type passed in %s", self.cfg.normalization_type)
        self.unembed = Unembed(self.cfg)

        embed_std = 1/math.sqrt(self.cfg.d_model)
        
        embed_time = time.time() - tokenizer_time-tokenizer_start_time
        print(f"Embedding setup time: {embed_time:.2f}s")
        self.image_newline = nn.Parameter(torch.randn(self.cfg.d_model,dtype=self.dtype) * embed_std)
        
        if self.cfg.init_weights:
            self.init_weights()

        if move_to_device:
            # We load the devices in a pipeline manner - the first device gets the embed and
            # pos_embed layers and the first n_layers // n_devicees blocks,the second gets the next
            # n_layers // n_devices blocks ... the last gets the last n_layers // n_devices blocks,
            # the final normalization layer (if it exists) and the unembed layer
            self.move_model_modules_to_device()

        move_device_time=time.time() - tokenizer_time-tokenizer_start_time-embed_time 
        print(f"Move device time: {move_device_time:.2f}s")
        # Helper variable to store a small (10K-20K) dataset of training data. Empty by default, can
        # be loaded with load_sample_training_dataset
        self.dataset = None

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()
        
        set_up_time=time.time() - tokenizer_time-tokenizer_start_time-embed_time-move_device_time
        print(f"Set up time: {set_up_time:.2f}s")
        
    def check_hooks_to_add(
        self,
        hook_point,
        hook_point_name,
        hook,
        dir="fwd",
        is_permanent=False,
        prepend=False,
    ) -> None:
        if hook_point_name.endswith("attn.hook_result"):
            assert (
                self.cfg.use_attn_result
            ), f"Cannot add hook {hook_point_name} if use_attn_result_hook is False"
        if hook_point_name.endswith(("hook_q_input", "hook_k_input", "hook_v_input")):
            assert (
                self.cfg.use_split_qkv_input
            ), f"Cannot add hook {hook_point_name} if use_split_qkv_input is False"
        if hook_point_name.endswith("mlp_in"):
            assert (
                self.cfg.use_hook_mlp_in
            ), f"Cannot add hook {hook_point_name} if use_hook_mlp_in is False"
        if hook_point_name.endswith("attn_in"):
            assert (
                self.cfg.use_attn_in
            ), f"Cannot add hook {hook_point_name} if use_attn_in is False"
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_kv_cache=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        use_kv_cache=True,
        **kwargs,
    ):
        if past_kv_cache is not None:
            if isinstance(past_kv_cache, HookedTransformerKeyValueCache):
                cache_length = past_kv_cache.get_seq_length()
                past_length =cache_length
            else:
                cache_length = past_length = past_kv_cache[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_kv_cache:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_kv_cache is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if use_kv_cache:   
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_kv_cache,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                }
            )
        else:
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                }
            )
        return model_inputs

    def input_to_embed(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_model"],  # residual
        Optional[Int[torch.Tensor, "batch pos"]],  # tokens
        Optional[Float[torch.Tensor, "batch pos d_model"]],  # shortformer_pos_embed
        Optional[torch.Tensor],  # attention_mask [batch pos]
    ]:
        """Convert input to first residual stream.

        Args:
            input (Union[str, List[str], Int[torch.Tensor, "batch pos"]]): The input to the model.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side ([Literal["left", "right"], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing
                multiple strings of different lengths.
            past_kv_cache (HookedTransformerKeyValueCache, optional): If passed, we're doing caching
                and attention_mask will be stored in the cache.
        """
         
        if isinstance(input, str) or isinstance(input, list) :
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # This is only intended to support passing in a single string
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        elif isinstance(input,dict):
            tokens = input["input_ids"]
            if type(tokens)==list:
                tokens=tokens[0]
        else:
            tokens = input
        if len(tokens.shape) == 1:
            # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
            tokens = tokens[None]
        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(devices.get_device_for_block_index(0, self.cfg))

        if attention_mask is not None:
            assert attention_mask.shape == tokens.shape, (
                f"Attention mask shape {attention_mask.shape} does not match tokens shape "
                f"{tokens.shape}"
            )
            attention_mask = attention_mask.to(devices.get_device_for_block_index(0, self.cfg))
        elif (
            self.tokenizer and self.tokenizer.padding_side == "left"
        ) or past_kv_cache is not None:
            # If the padding side is left or we are using caching, we need to compute the attention
            # mask for the adjustment of absolute positional embeddings and attention masking so
            # that pad tokens are not attended.

            if prepend_bos is USE_DEFAULT_VALUE:
                prepend_bos = self.cfg.default_prepend_bos
            attention_mask = utils.get_attention_mask(self.tokenizer, tokens, prepend_bos)

            if past_kv_cache is not None:
                # past_kv_cache is not None, so we're doing caching.
                # We need to extend the previous attention_mask.
                # Update the past_kv_cache with the new attention_mask (unless it's frozen)
                attention_mask = past_kv_cache.append_attention_mask(attention_mask)
        else:
            # We separate this case from for computational efficiency.
            attention_mask = None

        # If we're doing caching, then we reuse keys and values from previous runs, as that's the
        # only way that past activations will affect the final logits. The cache contains those so
        # we don't need to recompute them. This is useful for generating text. As we have absolute
        # positional encodings, to implement this we have a `pos_offset` variable, defaulting to
        # zero, which says to offset which positional encodings are used (cached keys and values
        # were calculated with their own positional encodings).
        if past_kv_cache is None:
            pos_offset = 0
        else:
            batch_size, ctx_length = tokens.shape
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            if self.cfg.n_key_value_heads is None:
                assert num_heads_in_cache == self.cfg.n_heads
            else:
                assert num_heads_in_cache == self.cfg.n_key_value_heads
            assert d_head_in_cache == self.cfg.d_head
            pos_offset = cache_ctx_length
        if self.cfg.use_hook_tokens:
            tokens = self.hook_tokens(tokens)
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        if self.cfg.positional_embedding_type == "standard":
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, attention_mask)
            )  # [batch, pos, d_model]
            residual = embed + pos_embed  # [batch, pos, d_model]
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "shortformer":
            # If we're using shortformer style attention, we don't add the positional embedding to
            # the residual stream. See HookedTransformerConfig for details
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, attention_mask)
            )  # [batch, pos, d_model]
            residual = embed
            shortformer_pos_embed = pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            # Rotary doesn't use positional embeddings, instead they're applied when dot producting
            # keys and queries. See HookedTransformerConfig for details
            residual = embed
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "alibi":
            # ALiBi does not add positional embeddings to word embeddings,instead it biases QK attention scores.
            residual = embed
            shortformer_pos_embed = None
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )
        return residual, tokens, shortformer_pos_embed, attention_mask

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
        image_token_index=None,
        ignore_index=-100,
    ):
        """
        Merge input_ids with with image features into final embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`):
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                :abels need to be recalculated to support training (if provided)
            image_token_index (`int`, *optional*)
                Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
            ignore_index (`int`, *optional*)
                Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (8)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but image token sizes are different, then cannot infer left or right padding
                ```python
                cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
                chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)
                prompts = [
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                ]
                inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                    chart_img has 2634 tokens, while cat_img has 2340 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        image_token_index = image_token_index if image_token_index is not None else self.cfg.image_token_index
        ignore_index = ignore_index if ignore_index is not None else self.cfg.ignore_index

        if self.training and self.padding_side == "left":
            Warning(
                "Padding side is set to 'left' but the model is in training mode. For training "
                "it is recommended to set `model.padding_side='right' and `processor.tokenizer.padding_side='right'`. "
                "If that's intended, ignore this warning"
            )
        if not self.training and self.padding_side == "right":
            Warning(
                "Padding side is set to 'right' but the model is in inference mode. For correct "
                "generation results, please set `model.padding_side='left'` and `processor.tokenizer.padding_side='left'`. "
                "If that's intended, ignore this warning"
            )

        with torch.no_grad():
            # ! in llava 1.6, number of patches is variable
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = self.padding_side == "left"
            if batch_size > 1:
                if _left_padding and _right_padding:
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")
                elif _right_padding and left_padding:
                    left_padding = False
                elif _left_padding and not left_padding:
                    left_padding = True

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens = feature_lens.to(input_ids.device)
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
            attention_mask=attention_mask.to(input_ids.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_embed_dim), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                max_embed_dim = max_embed_dim.to(target_device)
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        if torch.is_floating_point(final_attention_mask):
            final_attention_mask = final_attention_mask.to(torch.int) 
        final_attention_mask |= image_to_overwrite
        # final_attention_mask = final_attention_mask.to(torch.int)
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids, final_labels, final_input_ids,image_to_overwrite

    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.cfg.vision_config["image_size"] // self.cfg.vision_config["patch_size"]
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.cfg.image_grid_pinpoints,
                    self.cfg.vision_config["image_size"],
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_newline=image_newline.to(image_feature.device)
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens
    
    def vision_embed(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
    ):  
        device=input_ids.device
        self.vision_tower = self.vision_tower.to(device)
        self.multi_modal_projector = self.multi_modal_projector.to(device)
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.cfg.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.cfg.vision_feature_select_strategy
        )
        image_to_overwrite=None
        if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                # ! infer image_num_patches from image_sizes
                
                image_num_patches = [
                    image_size_to_num_patches(
                        image_size=imsize,
                        grid_pinpoints=self.cfg.image_grid_pinpoints,
                        patch_size=self.cfg.vision_config["image_size"],
                    )
                    for imsize in image_sizes
                ]
                # figure out if pixel_values is concatenated or stacked
                if pixel_values.dim() == 5:
                    # stacking when input is (batch_size, num_patches, num_channels, height, width)
                    _pixel_values_list = [
                        pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                    ]
                    pixel_values = torch.cat(_pixel_values_list, dim=0)
                elif pixel_values.dim() != 4:
                    # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                    raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")
                self.vision_tower=self.vision_tower.to(pixel_values.device)
                image_features = self.vision_tower(pixel_values, output_hidden_states=True)
                selected_image_feature = image_features.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                self.multi_modal_projector=self.multi_modal_projector.to(pixel_values.device)
                image_features = self.multi_modal_projector(selected_image_feature)

                image_features = torch.split(image_features, image_num_patches, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"

                image_features, feature_lens = self.pack_image_features(
                    image_features,
                    image_sizes,
                    image_newline=self.image_newline,
                )

                inputs_embeds = inputs_embeds.to(image_features.dtype)
                # image_to_overwrite:(batch,length), where False is the text embeds
                inputs_embeds, attention_mask, position_ids, labels, _,image_to_overwrite = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )

        # pixel_values is not None but is empty ---> text only cases
        elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass

        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0].past_keys[:, :, :, 0]
                first_layer_past_key_value=first_layer_past_key_value.transpose(1,2)
                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)

                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        return attention_mask,position_ids,past_key_values,inputs_embeds,image_to_overwrite
   
    @overload
    def forward(
        self,
        input,
        return_type: Literal["logits"],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Loss:
        ...

    @overload
    def forward(
        self,
        input,
        return_type: Literal["loss"],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Loss:
        ...

    @overload
    def forward(
        self,
        input,
        return_type: Literal["both"],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss]:
        ...

    @overload
    def forward(
        self,
        input,
        return_type: Literal[None],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> None:
        ...

    def forward(
        self,
        input: Union[
            str,
            List[str],
            Int[torch.Tensor, "batch pos"],
            Float[torch.Tensor, "batch pos d_model"],
        ],
        return_type: Optional[str] = "logits",
        loss_per_token: bool = False,
        model_inputs: Optional[dict[str, torch.Tensor]] = None, 
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
        image_sizes: Optional[int] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision:Optional[bool]=False,
        
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:  
        # import pdb;pdb.set_trace()
        
        if vision:
            # model_inputs={
                #     "position_ids": position_ids,
                #     "past_key_values": past_key_values,
                #     "attention_mask": attention_mask,
                #     "pixel_values": pixel_values,
                #     "image_sizes": image_sizes,
                # }
            
            position_ids=model_inputs.get("position_ids")
            past_kv_cache=model_inputs.get("past_kv_cache")
            attention_mask=model_inputs.get("attention_mask")
            pixel_values=model_inputs.get("pixel_values")
            image_sizes=model_inputs.get("image_sizes")
            input_ids=model_inputs.get("input_ids")
            if type(attention_mask)==list:
                attention_mask=attention_mask[0]
                Warning("attention_mask is list, which isn't supported in vision_embed now(241010)")
            if type(input_ids)==list:
                input_ids=input_ids[0]
            if type(image_sizes)==list:
                image_sizes=image_sizes[0]
            if type(pixel_values)==list:
                pixel_values=pixel_values[0]
            
            # import pdb;pdb.set_trace()
            if position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            
            # if not torch.equal(input, input_ids):
            input=input_ids
            #     print("input is not equal to input_ids")
        
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):  

            if start_at_layer is None:
                (
                    residual,
                    tokens,
                    shortformer_pos_embed,
                    _,
                ) = self.input_to_embed(
                    input,
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                    attention_mask=None,
                    past_kv_cache=past_kv_cache,
                )
                if vision:
                    attention_mask,position_ids,past_kv_cache,inputs_embeds,image_to_overwrite = self.vision_embed(
                        inputs_embeds=residual,
                        pixel_values=pixel_values,
                        image_sizes=image_sizes,
                        attention_mask=attention_mask,
                        past_key_values=past_kv_cache,
                        input_ids=tokens,
                        position_ids=position_ids,
                    )
                    residual=inputs_embeds
                    image_indice=None
                    if image_to_overwrite!= None:
                        image_indice_list=[]
                        for i in range(len(image_to_overwrite)):
                            image_indice_list.append(image_to_overwrite[i].nonzero(as_tuple=True)[0])
                        image_indice=torch.stack(image_indice_list,dim=0)
                
            else:
                assert type(input) == torch.Tensor
                logging.warning("unexpected input type, not vision")
                residual = input
                
            # residual=inputs_embeds

            if start_at_layer is None:
                start_at_layer = 0
            # If we explicitly want to start or stop at a layer, we only iterate through the blocks
            # between those indices. Note that start_at_layer is inclusive and stop_at_layer is
            # exclusive.
            # Eg: start_at_layer==None + stop_at_layer==0 means to only run the embed.
            # Eg: start_at_layer==3 + stop_at_layer==-1 means to run from layer 3 until the end of the PENULTIMATE layer
            # with torch.cuda.amp.autocast():
            blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
            for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:  # type: ignore
                    # Note that each block includes skip connections, so we don't need
                    # residual + block(residual)
                    # If we're using multiple GPUs, we need to send the residual and shortformer_pos_embed to the correct GPU
                residual = residual.to(devices.get_device_for_block_index(i, self.cfg))
                if shortformer_pos_embed is not None:
                    shortformer_pos_embed = shortformer_pos_embed.to(
                        devices.get_device_for_block_index(i, self.cfg)
                    )
                    # import pdb; pdb.set_trace()
                    # print(residual.shape)
                residual = block(
                    residual,
                        # Cache contains a list of HookedTransformerKeyValueCache objects, one for each
                        # block
                    past_kv_cache_entry=past_kv_cache[i] if past_kv_cache is not None else None,
                    shortformer_pos_embed=shortformer_pos_embed,
                    attention_mask=attention_mask,
                )  # [batch, pos, d_model]
                    
                    # if past_kv_cache[i] is not None:
                    #     del past_kv_cache[i]

            if stop_at_layer is not None:
                # When we stop at an early layer, we end here rather than doing further computation
                return residual,image_indice

            if self.cfg.normalization_type is not None:
                residual = self.ln_final(residual)  # [batch, pos, d_model]
            if return_type is None:
                return None
            else:

                logits = self.unembed(residual)  # [batch, pos, d_vocab]
                if self.cfg.output_logits_soft_cap > 0.0:
                    logits = self.cfg.output_logits_soft_cap * F.tanh(
                        logits / self.cfg.output_logits_soft_cap
                    )
                if return_type == "logits":
                    return logits
                elif return_type =="generate_with_saev":

                    return logits,image_indice
                else:
                    assert (
                        tokens is not None
                    ), "tokens must be passed in if return_type is 'loss' or 'both'"
                    loss = self.loss_fn(logits, tokens, attention_mask, per_token=loss_per_token)
                    if return_type == "loss":
                        return loss
                    elif return_type == "both":
                        return Output(logits, loss)
                    else:
                        logging.warning(f"Invalid return_type passed in: {return_type}")
                        return None

    def loss_fn(
        self,
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        tokens: Int[torch.Tensor, "batch pos"],
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        per_token: bool = False,
    ):
        """Wrapper around `utils.lm_cross_entropy_loss`.

        Used in forward() with return_type=="loss" or "both".
        """
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        return utils.lm_cross_entropy_loss(logits, tokens, attention_mask, per_token)

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Output, ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Output, Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around `run_with_cache` in HookedRootModule.

        If return_cache_object is True, this will return an ActivationCache object, with a bunch of
        useful HookedTransformer specific methods, otherwise it will return a dictionary of
        activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        # print(f"out",out)
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def set_tokenizer(
        self,
        tokenizer,
        default_padding_side="right",
    ):
        """Set the tokenizer to use for this model.

        Args:
            tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer.
            default_padding_side (str): "right" or "left", which side to pad on.

        """
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"

        assert default_padding_side in [
            "right",
            "left",
        ], f"padding_side must be 'right' or 'left', got {default_padding_side}"

        # Use a tokenizer that is initialized with add_bos_token=True as the default tokenizer.
        # Such a tokenizer should be set as the default tokenizer because the tokenization of some
        # tokenizers like LlamaTokenizer are different when bos token is automatically/manually
        # prepended, and add_bos_token cannot be dynamically controlled after initialization
        # (https://github.com/huggingface/transformers/issues/25886).
        tokenizer_with_bos = utils.get_tokenizer_with_bos(tokenizer)
        self.tokenizer = tokenizer_with_bos
        assert self.tokenizer is not None  # keep mypy happy
        self.tokenizer.padding_side = default_padding_side

        # Some tokenizers doesn't automatically prepend the BOS token even when they are initialized
        # with add_bos_token=True. Therefore, we need this information to dynamically control prepend_bos.
        self.cfg.tokenizer_prepends_bos = len(self.tokenizer.encode("")) > 0

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        # Infer vocab size from tokenizer
        if self.cfg.d_vocab == -1:
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Int[torch.Tensor, "batch pos"]:
        """Converts a string to a tensor of tokens.

        If prepend_bos is True, prepends the BOS token to the input - this is recommended when
        creating a sequence of tokens to be input to a model.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when
        inputting a prompt to the model as the first token is often treated weirdly, but should only
        be done at the START of the prompt. Make sure to turn it off if you're looking at the
        tokenization of part of the prompt! (Note: some models eg GPT-2 were not trained with a BOS
        token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether
        the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not
        careful!

        Args:
            input (Union[str, List[str]]): The input to tokenize.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing
                multiple strings of different lengths.
            move_to_device (bool): Whether to move the output tensor of tokens to the device the
                model lives on. Defaults to True truncate (bool): If the output tokens are too long,
                whether to truncate the output tokens to the model's max context window. Does nothing
                for shorter inputs. Defaults to True.
        """
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
            assert (
                self.cfg.tokenizer_prepends_bos is not None
            ), "Set the tokenizer for the model by calling set_tokenizer"

            if self.cfg.default_prepend_bos and not self.cfg.tokenizer_prepends_bos:
                # We want to prepend bos but the tokenizer doesn't automatically do it, so we add it manually
                input = utils.get_input_with_manually_prepended_bos(self.tokenizer, input)

            tokens = self.tokenizer(
                input,
                return_tensors="pt",
                padding=True,
                truncation=truncate,
                max_length=self.cfg.n_ctx if truncate else None,
            )["input_ids"]

            if not self.cfg.default_prepend_bos and self.cfg.tokenizer_prepends_bos:
                # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
                tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)

            if move_to_device:
                tokens = tokens.to(self.cfg.device)
            return tokens

    def to_string(
        self,
        tokens: Union[
            List[int],
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "batch pos"],
            Int[torch.Tensor, "pos"],
            np.ndarray,
            List[Int[torch.Tensor, "pos"]],
        ],
    ) -> Union[str, List[str]]:
        """Tokens to String(s).

        Converts a tensor of tokens to a string (if rank 1) or a list of strings (if rank 2).

        Accepts lists of tokens and numpy arrays as inputs too (and converts to tensors internally)
        """
        assert self.tokenizer is not None, "Cannot use to_string without a tokenizer"

        if not isinstance(tokens, torch.Tensor):
            # We allow lists to be input
            tokens = torch.tensor(tokens)

        # I'm not sure what exactly clean_up_tokenization_spaces does, but if
        # it's set, then tokenization is no longer invertible, and some tokens
        # with a bunch of whitespace get collapsed together
        if len(tokens.shape) == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passed in: {tokens.shape}")

    def to_str_tokens(
        self,
        input: Union[
            str,
            Int[torch.Tensor, "pos"],
            Int[torch.Tensor, "1 pos"],
            Int[np.ndarray, "pos"],
            Int[np.ndarray, "1 pos"],
            list,
        ],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
    ) -> Union[List[str], List[List[str]]]:
        """Map text, a list of text or tokens to a list of tokens as strings.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when
        inputting a prompt to the model as the first token is often treated weirdly, but should only
        be done at the START of the prompt. If prepend_bos=None is passed, it implies the usage of
        self.cfg.default_prepend_bos which is set to True unless specified otherwise. Therefore,
        make sure to locally turn it off by passing prepend_bos=False if you're looking at the
        tokenization of part of the prompt! (Note: some models eg GPT-2 were not trained with a BOS
        token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether
        the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not
        careful!

        Gotcha3: If passing a string that exceeds the model's context length (model.cfg.n_ctx), it
        will be truncated.

        Args:
            input (Union[str, list, torch.Tensor]): The input - either a string or a tensor of
                tokens. If tokens, should be a tensor of shape [pos] or [1, pos].
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.

        Returns:
            str_tokens: List of individual tokens as strings
        """
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert self.tokenizer is not None  # keep mypy happy
            tokens: Union[np.ndarray, torch.Tensor]
            if isinstance(input, list):
                return list(
                    map(
                        lambda tokens: self.to_str_tokens(tokens, prepend_bos, padding_side),
                        input,
                    )
                )  # type: ignore
            elif isinstance(input, str):
                tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)[
                    0
                ]
                # Gemma tokenizer expects a batch dimension
                if "gemma" in self.tokenizer.name_or_path and tokens.ndim == 1:
                    tokens = tokens.unsqueeze(1)
            elif isinstance(input, torch.Tensor):
                tokens = input
                tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
                if tokens.dim() == 0:
                    # Don't pass dimensionless tensor
                    tokens = tokens.unsqueeze(0)
                assert (
                    tokens.dim() == 1
                ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
            elif isinstance(input, np.ndarray):
                tokens = input
                tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
                if tokens.ndim == 0:
                    # Don't pass dimensionless tensor
                    tokens = np.expand_dims(tokens, axis=0)
                assert (
                    tokens.ndim == 1
                ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
            else:
                raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
            str_tokens = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
            return str_tokens

    def to_single_token(self, string):
        """Map a string that makes up a single token to the id for that token.

        Raises an error for strings that are not a single token! If uncertain use to_tokens.
        """

        # We use the to_tokens method, do not append a BOS token
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        # If token shape is non-empty, raise error
        assert not token.shape, f"Input string: {string} is not a single token!"
        return token.item()

    def to_single_str_token(self, int_token: int) -> str:
        # Gives the single token corresponding to an int in string form
        assert isinstance(int_token, int)
        token = self.to_str_tokens(torch.tensor([int_token]))
        assert len(token) == 1
        return cast(str, token[0])

    def get_token_position(
        self,
        single_token: Union[str, int],
        input: Union[str, Union[Float[torch.Tensor, "pos"], Float[torch.Tensor, "1 pos"]]],
        mode="first",
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
    ):
        """Get the position of a single_token in a string or sequence of tokens.

        Raises an error if the token is not present.

        Gotcha: If you're inputting a string, it'll automatically be tokenized. Be careful about the
        setting for prepend_bos! When a string is input to the model, a BOS (beginning of sequence)
        token is prepended by default when the string is tokenized because
        self.cfg.default_prepend_bos is set to True unless specified otherwise. But this should only
        be done at the START of the input, not when inputting part of the prompt. If you're getting
        weird off-by-one errors, check carefully for what the setting should be!

        Args:
            single_token (Union[str, int]): The token to search for. Can
                be a token index, or a string (but the string must correspond to a single token).
            input (Union[str, torch.Tensor]): The sequence to
                search in. Can be a string or a rank 1 tensor of tokens or a rank 2 tensor of tokens
                with a dummy batch dimension.
            mode (str, optional): If there are multiple matches, which match to return. Supports
                "first" or "last". Defaults to "first".
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
        """
        if isinstance(input, str):
            # If the input is a string, convert to tensor
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input

        if len(tokens.shape) == 2:
            # If the tokens have shape [1, seq_len], flatten to [seq_len]
            assert (
                tokens.shape[0] == 1
            ), f"If tokens are rank two, they must have shape [1, seq_len], not {tokens.shape}"
            tokens = tokens[0]

        if isinstance(single_token, str):
            # If the single token is a string, convert to an integer
            single_token = self.to_single_token(single_token)
        elif isinstance(single_token, torch.Tensor):
            single_token = single_token.item()

        indices = torch.arange(len(tokens), device=tokens.device)[tokens == single_token]
        assert len(indices) > 0, "The token does not occur in the prompt"
        if mode == "first":
            return indices[0].item()
        elif mode == "last":
            return indices[-1].item()
        else:
            raise ValueError(f"mode must be 'first' or 'last', not {mode}")

    def tokens_to_residual_directions(
        self,
        tokens: Union[
            str,
            int,
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "pos"],
            Int[torch.Tensor, "batch pos"],
        ],
    ) -> Union[
        Float[torch.Tensor, "d_model"],
        Float[torch.Tensor, "pos d_model"],
        Float[torch.Tensor, "batch pos d_model"],
    ]:
      
        if isinstance(tokens, torch.Tensor) and tokens.numel() > 1:
            # If the tokens are a tensor, and have more than one element, assume they are a batch of
            # tokens.
            residual_directions = self.W_U[:, tokens]
            residual_directions = einops.rearrange(
                residual_directions, "d_model ... -> ... d_model"
            )
            return residual_directions
        else:
            # Otherwise there is a single token
            if isinstance(tokens, str):
                token = self.to_single_token(tokens)
            elif isinstance(tokens, int):
                token = tokens
            elif isinstance(tokens, torch.Tensor) and tokens.numel() == 1:
                token = tokens.item()
            else:
                raise ValueError(f"Invalid token type: {type(tokens)}")
            residual_direction = self.W_U[:, token]
            return residual_direction

    def to(  # type: ignore
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):
        """Wrapper around cuda that also changes `self.cfg.device`."""
        return self.to("cuda")

    def cpu(self):
        """Wrapper around cuda that also changes `self.cfg.device`."""
        return self.to("cpu")

    def mps(self):
        """Wrapper around mps that also changes `self.cfg.device`."""
        return self.to("mps")

    def move_model_modules_to_device(self):
        # all change there is temporally for 3090 situation, where the gpu memory is limited to 24GB.
        # Warning.warn("All changes in move_model_modules_to_device are temporally for 3090 situation, where the gpu memory is limited to 24GB",UserWarning)
        # if self.cfg.stop_at_layer is not None:
        #     self.cfg.n_layers=self.cfg.stop_at_layer
        self.embed.to(devices.get_device_for_block_index(0, self.cfg))
        self.hook_embed.to(devices.get_device_for_block_index(0, self.cfg))
        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed.to(devices.get_device_for_block_index(0, self.cfg))
            self.hook_pos_embed.to(devices.get_device_for_block_index(0, self.cfg))
        # import pdb;pdb.set_trace()
        if hasattr(self, "ln_final"):
            self.ln_final.to(devices.get_device_for_block_index(self.cfg.n_layers - 1, self.cfg))
        self.unembed.to(devices.get_device_for_block_index(self.cfg.n_layers - 1, self.cfg))
        
        for i, block in enumerate(self.blocks):
            if self.cfg.stop_at_layer is not None and i > self.cfg.stop_at_layer:
                break
            block.to(devices.get_device_for_block_index(i, self.cfg))

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        refactor_factored_attn_matrices: bool = False,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model: Optional[AutoModelForCausalLM] = None,
        device: Optional[Union[str, torch.device]] = None,
        n_devices: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        move_to_device: bool = True,
        fold_value_biases: bool = True,
        default_prepend_bos: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
        dtype=torch.float32,
        vision_tower=Optional[None],
        multi_modal_projector=Optional[None],
        stop_at_layer=Optional[None],
        **from_pretrained_kwargs,
    ) -> "HookedTransformer":
        
        start_time = time.time() 

        assert not (
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"
        
        quantization_time = time.time() - start_time
        print(f"Quantization check time: {quantization_time:.2f}s")

        if hf_model is not None:
            hf_cfg = hf_model.config.to_dict()
            qc = hf_cfg.get("quantization_config", {})
            load_in_4bit = qc.get("load_in_4bit", False)
            load_in_8bit = qc.get("load_in_8bit", False)
            quant_method = qc.get("quant_method", "")
            assert not load_in_8bit, "8-bit quantization is not supported"
            assert not (
                load_in_4bit and (version.parse(torch.__version__) < version.parse("2.1.1"))
            ), "Quantization is only supported for torch versions >= 2.1.1"
            assert not (
                load_in_4bit and ("llama" not in model_name.lower())
            ), "Quantization is only supported for Llama models"
            if load_in_4bit:
                assert (
                    qc.get("quant_method", "") == "bitsandbytes"
                ), "Only bitsandbytes quantization is supported"
        else:
            hf_cfg = {}
            
        config_load_time = time.time() - start_time
        print(f"Configuration loading time: {config_load_time:.2f}s")

        if isinstance(dtype, str):
            # Convert from string to a torch dtype
            dtype = DTYPE_FROM_STRING[dtype]
        if "torch_dtype" in from_pretrained_kwargs:
            # For backwards compatibility with the previous way to do low precision loading
            # This should maybe check the user did not explicitly set dtype *and* torch_dtype
            dtype = from_pretrained_kwargs["torch_dtype"]

        if (
            (from_pretrained_kwargs.get("torch_dtype", None) == torch.float16)
            or dtype == torch.float16
        ) and device in ["cpu", None]:
            logging.warning("float16 models may not work on CPU. Consider using a GPU or bfloat16.")

        # Get the model name used in HuggingFace, rather than the alias.
        official_model_name = loading.get_official_model_name(model_name)

        # pdb.set_trace()

        # Load the config into an HookedTransformerConfig object. If loading from a
        # checkpoint, the config object will contain the information about the
        # checkpoint
        cfg = loading.get_pretrained_model_config(
            official_model_name,
            hf_cfg=hf_cfg,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=fold_ln,
            device=device,
            n_devices=n_devices,
            default_prepend_bos=default_prepend_bos,
            dtype=dtype,
            **from_pretrained_kwargs,
        )
        
        config_process_time = time.time() - config_load_time - start_time
        print(f"Model configuration processing time: {config_process_time:.2f}s")
        
        # pdb.set_trace()
        if cfg.positional_embedding_type == "shortformer":
            if fold_ln:
                logging.warning(
                    "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_"
                    "ln=False instead."
                )
                fold_ln = False
            if center_unembed:
                logging.warning(
                    "You tried to specify center_unembed=True for a shortformer model, but this can't be done! "
                    "Setting center_unembed=False instead."
                )
                center_unembed = False
            if center_writing_weights:
                logging.warning(
                    "You tried to specify center_writing_weights=True for a shortformer model, but this can't be done! "
                    "Setting center_writing_weights=False instead."
                )
                center_writing_weights = False
        if center_unembed and cfg.output_logits_soft_cap > 0.0:
            logging.warning(
                "You tried to specify center_unembed=True for a model using logit softcap, but this can't be done! Softcapping is not invariant upon adding a constant"
                "Setting center_unembed=False instead."
            )
            center_unembed = False

        # Get the state dict of the model (ie a mapping of parameter names to tensors), processed to
        # match the HookedTransformer parameter names.
        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )
        
        state_dict_load_time = time.time() - config_process_time - config_load_time - start_time
        print(f"State dict loading time: {state_dict_load_time:.2f}s")

        # Create the HookedTransformer object
        model = cls(
            cfg,
            tokenizer,
            move_to_device=False,
            default_padding_side=default_padding_side,
            stop_at_layer=stop_at_layer,
        )
        
        model_creation_time = time.time() - state_dict_load_time - config_process_time - config_load_time - start_time
        print(f"Model creation time: {model_creation_time:.2f}s")

        model.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )
        
        state_dict_processing_time = time.time() - model_creation_time - state_dict_load_time - config_process_time - config_load_time - start_time
        print(f"State dict processing time: {state_dict_processing_time:.2f}s")

        if move_to_device:
            model.move_model_modules_to_device()

        device_move_time = time.time() - state_dict_processing_time - model_creation_time - state_dict_load_time - config_process_time - config_load_time - start_time
        print(f"Device moving time: {device_move_time:.2f}s")

        print(f"Total loading time: {time.time() - start_time:.2f}s")
        if vision_tower != None:
            model.vision_tower=vision_tower
            model.multi_modal_projector=multi_modal_projector
        else:
            Warning("no vision_tower")
        return model

    @classmethod
    def from_pretrained_no_processing(
        cls,
        model_name: str,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        refactor_factored_attn_matrices=False,
        fold_value_biases=False,
        dtype=torch.float32,
        default_prepend_bos=True,
        default_padding_side="right",
        **from_pretrained_kwargs,
    ):
        """Wrapper for from_pretrained.

        Wrapper for from_pretrained with all boolean flags related to simplifying the model set to
        False. Refer to from_pretrained for details.
        """
        return cls.from_pretrained(
            model_name,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            dtype=dtype,
            default_prepend_bos=default_prepend_bos,
            default_padding_side=default_padding_side,
            **from_pretrained_kwargs,
        )

    def init_weights(self):
        """Initialize weights.

        LayerNorm weights are already initialized to 1.0, and all biases are initialized to 0.0
        (including LayerNorm), so this just initializes weight matrices.

        Weight matrices are set to empty by default (to save space + compute, since they're the bulk
        of the parameters), so it is important to call this if you are not loading in pretrained
        weights! Note that this function assumes that weight names being with `W_`.

        Set seed here to ensure determinism.

        This does NOT follow the PyTorch scheme, which as far as I can tell is super out of date but
        no one has gotten round to updating it? https://github.com/pytorch/pytorch/issues/18182

        The default PyTorch scheme is the following: all linear layers use uniform(-1/sqrt(fan_in),
        1/sqrt(fan_in)) for weights, and uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for biases. For
        biases, fan_in is computed using the fan_in for the weight matrix of the linear layer. Note
        tha it *does not actually* use Kaiming initialization, despite the fact that it calls the
        function.

        However, for Transformer blocks, it instead initializes biases to zero and weights using Xavier uniform, that
        is: uniform(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))) for weights.

        PyTorch Transformers are especially bad - TransformerEncoder initializes all layers to the
        exact same weights?! https://github.com/pytorch/pytorch/issues/72253.

        The best paper I've found on transformer initialization is the muP paper, but haven't
        integrated those ideas yet: https://arxiv.org/abs/2203.03466

        We split off the initialization into separate functions because muP initialization handles
        different parts of the model differently.
        """

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        if self.cfg.init_mode == "gpt2":
            self._init_weights_gpt2()
        elif self.cfg.init_mode == "xavier_uniform":
            self._init_weights_xavier(dist_type="uniform")
        elif self.cfg.init_mode == "xavier_normal":
            self._init_weights_xavier(dist_type="normal")
        elif self.cfg.init_mode == "kaiming_uniform":
            self._init_weights_kaiming(dist_type="uniform")
        elif self.cfg.init_mode == "kaiming_normal":
            self._init_weights_kaiming(dist_type="normal")
        elif self.cfg.init_mode == "muP":
            self._init_weights_muP(dist_type="normal")  # muP uses normal initialization

    def _init_weights_gpt2(self):
        """Initialize weights with GPT-2 initialization. Biases are initialized to 0.0 and weights
        are initialized to N(0, 0.64/d_model) if initializer_range is not set, otherwise std is initializer_range.
        """
        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=self.cfg.initializer_range)

    def _init_weights_xavier(self, dist_type="normal"):
        """
        Initialize weights with Xavier initialization -- that is, scale the weights by sqrt(6 /
        (fan_in + fan_out)) for a [-1, 1] uniform distribution, or sqrt(2 / (fan_in + fan_out)) for a
        standard normal.

        Note that since TransformerLens implements the matrices in the opposite orientation to what
        torch does (e.g. it's d_in x d_out, not d_out x d_in as in torch), we need to calculate it
        ourselves.
        """
        gain = self.cfg.initializer_range
        for name, param in self.named_parameters():
            if "W_" in name:
                if dist_type == "uniform":
                    init_xavier_uniform_(param, gain=gain)
                elif dist_type == "normal":
                    init_xavier_normal_(param, gain=gain)

    def _init_weights_kaiming(self, dist_type="uniform"):
        """
        Initialize weights with Kaiming initialization -- that is, scale the weights by
        c / sqrt(fan_in), where c = sqrt(2) if the params were immediately preceded by a relu and 1 for
        everything else.

        Note that the numbers are actually incorrect here when you're using a nonlinearity other
        than relu, e.g. the correct c for SiLu is ~1.74, for tanh it's 5/3 ~= 1.67, and for GeLU it's ~1.57.
        But this is unlikely to matter in practice.

        I'm just using fan_mode = "fan_in" for now, but it should be trivial to add fan_out.

        Again, we have to implement it ourselves because of the orientation of the matrices.
        """
        gain = self.cfg.initializer_range
        for name, param in self.named_parameters():
            if "W_" in name:
                if dist_type == "uniform":
                    init_kaiming_uniform_(param, gain=gain, nonlinearity="relu", mode="fan_in")
                elif dist_type == "normal":
                    init_kaiming_normal_(param, gain=gain, nonlinearity="relu", mode="fan_in")

    def _init_weights_muP(self, dist_type="uniform"):
        """
        Initialize weights with muParameterization. This involves scaling output weights by a factor
        of 1/fan_in, input weights and biases by 1, everything else by a factor of 1/sqrt(fan_in).

        Also, you need to use muAdamW, which rescales the learning rate for output weights and
        hidden weights by a factor of 1/fan_in.

        All biases are still assumed to be initialized to 0.0, so we only need to change the
        weights.
        """
        for name, param in self.named_parameters():
            if "W_" in name:
                fan_in, _ = utils.calc_fan_in_and_fan_out(param)
                if "embed" in name:
                    scale = float(1)
                elif "unembed" in name:
                    scale = 1 / fan_in
                else:
                    scale = 1 / fan_in**0.5

                if dist_type == "uniform":
                    scale *= 3**0.5
                    nn.init.uniform_(param, -scale, scale)
                elif dist_type == "normal":
                    nn.init.normal_(param, std=scale)

    def load_and_process_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ):
        """Load & Process State Dict.

        Load a state dict into the model, and to apply processing to simplify it. The state dict is
        assumed to be in the HookedTransformer format.

        See the relevant method (same name as the flag) for more details on the folding, centering
        and processing flags.

        Args:
            state_dict (dict): The state dict of the model, in HookedTransformer format. fold_ln
            fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the
                subsequent linear layer. This does not change the computation. Defaults to True.
            center_writing_weights (bool, optional): Whether to center weights writing to the
                residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the
                computation. Defaults to True.
            center_unembed (bool, optional): Whether to center W_U (ie set mean to be zero).
                Softmax is translation invariant so this doesn't affect log probs or loss, but does
                change logits. Defaults to True.
            fold_value_biases (bool, optional): Whether to fold the value biases into the output
                bias. Because attention patterns add up to 1, the value biases always have a
                constant effect on a layer's output, and it doesn't matter which head a bias is
                associated with. We can factor this all into a single output bias to the layer, and
                make it easier to interpret the head's output.
            refactor_factored_attn_matrices (bool, optional): Whether to convert the factored
                matrices (W_Q & W_K, and W_O & W_V) to be "even". Defaults to False.
            model_name (str, optional): checks the model name for special cases of state dict
                loading. Only used for Redwood 2L model currently.
        """
        if self.cfg.dtype not in [torch.float32, torch.float64] and fold_ln:
            logging.warning(
                "With reduced precision, it is advised to use `from_pretrained_no_processing` instead of `from_pretrained`."
            )

        if (
            self.cfg.dtype not in [torch.float32, torch.float64]
            and self.cfg.num_experts
            and self.cfg.num_experts > 1
        ):
            logging.warning(
                "When running MoE models, it is advised to use a higher precision data type. See docs for more info."
            )

        state_dict = self.fill_missing_keys(state_dict)
        if fold_ln:
            if self.cfg.num_experts and self.cfg.num_experts > 1:
                logging.warning(
                    "You are using MoE, so the layer norm weights can't be folded! Skipping"
                )
            elif self.cfg.normalization_type in ["LN", "LNPre"]:
                state_dict = self.fold_layer_norm(state_dict)
            elif self.cfg.normalization_type in ["RMS", "RMSPre"]:
                state_dict = self.fold_layer_norm(
                    state_dict, fold_biases=False, center_weights=False
                )
            else:
                logging.warning(
                    "You are not using LayerNorm or RMSNorm, so the layer norm weights can't be folded! Skipping"
                )

        if center_writing_weights:
            if self.cfg.normalization_type not in ["LN", "LNPre"]:
                logging.warning(
                    "You are not using LayerNorm, so the writing weights can't be centered! Skipping"
                )
            elif self.cfg.final_rms:
                logging.warning(
                    "This model is using final RMS normalization, so the writing weights can't be centered! Skipping"
                )
            else:
                state_dict = self.center_writing_weights(state_dict)

        if center_unembed:
            state_dict = self.center_unembed(state_dict)
        if fold_value_biases:
            state_dict = self.fold_value_biases(state_dict)
        if refactor_factored_attn_matrices:
            state_dict = self.refactor_factored_attn_matrices(state_dict)

        if self.cfg.load_in_4bit:
            # with quantization, parameters should be assigned
            # so that quantization settings are not lost
            self.load_state_dict(state_dict, assign=True, strict=False)
        else:
            self.load_state_dict(state_dict, strict=False)

    def fill_missing_keys(self, state_dict):
        return loading.fill_missing_keys(self, state_dict)

    def fold_layer_norm(
        self, state_dict: Dict[str, torch.Tensor], fold_biases=True, center_weights=True
    ):
        """Fold Layer Norm. Can also be used to fold RMS Norm, when fold_biases and center_weights are set to False.

        Takes in a state dict from a pretrained model, formatted to be consistent with
        HookedTransformer but with LayerNorm weights and biases. Folds these into the neighbouring
        weights. See further_comments.md for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model.
            fold_biases (bool): Enables folding of LN biases. Should be disabled when RMS Norm is used.
            center_weights (bool): Enables the centering of weights after folding in LN. Should be disabled when RMS Norm is used.
        """

        # Models that use Grouped Query Attention (Only Mistral at the time of writing) prefix their K/V weights and
        # biases with an underscore in order to distinguish them, but folding the LN into them still works the same,
        # so we just add the underscore if GQA is used (i.e. if `cfg.n_key_value_heads is specified`).
        gqa = "" if self.cfg.n_key_value_heads is None else "_"

        for l in range(self.cfg.n_layers):
            # Fold ln1 into attention - it's important to fold biases first, since biases depend on
            # weights but not vice versa The various indexing is just to broadcast ln.b and ln.w
            # along every axis other than d_model. Each weight matrix right multiplies. To fold in
            # the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need
            # to sum along axis -2, which is the residual stream space axis.
            if fold_biases:
                state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[f"blocks.{l}.attn.b_Q"] + (
                    state_dict[f"blocks.{l}.attn.W_Q"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(-2)
                state_dict[f"blocks.{l}.attn.{gqa}b_K"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_K"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                state_dict[f"blocks.{l}.attn.{gqa}b_V"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_V"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                del state_dict[f"blocks.{l}.ln1.b"]

            state_dict[f"blocks.{l}.attn.W_Q"] = (
                state_dict[f"blocks.{l}.attn.W_Q"] * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_K"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_V"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            del state_dict[f"blocks.{l}.ln1.w"]

            # Finally, we center the weights reading from the residual stream. The output of the
            # first part of the LayerNorm is mean 0 and standard deviation 1, so the mean of any
            # input vector of the matrix doesn't matter and can be set to zero. Equivalently, the
            # output of LayerNormPre is orthogonal to the vector of all 1s (because dotting with
            # that gets the sum), so we can remove the component of the matrix parallel to this.
            if center_weights:
                state_dict[f"blocks.{l}.attn.W_Q"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_K"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_V"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )

            # Fold ln2 into MLP
            if not self.cfg.attn_only:
                if fold_biases:
                    state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[f"blocks.{l}.mlp.b_in"] + (
                        state_dict[f"blocks.{l}.mlp.W_in"]
                        * state_dict[f"blocks.{l}.ln2.b"][:, None]
                    ).sum(-2)
                    del state_dict[f"blocks.{l}.ln2.b"]

                state_dict[f"blocks.{l}.mlp.W_in"] = (
                    state_dict[f"blocks.{l}.mlp.W_in"] * state_dict[f"blocks.{l}.ln2.w"][:, None]
                )

                if self.cfg.gated_mlp:
                    state_dict[f"blocks.{l}.mlp.W_gate"] = (
                        state_dict[f"blocks.{l}.mlp.W_gate"]
                        * state_dict[f"blocks.{l}.ln2.w"][:, None]
                    )

                del state_dict[f"blocks.{l}.ln2.w"]

                if center_weights:
                    # Center the weights that read in from the LayerNormPre
                    state_dict[f"blocks.{l}.mlp.W_in"] -= einops.reduce(
                        state_dict[f"blocks.{l}.mlp.W_in"],
                        "d_model d_mlp -> 1 d_mlp",
                        "mean",
                    )

                if self.cfg.act_fn is not None and self.cfg.act_fn.startswith("solu"):
                    # Fold ln3 into activation
                    if fold_biases:
                        state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[
                            f"blocks.{l}.mlp.b_out"
                        ] + (
                            state_dict[f"blocks.{l}.mlp.W_out"]
                            * state_dict[f"blocks.{l}.mlp.ln.b"][:, None]
                        ).sum(
                            -2
                        )

                        del state_dict[f"blocks.{l}.mlp.ln.b"]

                    state_dict[f"blocks.{l}.mlp.W_out"] = (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.w"][:, None]
                    )

                    if center_weights:
                        # Center the weights that read in from the LayerNormPre
                        state_dict[f"blocks.{l}.mlp.W_out"] -= einops.reduce(
                            state_dict[f"blocks.{l}.mlp.W_out"],
                            "d_mlp d_model -> 1 d_model",
                            "mean",
                        )

                    del state_dict[f"blocks.{l}.mlp.ln.w"]

        # Fold ln_final into Unembed
        if not self.cfg.final_rms and fold_biases:
            # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm
            # pre unembed.
            state_dict[f"unembed.b_U"] = state_dict[f"unembed.b_U"] + (
                state_dict[f"unembed.W_U"] * state_dict[f"ln_final.b"][:, None]
            ).sum(dim=-2)
            del state_dict[f"ln_final.b"]

        state_dict[f"unembed.W_U"] = state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"][:, None]
        del state_dict[f"ln_final.w"]

        if center_weights:
            # Center the weights that read in from the LayerNormPre
            state_dict[f"unembed.W_U"] -= einops.reduce(
                state_dict[f"unembed.W_U"], "d_model d_vocab -> 1 d_vocab", "mean"
            )

        return state_dict

    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Center Writing Weights.

        Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and
        W_out. This is done by subtracting the mean of the weights from the weights themselves. This
        is done in-place. See fold_layer_norm for more details.
        """
        state_dict["embed.W_E"] = state_dict["embed.W_E"] - state_dict["embed.W_E"].mean(
            -1, keepdim=True
        )
        if self.cfg.positional_embedding_type != "rotary":
            state_dict["pos_embed.W_pos"] = state_dict["pos_embed.W_pos"] - state_dict[
                "pos_embed.W_pos"
            ].mean(-1, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f"blocks.{l}.attn.W_O"] = state_dict[f"blocks.{l}.attn.W_O"] - state_dict[
                f"blocks.{l}.attn.W_O"
            ].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[f"blocks.{l}.attn.b_O"] = (
                state_dict[f"blocks.{l}.attn.b_O"] - state_dict[f"blocks.{l}.attn.b_O"].mean()
            )  # b_O is [d_model]
            if not self.cfg.attn_only:
                state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[
                    f"blocks.{l}.mlp.W_out"
                ] - state_dict[f"blocks.{l}.mlp.W_out"].mean(-1, keepdim=True)
                state_dict[f"blocks.{l}.mlp.b_out"] = (
                    state_dict[f"blocks.{l}.mlp.b_out"] - state_dict[f"blocks.{l}.mlp.b_out"].mean()
                )
        return state_dict

    def center_unembed(self, state_dict: Dict[str, torch.Tensor]):
        """Center the unembedding weights W_U.

        This is done by subtracting the mean of the weights from the weights themselves. This is
        done in-place. As softmax is translation invariant, this changes the logits but not the log
        probs, and makes the model logits (slightly) more interpretable - when trying to understand
        how components contribute to the logits, we'll be less misled by components that just add
        something to every logit.
        """
        state_dict["unembed.W_U"] = state_dict["unembed.W_U"] - state_dict["unembed.W_U"].mean(
            -1, keepdim=True
        )
        state_dict["unembed.b_U"] = state_dict["unembed.b_U"] - state_dict["unembed.b_U"].mean()
        return state_dict

    def fold_value_biases(self, state_dict: Dict[str, torch.Tensor]):
        """Fold the value biases into the output bias.

        Because attention patterns add up to 1, the value biases always have a constant effect on a
        head's output. Further, as the outputs of each head in a layer add together, each head's
        value bias has a constant effect on the *layer's* output, which can make it harder to
        interpret the effect of any given head, and it doesn't matter which head a bias is
        associated with. We can factor this all into a single output bias to the layer, and make it
        easier to interpret the head's output. Formally, we take b_O_new = b_O_original +
        sum_head(b_V_head @ W_O_head).
        """
        for layer in range(self.cfg.n_layers):
            # shape [head_index, d_head]
            if self.cfg.n_key_value_heads is None:
                b_V = state_dict[f"blocks.{layer}.attn.b_V"]
            else:
                b_V = state_dict[f"blocks.{layer}.attn._b_V"]
                b_V = torch.repeat_interleave(
                    b_V, dim=0, repeats=self.cfg.n_heads // self.cfg.n_key_value_heads
                )
            # [head_index, d_head, d_model]
            W_O = state_dict[f"blocks.{layer}.attn.W_O"]
            # [d_model]
            b_O_original = state_dict[f"blocks.{layer}.attn.b_O"]
            folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])

            state_dict[f"blocks.{layer}.attn.b_O"] = folded_b_O
            if self.cfg.n_key_value_heads is None:
                state_dict[f"blocks.{layer}.attn.b_V"] = torch.zeros_like(b_V)
            else:
                state_dict[f"blocks.{layer}.attn._b_V"] = torch.zeros_like(
                    state_dict[f"blocks.{layer}.attn._b_V"]
                )
        return state_dict

    def refactor_factored_attn_matrices(self, state_dict: Dict[str, torch.Tensor]):
        """Experimental method for managing queries, keys and values.

        As argued in [A Mathematical Framework for Transformer
        Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and
        values are somewhat arbitrary intermediate terms when computing with the low rank factored
        matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing
        determining head behaviour. But there are many ways to find a low rank factorization to a
        given matrix, and hopefully some of these are more interpretable than others! This method is
        one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a
        rotation and W_Q and W_K having the nth column in each having the same norm. The formula is
        $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not
        R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank
        factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and
        $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now
        $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the
        result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$,
        which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the
        matrices as having the same norm, since there's not an obvious asymmetry between the keys
        and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V)
        @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that
        b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along
        the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q + b_Q) * (y @ W_K +
        b_K), which is fairly messy. To deal with the biases, we concatenate them to W_Q and W_K to
        simulate a d_model+1 dimensional input (whose final coordinate is always 1), do the SVD
        factorization on this effective matrix, then separate out into final weights and biases.
        """

        assert (
            self.cfg.positional_embedding_type != "rotary"
        ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

        for l in range(self.cfg.n_layers):
            # W_QK = W_Q @ W_K.T
            # Concatenate biases to make a d_model+1 input dimension
            W_Q_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    state_dict[f"blocks.{l}.attn.b_Q"][:, None, :],
                ],
                dim=1,
            )
            W_K_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_K"],
                    state_dict[f"blocks.{l}.attn.b_K"][:, None, :],
                ],
                dim=1,
            )

            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

            state_dict[f"blocks.{l}.attn.W_Q"] = W_Q_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_Q"] = W_Q_eff_even[:, -1, :]
            state_dict[f"blocks.{l}.attn.W_K"] = W_K_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_K"] = W_K_eff_even[:, -1, :]

            # W_OV = W_V @ W_O
            W_V = state_dict[f"blocks.{l}.attn.W_V"]
            W_O = state_dict[f"blocks.{l}.attn.W_O"]

            # Factors the bias to be consistent.
            b_V = state_dict[f"blocks.{l}.attn.b_V"]
            b_O = state_dict[f"blocks.{l}.attn.b_O"]
            effective_bias = b_O + einsum(
                "head_index d_head, head_index d_head d_model -> d_model", b_V, W_O
            )
            state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros_like(b_V)
            state_dict[f"blocks.{l}.attn.b_O"] = effective_bias

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[f"blocks.{l}.attn.W_V"] = U @ S.diag_embed()
            state_dict[f"blocks.{l}.attn.W_O"] = utils.transpose(Vh)

        return state_dict

    def set_use_attn_result(self, use_attn_result: bool):
        """Toggle whether to explicitly calculate and expose the result for each attention head.

        Useful for interpretability but can easily burn through GPU memory.
        """
        self.cfg.use_attn_result = use_attn_result

    def set_use_split_qkv_input(self, use_split_qkv_input: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        self.cfg.use_split_qkv_input = use_split_qkv_input

    def set_use_hook_mlp_in(self, use_hook_mlp_in: bool):
        """Toggles whether to allow storing and editing inputs to each MLP layer."""

        assert not self.cfg.attn_only, "Can't use hook_mlp_in with attn_only model"
        self.cfg.use_hook_mlp_in = use_hook_mlp_in

    def set_use_attn_in(self, use_attn_in: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        self.cfg.use_attn_in = use_attn_in

    def process_weights_(
        self,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ):
        """Wrapper around `load_and_process_state_dict`.

        Wrapper around load_and_process_state_dict to allow for in-place processing of the weights.
        This is useful if using HookedTransformer for training, if we then want to analyse a cleaner
        version of the same model.
        """
        state_dict = self.state_dict()
        if fold_ln and self.cfg.num_experts and self.cfg.num_experts > 1:
            # If we're using MoE, we don't fold the layer norm weights, so we don't need to do any preprocessing
            # A warning is already issued in `load_and_process_state_dict`
            pass
        elif fold_ln and self.cfg.normalization_type == "LN":
            # If we're folding the LN into the weights, we need to replace all the layernorm layers
            # with LayerNormPres, which do not have learnable parameters. This is somewhat hacky,
            # but it's the easiest way to do it.
            self.cfg.normalization_type = "LNPre"
            self.ln_final = LayerNormPre(self.cfg)
            for layer in self.blocks:
                layer.ln1 = LayerNormPre(self.cfg)
                layer.ln2 = LayerNormPre(self.cfg)
                if self.cfg.is_layer_norm_activation():
                    layer.mlp.ln = LayerNormPre(self.cfg)
        elif fold_ln and self.cfg.normalization_type == "RMS":
            # We do the same for RMSNorm if used
            self.cfg.normalization_type = "RMSPre"
            self.ln_final = RMSNormPre(self.cfg)
            for layer in self.blocks:
                layer.ln1 = RMSNormPre(self.cfg)
                layer.ln2 = RMSNormPre(self.cfg)
                if self.cfg.is_layer_norm_activation():
                    layer.mlp.ln = RMSNormPre(self.cfg)

        self.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )

    
    
    @torch.inference_mode()
    def generate(
        self,
        inputs: dict[str, torch.Tensor] = None,
        max_new_tokens: int = 100,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        return_type: Optional[str] = "tokens",
        verbose: bool = True,
        sae_hook_name: Optional[str] = None,
    ) -> Union[Int[torch.Tensor, "batch pos_plus_new_tokens"], str]:

        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):  
            vision = False  

            if isinstance(inputs, str):
                input_ids = inputs
                vision = False
            elif isinstance(inputs, dict):
                input_ids = inputs.get("input_ids", inputs)
                attention_mask = inputs.get("attention_mask", None)
                pixel_values = inputs.get("pixel_values", None)
                image_sizes = inputs.get("image_sizes", None)

                if pixel_values is not None:
                    vision = True
            else:
                input_ids = inputs

            if isinstance(input_ids, str):
                assert self.tokenizer is not None, "Most provide tokenizer"
                tokens = self.to_tokens(input_ids, prepend_bos=prepend_bos, padding_side=padding_side)
                return_type = "str" if return_type == "input" else "str"
            else:
                tokens = input_ids
                return_type = "tensor"  

            if vision:
                return_type = "tensor"

            assert isinstance(tokens, torch.Tensor)
            batch_size, ctx_length = tokens.shape
            device = devices.get_device_for_block_index(0, self.cfg)
            tokens = tokens.to(device)
            if use_past_kv_cache:
                past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                    self.cfg, self.cfg.device, batch_size
                )
            else:
                past_kv_cache = None

            stop_tokens: list[int] = []
            eos_token_for_padding = 0
            assert self.tokenizer is not None
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tokenizer is not None and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                    eos_token_id = self.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                    )

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)
            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.eval()
            
            if sae_hook_name is not None:
                image_indice=None
                tmp_cache_list=[]
            
            for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
                # While generating, we keep generating logits, throw away all but the final logits,
                # and then use those logits to sample from the distribution We keep adding the
                # sampled tokens to the end of tokens.
                if vision:
                    model_inputs=self.prepare_inputs_for_generation(tokens, past_kv_cache=past_kv_cache, image_sizes=image_sizes, attention_mask=attention_mask, pixel_values=pixel_values, vision=vision,use_kv_cache=use_past_kv_cache)
                    attention_mask=torch.cat([attention_mask,torch.ones((batch_size,1),device=device)],dim=1)
                
                if use_past_kv_cache:
                    # We just take the final tokens, as a [batch, 1] tensor
                    if vision:
                        if sae_hook_name is not None:
                            if index > 0:
                                (logits,image_indice),cache = self.run_with_cache(
                                    tokens[:, -1:],
                                    model_inputs=model_inputs,
                                    vision=vision,
                                    prepend_bos=prepend_bos,
                                    padding_side=padding_side,
                                    names_filter=lambda name: name == sae_hook_name,
                                    return_type="generate_with_saev",
                                )
                            else:
                                (logits,image_indice),cache = self.run_with_cache(
                                    tokens,
                                    model_inputs=model_inputs,
                                    vision=vision,
                                    prepend_bos=prepend_bos,
                                    padding_side=padding_side,
                                    names_filter=lambda name: name == sae_hook_name,
                                    return_type="generate_with_saev",
                                )
                            image_indice = image_indice
                            tmp_cache=cache[sae_hook_name]
                            tmp_cache=tmp_cache.to("cpu")
                            tmp_cache_list.append(tmp_cache)
                        elif return_type == "logits":
                            if index > 0:
                                logits = self.forward(
                                    tokens[:, -1:],
                                    return_type="logits",
                                    prepend_bos=prepend_bos,
                                    padding_side=padding_side,
                                    model_inputs=model_inputs,
                                    vision=vision,
                                )
                            else:
                                logits = self.forward(
                                    tokens,
                                    return_type="logits",
                                    prepend_bos=prepend_bos,
                                    padding_side=padding_side,
                                    model_inputs=model_inputs,
                                    vision=vision,
                                )
                    else:
                        if index > 0:
                            logits = self.forward(
                                tokens[:, -1:],
                                return_type="logits",
                                prepend_bos=prepend_bos,
                                padding_side=padding_side,
                                past_kv_cache=past_kv_cache,
                            )
                        else:
                            logits = self.forward(
                                tokens,
                                return_type="logits",
                                prepend_bos=prepend_bos,
                                padding_side=padding_side,
                                past_kv_cache=past_kv_cache,        
                            )
                else:
                    # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                    # the cache.
                    
                    # no cache no vision
                    logits = self.forward(
                        tokens,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                    )
                final_logits = logits[:, -1, :]

                if do_sample:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(devices.get_device_for_block_index(0, self.cfg))
                else:
                    sampled_tokens = final_logits.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.cfg)
                    )

                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.cfg.device),
                            torch.tensor(stop_tokens).to(self.cfg.device),
                        )
                    )

                tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

                if stop_at_eos and finished_sequences.all():
                    break

            if return_type == "str":
                if self.cfg.default_prepend_bos:
                    # If we prepended a BOS token, remove it when returning output.
                    return self.tokenizer.decode(tokens[0, 1:])
                else:
                    return self.tokenizer.decode(tokens[0])

            else:
                if sae_hook_name is not None:

                    return tokens,image_indice,tmp_cache_list
                return tokens

    # Give access to all weights as properties.
    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """Convenience to get the unembedding matrix.

        I.e. the linear map from the final residual stream to the output logits).
        """
        return self.unembed.W_U

    @property
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        return self.unembed.b_U

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """Convenience to get the embedding matrix."""
        return self.embed.W_E

    @property
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        """Convenience function to get the positional embedding.

        Only works on models with absolute positional embeddings!
        """
        return self.pos_embed.W_pos

    @property
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        """Concatenated W_E and W_pos.

        Used as a full (overcomplete) basis of the input space, useful for full QK and full OV
        circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=0)

    # Layer-specific weights are stacked into one massive tensor and given as properties for
    # convenience and a cache is used to avoid repeated computation. Often a useful convenience when
    # we want to do analysis on weights across all layers. If GPU memory is a bottleneck, don't use
    # these properties!

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the key weights across all layers."""
        return torch.stack([block.attn.W_K for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the query weights across all layers."""
        return torch.stack([block.attn.W_Q for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the value weights across all layers."""
        return torch.stack([block.attn.W_V for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stack the attn output weights across all layers."""
        return torch.stack([block.attn.W_O for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stack the MLP input weights across all layers."""
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    def W_gate(self) -> Union[Float[torch.Tensor, "n_layers d_model d_mlp"], None]:
        """Stack the MLP gate weights across all layers.

        Only works for models with gated MLPs.
        """
        if self.cfg.gated_mlp:
            return torch.stack([block.mlp.W_gate for block in self.blocks], dim=0)
        else:
            return None

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stack the MLP output weights across all layers."""
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the key biases across all layers."""
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the query biases across all layers."""
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the value biases across all layers."""
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stack the attn output biases across all layers."""
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stack the MLP input biases across all layers."""
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stack the MLP output biases across all layers."""
        return torch.stack([block.mlp.b_out for block in self.blocks], dim=0)

    @property
    def QK(self):
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self):
        return FactoredMatrix(self.W_V, self.W_O)

    # Various utility functions
    def accumulated_bias(
        self, layer: int, mlp_input: bool = False, include_mlp_biases=True
    ) -> Float[torch.Tensor, "d_model"]:
        """Accumulated Bias.

        Returns the accumulated bias from all layer outputs (ie the b_Os and b_outs), up to the
        input of layer L.

        Args:
            layer (int): Layer number, in [0, n_layers]. layer==0 means no layers, layer==n_layers
                means all layers.
            mlp_input (bool): If True, we take the bias up to the input of the MLP
                of layer L (ie we include the bias from the attention output of the current layer,
                otherwise just biases from previous layers)
            include_mlp_biases (bool): Whether to include the biases of MLP layers. Often useful to
                have as False if we're expanding attn_out into individual heads, but keeping mlp_out
                as is.

        Returns:
            bias (torch.Tensor): [d_model], accumulated bias
        """
        accumulated_bias = torch.zeros(self.cfg.d_model, device=self.cfg.device)

        for i in range(layer):
            accumulated_bias += self.blocks[i].attn.b_O
            if include_mlp_biases:
                accumulated_bias += self.blocks[i].mlp.b_out
        if mlp_input:
            assert layer < self.cfg.n_layers, "Cannot include attn_bias from beyond the final layer"
            accumulated_bias += self.blocks[layer].attn.b_O
        return accumulated_bias

    def all_composition_scores(
        self, mode
    ) -> Float[torch.Tensor, "n_layers n_heads n_layers n_heads"]:
        """All Composition Scores.

        Returns the Composition scores for all pairs of heads, as a L1, H1, L2, H2 tensor (which is
        upper triangular on the first and third axes).

        See
        https://transformer-circuits.pub/2021/framework/index.html#:~:text=The%20above%20diagram%20shows%20Q%2D%2C%20K%2D%2C%20and%20V%2DComposition
        for three metrics used.

        Args:
            mode (str): One of ["Q", "K", "V"], the mode to use for the composition score.
        """
        left = self.OV
        if mode == "Q":
            right = self.QK
        elif mode == "K":
            right = self.QK.T
        elif mode == "V":
            right = self.OV
        else:
            raise ValueError(f"mode must be one of ['Q', 'K', 'V'] not {mode}")

        scores = utils.composition_scores(left, right, broadcast_dims=True)
        # Mask scores to be zero for all pairs with the right head in the same layer or earlier
        # layer than the left head.
        mask = (
            torch.arange(self.cfg.n_layers, device=self.cfg.device)[:, None, None, None]
            < torch.arange(self.cfg.n_layers, device=self.cfg.device)[None, None, :, None]
        )
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        return scores

    def all_head_labels(self):
        """Returns a list of all head names in the model."""
        return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]

    def load_sample_training_dataset(self, **kwargs):
        """Load Sample Training Dataset.

        Helper function to load in a 10K-20K dataset of elements from the model's training data
        distribution.

        Wrapper around utils.get_dataset, which identifies the appropriate dataset the pretrained
        models. Each dataset has a 'text' field, which contains the relevant info, some have several
        meta data fields.

        Kwargs will be passed to utils.get_dataset (e.g. cache_dir to set download location)

        Notes:

        - PT-2's training data is not open source. OpenWebText is a replication (links with
            >3 karma on Reddit)
        - OPT's training data is not open source, and is a mess of different things that is hard to
          replicate. I default to the Pile, which covers some of it, but imperfectly.

        (Some models will have actually been trained on the data supplied here, for some it's from
        the validation set).
        """
        model_dataset_map = {
            "neel": "c4_code",
            "neel-solu-old": "pile",
            "GPT2LMHeadModel": "openwebtext",
            "GPTNeoForCausalLM": "pile",
            "GPTNeoXForCausalLM": "pile",
            "GPTJForCausalLM": "pile",
            "OPTForCausalLM": "pile",
        }
        if self.cfg.original_architecture in model_dataset_map:
            self.dataset = utils.get_dataset(
                model_dataset_map[self.cfg.original_architecture], **kwargs
            )
        else:
            raise ValueError(
                f"We do not have an available dataset for the relevant model: {self.cfg.original_architecture}"
            )
        return self.dataset

    def sample_datapoint(
        self,
        tokenize: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
    ) -> Union[str, Float[torch.Tensor, "1 pos"]]:
        """Sample Data Point from Dataset.

        Helper function to randomly sample a data point from self.dataset, a small dataset from the
        data distribution the model was trained on.

        Implicitly calls self.load_sample_training_dataset if it hasn't already been called. Only
        works for pretrained models with an associated dataset. But you can manually replace
        self.dataset with a dataset of your choice if you want.

        Args:
            tokenize (bool): Whether to return tokens (instead of text). Defaults to False. Note
                that the returned tokens will be automatically truncated to the model's max context
                size.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (applicable when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos (default is True unless specified
                otherwise). Pass True or False to override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
        """
        if self.dataset is None:
            self.load_sample_training_dataset()
        assert self.dataset is not None  # keep mypy happy
        sample_dataset_size = len(self.dataset)
        index = np.random.randint(0, sample_dataset_size)
        if not tokenize:
            return self.dataset[index]["text"]
        else:
            return self.to_tokens(
                self.dataset[index]["text"],
                prepend_bos=prepend_bos,
                padding_side=padding_side,
                truncate=True,
            )
