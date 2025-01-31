"""Devices.

Utilities to get the correct device, and assist in distributing model layers across multiple
devices.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

import transformer_lens


def get_device_for_block_index(
    index: int,
    cfg: "transformer_lens.HookedTransformerConfig",
    device: Optional[Union[torch.device, str]] = None,
):
    """
    Determine the device for a given layer index based on the model configuration.

    This function assists in distributing model layers across multiple devices. The distribution
    is based on the configuration's number of layers (cfg.n_layers) and devices (cfg.n_devices).

    Args:
        index (int): Model layer index.
        cfg (HookedTransformerConfig): Model and device configuration.
        device (Optional[Union[torch.device, str]], optional): Initial device used for determining the target device.
            If not provided, the function uses the device specified in the configuration (cfg.device).

    Returns:
        torch.device: The device for the specified layer index.
    """
    assert cfg.device is not None

    layers_per_device = cfg.n_layers // (cfg.n_devices)
    if device is None:
        device = cfg.device
    device = torch.device(device)
    if device.type == "cpu":
        return device
    device_index = (device.index or 0) + (index // layers_per_device)

    return torch.device(device.type, device_index)


def move_to_and_update_config(
    model: Union[
        "transformer_lens.HookedTransformer",
        "transformer_lens.HookedEncoder",
        "transformer_lens.HookedEncoderDecoder",
    ],
    device_or_dtype: Union[torch.device, str, torch.dtype],
    print_details=True,
):
    """
    Wrapper around `to` that also updates `model.cfg`.
    """
    if isinstance(device_or_dtype, torch.device):
        model.cfg.device = device_or_dtype.type
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, str):
        model.cfg.device = device_or_dtype
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, torch.dtype):
        model.cfg.dtype = device_or_dtype
        if print_details:
            print("Changing model dtype to", device_or_dtype)
        # change state_dict dtypes
        for k, v in model.state_dict().items():
            model.state_dict()[k] = v.to(device_or_dtype)
    return nn.Module.to(model, device_or_dtype)

def compute_block_device_mapping(cfg):

    assert cfg.device is not None
    n_layers = cfg.n_layers
    n_devices = cfg.n_devices
    device=torch.device(cfg.device)
    device_index = device.index or 0  
    n_other_devices = n_devices - 1
    total_weight_units = 1 + n_other_devices * 2 
    layers_per_unit = n_layers / total_weight_units

    layers_first_device = int(layers_per_unit * 1)

    layers_per_other_device = int(layers_per_unit * 2)

    total_assigned_layers = layers_first_device + layers_per_other_device * n_other_devices
    remaining_layers = n_layers - total_assigned_layers

    block_device_map = []

    first_device = torch.device(device.type, int(device_index))
    block_device_map.extend([first_device] * layers_first_device)

    for i in range(n_other_devices):
        device_id = device_index + i + 1 
        device = torch.device(device.type, device_id)
        device_layers = layers_per_other_device

        if remaining_layers > 0:
            device_layers += 1
            remaining_layers -= 1

        block_device_map.extend([device] * device_layers)

    return block_device_map