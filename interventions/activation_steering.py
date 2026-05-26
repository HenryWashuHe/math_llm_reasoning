"""SAE-free residual-stream steering baseline.

Directly adds a fixed direction vector d ∈ R^{d_model} to the hidden stream
at a chosen layer. Used as a control for "does the SAE matter, or is any
non-trivial push in residual space enough to change behavior?"

Two load sources for the direction:
  --direction-path  a .npy file of shape (d_model,)
  --direction-from-sae  load W_dec[feature_id] from an SAE and use that —
                        lets you isolate the effect of skipping the
                        encode→decode round-trip while keeping the same
                        direction as a targeted feature.

Install signature mirrors sae_hook.install so the generation driver can
swap backends by changing intervention_spec.module.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch


def _default_layer_getter(model, layer: int):
    base = getattr(model, "model", model)
    return base.layers[layer]


class InterventionHandle:
    def __init__(self, torch_handle):
        self._handle = torch_handle
        self.recon_stats = None

    def remove(self) -> None:
        self._handle.remove()


def install(
    model,
    *,
    layer: int,
    direction: list[float],
    magnitude: float = 1.0,
    phase: str = "prompt_and_decode",
    layer_getter: Optional[Callable[[Any, int], Any]] = None,
) -> InterventionHandle:
    """Install a forward-hook that adds `magnitude * direction` to every
    hidden-state position at `layer`.

    `direction` is a flat list of length d_model. The hook normalizes it to
    unit length and scales by `magnitude` — this keeps the knob intuitive
    (magnitude ~= how many "units of direction" to inject per token).
    """
    if phase not in {"prompt_and_decode", "decode_only", "prefill_only"}:
        raise ValueError(f"unknown phase: {phase}")

    dev = next(model.parameters()).device
    d = torch.tensor(direction, device=dev, dtype=torch.float32)
    d = d / (d.norm() + 1e-10)
    delta_vec = (d * float(magnitude))

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return output
        is_prefill = hidden.shape[1] > 1
        if phase == "decode_only" and is_prefill:
            return output
        if phase == "prefill_only" and not is_prefill:
            return output
        v = delta_vec.to(hidden.dtype)
        updated = hidden + v
        return (updated,) + output[1:] if isinstance(output, tuple) else updated

    get_layer = layer_getter or _default_layer_getter
    torch_handle = get_layer(model, layer).register_forward_hook(hook)
    return InterventionHandle(torch_handle)
