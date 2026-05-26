"""SAE-based residual-stream intervention hook.

Single entry point `install(model, **kwargs)` matching the contract
`run_generations.py` expects. The caller loads the SAE object and passes it
in via kwargs — this module only owns the forward-hook mechanics.

Supported kinds
---------------
  ablate         encode → zero target feature rows → decode → apply delta
  steer          encode → add steer_values to target rows → decode → apply delta
  passthrough    encode → decode → replace hidden with reconstruction
                 (measures pure SAE noise floor; no feature modification)
  steer_direct   skip encode/decode; add W_dec[feature_ids]·steer_values to
                 hidden directly. Cleaner than `steer` because it avoids the
                 encode→decode round-trip contamination of all other features
                 (cf. Arad et al., EMNLP 2025).

Phase gating
------------
  prompt_and_decode   fire on prefill and decode steps (default)
  decode_only         only when hidden.shape[1] == 1 (one new token)
  prefill_only        only when hidden.shape[1]  > 1

Layer resolution
----------------
For Gemma-3 HuggingFace models the target module is `model.model.layers[layer]`
(the TransformerBlock). The hook reads/writes `output[0]` (the hidden stream).
Override via `layer_getter` if your model class differs.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch


# ---------------- module resolution ----------------

def _default_layer_getter(model, layer: int):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers[layer]
    raise AttributeError(f"cannot locate .layers on {type(model).__name__}")


# ---------------- hook factory ----------------

def _make_hook(
    *,
    sae,
    feature_ids: torch.Tensor | None,
    kind: str,
    phase: str,
    delta_scale: float,
    steer_values: torch.Tensor | None,
    recon_stats: list[dict[str, float]] | None,
) -> Callable:
    def hook(_module, _inputs, output):
        # skip if nothing to do
        if kind != "passthrough" and (feature_ids is None or feature_ids.numel() == 0):
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return output

        is_prefill = hidden.shape[1] > 1
        if phase == "decode_only" and is_prefill:
            return output
        if phase == "prefill_only" and not is_prefill:
            return output

        hidden_fp32 = hidden.to(torch.float32)

        # Diagnostics: record SAE recon error every hook firing (unless bypassed).
        if recon_stats is not None and kind != "steer_direct":
            with torch.no_grad():
                feats = sae.encode(hidden_fp32)
                recon = sae.decode(feats)
                diff = recon - hidden_fp32
                mse = float((diff * diff).mean().item())
                norm_sq = float((hidden_fp32 * hidden_fp32).mean().item())
                recon_stats.append({
                    "mse": mse,
                    "nmse": mse / max(norm_sq, 1e-10),
                    "n_tokens": int(hidden.shape[1]),
                    "is_prefill": bool(is_prefill),
                })

        # Direct decoder steering: bypass encode/decode round-trip.
        if kind == "steer_direct":
            if steer_values is None:
                raise ValueError("steer_direct requires steer_values")
            W_dec = sae.W_dec[feature_ids].to(torch.float32)  # (k, d_model)
            sv = steer_values.to(W_dec.device, dtype=torch.float32)  # (k,)
            delta = (W_dec * sv.unsqueeze(-1)).sum(0)  # (d_model,)
            delta = (delta * float(delta_scale)).to(hidden.dtype)
            updated = hidden + delta
            return (updated,) + output[1:] if isinstance(output, tuple) else updated

        # encode → modify → decode → delta
        feats = sae.encode(hidden_fp32)
        modified = feats.clone()
        if kind == "ablate":
            modified[..., feature_ids] = 0.0
        elif kind == "steer":
            if steer_values is None:
                raise ValueError("steer requires steer_values")
            sv = steer_values.to(modified.device, dtype=modified.dtype)
            modified[..., feature_ids] = modified[..., feature_ids] + sv
        elif kind == "passthrough":
            pass
        else:
            raise ValueError(f"unsupported kind: {kind}")

        original_recon = sae.decode(feats)
        if kind == "passthrough":
            updated = original_recon.to(hidden.dtype)
        else:
            modified_recon = sae.decode(modified)
            delta = ((modified_recon - original_recon) * float(delta_scale)).to(hidden.dtype)
            updated = hidden + delta

        return (updated,) + output[1:] if isinstance(output, tuple) else updated

    return hook


# ---------------- public API ----------------

class InterventionHandle:
    """Returned by install(); .remove() tears down the hook."""
    def __init__(self, torch_handle, recon_stats: list[dict[str, float]] | None):
        self._handle = torch_handle
        self.recon_stats = recon_stats

    def remove(self) -> None:
        self._handle.remove()


def install(
    model,
    *,
    sae,
    layer: int,
    feature_ids: list[int] | None = None,
    kind: str = "ablate",
    phase: str = "prompt_and_decode",
    delta_scale: float = 1.0,
    steer_values: list[float] | None = None,
    record_recon: bool = True,
    layer_getter: Optional[Callable[[Any, int], Any]] = None,
) -> InterventionHandle:
    """Install a forward-hook on model layer `layer`. Returns InterventionHandle.

    Pass `sae` as an already-loaded module exposing `.encode(h)`, `.decode(f)`,
    `.W_dec` (feature_count × d_model), and `.device`.
    """
    if kind not in {"ablate", "steer", "steer_direct", "passthrough"}:
        raise ValueError(f"unknown kind: {kind}")
    if phase not in {"prompt_and_decode", "decode_only", "prefill_only"}:
        raise ValueError(f"unknown phase: {phase}")

    feat_t = None
    if feature_ids:
        feat_t = torch.tensor(feature_ids, device=sae.device, dtype=torch.long)
    steer_t = None
    if steer_values is not None:
        steer_t = torch.tensor(steer_values, device=sae.device, dtype=torch.float32)

    recon_stats: list[dict[str, float]] | None = [] if record_recon else None

    hook = _make_hook(
        sae=sae,
        feature_ids=feat_t,
        kind=kind,
        phase=phase,
        delta_scale=delta_scale,
        steer_values=steer_t,
        recon_stats=recon_stats,
    )

    get_layer = layer_getter or _default_layer_getter
    module = get_layer(model, layer)
    torch_handle = module.register_forward_hook(hook)
    return InterventionHandle(torch_handle, recon_stats)
