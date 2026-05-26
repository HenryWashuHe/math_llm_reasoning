"""Materialize intervention-spec JSON files consumed by run_generations.py.

Usage pattern: one JSON per arm; then loop a shell driver over
  --intervention-spec specs/{arm}.json

Arms emitted
------------
  none                   no hook installed (baseline)
  passthrough            SAE encode→decode only, no feature change
  ablate_features        zero out the listed features
  steer_features         add steer_values to the listed features (encode/decode round-trip)
  steer_direct           W_dec[features]·steer — skip round-trip
  random_k_features      ablate k random features (seeded); control for ablate_features
  act_steer_no_sae       SAE-free residual push along a direction vector (control)

Every JSON has schema {"module": "<qualified.path>", "args": {...}} so
run_generations.py can importlib it dynamically.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SAE_HOOK = "mathreason_sae.interventions.sae_hook"
ACT_STEER = "mathreason_sae.interventions.activation_steering"


def spec_none() -> dict[str, Any] | None:
    return None  # caller omits --intervention-spec


def spec_passthrough(*, layer: int, delta_scale: float = 1.0) -> dict[str, Any]:
    return {
        "module": SAE_HOOK,
        "args": {
            "layer": layer,
            "kind": "passthrough",
            "delta_scale": delta_scale,
        },
    }


def spec_ablate(*, layer: int, feature_ids: list[int],
                delta_scale: float = 1.0, phase: str = "prompt_and_decode") -> dict[str, Any]:
    return {
        "module": SAE_HOOK,
        "args": {
            "layer": layer,
            "kind": "ablate",
            "feature_ids": feature_ids,
            "delta_scale": delta_scale,
            "phase": phase,
        },
    }


def spec_steer(*, layer: int, feature_ids: list[int],
               steer_values: list[float], delta_scale: float = 1.0,
               phase: str = "prompt_and_decode") -> dict[str, Any]:
    return {
        "module": SAE_HOOK,
        "args": {
            "layer": layer,
            "kind": "steer",
            "feature_ids": feature_ids,
            "steer_values": steer_values,
            "delta_scale": delta_scale,
            "phase": phase,
        },
    }


def spec_steer_direct(*, layer: int, feature_ids: list[int],
                      steer_values: list[float], delta_scale: float = 1.0,
                      phase: str = "prompt_and_decode") -> dict[str, Any]:
    return {
        "module": SAE_HOOK,
        "args": {
            "layer": layer,
            "kind": "steer_direct",
            "feature_ids": feature_ids,
            "steer_values": steer_values,
            "delta_scale": delta_scale,
            "phase": phase,
        },
    }


def spec_random_k_features(*, layer: int, k: int, n_features: int,
                           seed: int, delta_scale: float = 1.0,
                           phase: str = "prompt_and_decode") -> dict[str, Any]:
    rng = random.Random(seed)
    fids = rng.sample(range(n_features), k)
    return {
        "module": SAE_HOOK,
        "args": {
            "layer": layer,
            "kind": "ablate",
            "feature_ids": fids,
            "delta_scale": delta_scale,
            "phase": phase,
            "_seed": seed,  # logged-only; ignored by install()
        },
    }


def spec_act_steer_no_sae(*, layer: int, direction: list[float], magnitude: float = 1.0,
                          phase: str = "prompt_and_decode") -> dict[str, Any]:
    return {
        "module": ACT_STEER,
        "args": {
            "layer": layer,
            "direction": direction,
            "magnitude": magnitude,
            "phase": phase,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit intervention-spec JSONs.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-features", type=int, default=65536)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--phase", default="prompt_and_decode")
    ap.add_argument("--features", type=int, nargs="+", required=True,
                    help="feature IDs for ablate/steer arms")
    ap.add_argument("--steer-values", type=float, nargs="+", default=None,
                    help="per-feature steer magnitudes; required for steer arms")
    ap.add_argument("--random-seeds", type=int, nargs="+", default=[7, 13, 42])
    ap.add_argument("--skip", nargs="+", default=[],
                    help="arm names to omit (e.g. --skip steer steer_direct)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    k = len(args.features)

    specs: dict[str, dict[str, Any] | None] = {}
    specs["passthrough"] = spec_passthrough(layer=args.layer, delta_scale=args.delta_scale)
    specs["ablate"] = spec_ablate(
        layer=args.layer, feature_ids=list(args.features),
        delta_scale=args.delta_scale, phase=args.phase,
    )
    if args.steer_values is not None:
        specs["steer"] = spec_steer(
            layer=args.layer, feature_ids=list(args.features),
            steer_values=list(args.steer_values),
            delta_scale=args.delta_scale, phase=args.phase,
        )
        specs["steer_direct"] = spec_steer_direct(
            layer=args.layer, feature_ids=list(args.features),
            steer_values=list(args.steer_values),
            delta_scale=args.delta_scale, phase=args.phase,
        )
    for seed in args.random_seeds:
        specs[f"random_k{k}_seed{seed}"] = spec_random_k_features(
            layer=args.layer, k=k, n_features=args.n_features, seed=seed,
            delta_scale=args.delta_scale, phase=args.phase,
        )

    for name, spec in specs.items():
        if spec is None or name in args.skip:
            continue
        out = args.out_dir / f"{name}.json"
        out.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
