#!/usr/bin/env python3
"""
Prepare initialization adapter bundles for DAPO 1.5B reproduction.

The public training scripts use four generated adapter families:

- PiSSA and MiLoRA save both a compensated base model and an adapter.
- RLPO and RLMO save only an adapter; the base model remains the original
  pretrained model.

Example:
  python scripts/prepare_init_adapters.py \
    --model-name-or-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir adapter_bundles/deepseek_r1_distill_qwen_1_5b \
    --methods pissa milora --rank 32 --lora-alpha 32
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]

PUBLIC_METHODS = ("pissa", "milora", "rlpo", "rlmo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DAPO 1.5B initialization adapter bundles.")
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Base Hugging Face model path or name.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pissa", "milora"],
        choices=PUBLIC_METHODS,
        help="Methods to prepare. Use separate runs when methods use different ranks.",
    )
    parser.add_argument("--output-dir", required=True, help="Output root for generated bundles.")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--target-modules",
        type=str,
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated modules, JSON list, or all-linear.",
    )
    parser.add_argument(
        "--pissa-init",
        type=str,
        default="pissa_niter_4",
        help="PiSSA init string for PEFT, for example pissa or pissa_niter_4.",
    )
    parser.add_argument(
        "--milora-mode",
        type=str,
        default="min",
        choices=["min", "mid", "max", "random"],
        help="MiLoRA singular spectrum selection mode.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for initialization, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Optional transformers device_map, for example auto.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loading.")
    return parser.parse_args()


def parse_target_modules(value: str) -> str | list[str]:
    value = value.strip()
    if value == "all-linear":
        return "all-linear"
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("--target-modules JSON must be a list of strings.")
        return parsed
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--target-modules cannot be empty.")
    return modules


def matches_target(name: str, target_modules: str | list[str]) -> bool:
    if target_modules == "all-linear":
        return True
    return any(target in name for target in target_modules)


def to_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def select_svd_components(weights: torch.Tensor, rank: int, mode: str):
    u, singular_values, vh = torch.linalg.svd(weights, full_matrices=False)
    max_rank = min(weights.shape[0], weights.shape[1])
    if rank > max_rank:
        raise ValueError(f"rank={rank} exceeds max rank {max_rank} for weight shape {tuple(weights.shape)}")

    if mode == "min":
        indices = torch.arange(len(singular_values) - rank, len(singular_values), device=weights.device)
    elif mode == "max":
        indices = torch.arange(rank, device=weights.device)
    elif mode == "mid":
        start = (len(singular_values) - rank) // 2
        indices = torch.arange(start, start + rank, device=weights.device)
    elif mode == "random":
        indices = torch.randperm(len(singular_values), device=weights.device)[:rank]
        indices = torch.sort(indices).values
    else:
        raise ValueError(f"Unknown SVD selection mode: {mode}")

    return u[:, indices], singular_values[indices], vh[indices, :]


def initialize_milora_layer(weights: torch.Tensor, rank: int, mode: str):
    u_sel, singular_values, vh_sel = select_svd_components(weights, rank=rank, mode=mode)
    sqrt_s = torch.sqrt(singular_values)
    lora_b = u_sel @ torch.diag(sqrt_s)
    lora_a = torch.diag(sqrt_s) @ vh_sel
    delta = lora_b @ lora_a
    return lora_a, lora_b, delta


def initialize_orthogonal_zero_b_layer(weights: torch.Tensor, rank: int, mode: str):
    _, _, vh_sel = select_svd_components(weights, rank=rank, mode=mode)
    lora_a = vh_sel.contiguous()
    lora_b = torch.zeros(weights.shape[0], rank, device=weights.device, dtype=weights.dtype)
    return lora_a, lora_b


def get_effective_lora_scale(module: torch.nn.Module, adapter_name: str) -> float:
    scaling = getattr(module, "scaling", None)
    if isinstance(scaling, dict):
        return float(scaling[adapter_name])
    return float(scaling)


def apply_milora_init(peft_model: torch.nn.Module, target_modules: str | list[str], rank: int, mode: str) -> int:
    updated = 0
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not (hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if not isinstance(module.base_layer, torch.nn.Linear):
                continue
            if not matches_target(name, target_modules):
                continue

            adapter_name = next(iter(module.lora_A.keys()))
            base_weight = module.base_layer.weight.data
            lora_a, lora_b, delta = initialize_milora_layer(base_weight.float(), rank=rank, mode=mode)
            scale = get_effective_lora_scale(module, adapter_name)
            module.base_layer.weight.data -= (scale * delta).to(base_weight.device, base_weight.dtype)
            module.lora_A[adapter_name].weight.data = lora_a.to(base_weight.device, base_weight.dtype).contiguous()
            module.lora_B[adapter_name].weight.data = lora_b.to(base_weight.device, base_weight.dtype).contiguous()
            updated += 1
    return updated


def apply_orthogonal_zero_b_init(peft_model: torch.nn.Module, target_modules: str | list[str], rank: int, mode: str):
    updated = 0
    timings = []
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not (hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if not isinstance(module.base_layer, torch.nn.Linear):
                continue
            if not matches_target(name, target_modules):
                continue

            adapter_name = next(iter(module.lora_A.keys()))
            base_weight = module.base_layer.weight.data
            start = time.perf_counter()
            lora_a, lora_b = initialize_orthogonal_zero_b_layer(base_weight.float(), rank=rank, mode=mode)
            timings.append(time.perf_counter() - start)
            module.lora_A[adapter_name].weight.data = lora_a.to(base_weight.device, base_weight.dtype).contiguous()
            module.lora_B[adapter_name].weight.data = lora_b.to(base_weight.device, base_weight.dtype).contiguous()
            updated += 1

    timing = {
        "layers": len(timings),
        "total_s": float(sum(timings)),
        "mean_s": float(sum(timings) / len(timings)) if timings else 0.0,
        "max_s": float(max(timings)) if timings else 0.0,
    }
    return updated, timing


def load_model(model_name_or_path: str, dtype: torch.dtype, device: str, device_map: str | None, trust_remote_code: bool):
    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if device_map:
        load_kwargs["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, **load_kwargs)
    if not device_map:
        model.to(device)
    return model


def build_lora_model(
    method: str,
    model_name_or_path: str,
    dtype: torch.dtype,
    device: str,
    device_map: str | None,
    trust_remote_code: bool,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str | list[str],
    pissa_init: str,
    milora_mode: str,
):
    model = load_model(model_name_or_path, dtype, device, device_map, trust_remote_code)
    lora_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    if method == "pissa":
        lora_kwargs["init_lora_weights"] = pissa_init
        peft_model = get_peft_model(model, LoraConfig(**lora_kwargs))
        stats = {"updated_modules": "PEFT-internal", "method": method}
    elif method == "milora":
        peft_model = get_peft_model(model, LoraConfig(**lora_kwargs))
        updated = apply_milora_init(peft_model, target_modules, rank=rank, mode=milora_mode)
        stats = {"updated_modules": updated, "method": method, "mode": milora_mode}
    elif method in ("rlpo", "rlmo"):
        orthogonal_mode = "max" if method == "rlpo" else "min"
        peft_model = get_peft_model(model, LoraConfig(**lora_kwargs))
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        updated, timing = apply_orthogonal_zero_b_init(peft_model, target_modules, rank=rank, mode=orthogonal_mode)
        timing["wall_time_s"] = time.perf_counter() - start
        if torch.cuda.is_available() and "cuda" in device:
            timing["cuda_peak_mem_bytes"] = torch.cuda.max_memory_allocated()
        stats = {"updated_modules": updated, "method": method, "mode": orthogonal_mode, "delta": 0, "svd_timing": timing}
    else:
        raise ValueError(f"Unsupported method: {method}")

    return peft_model, stats


def save_bundle(
    peft_model: torch.nn.Module,
    tokenizer,
    output_root: Path,
    method: str,
    rank: int,
    alpha: int,
    metadata: dict[str, Any],
) -> tuple[Path, Path | None]:
    method_dir = output_root / f"{method}_r{rank}_a{alpha}"
    adapter_dir = method_dir / "adapter"
    method_dir.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(str(adapter_dir), safe_serialization=True)

    if method == "pissa":
        adapter_config_path = adapter_dir / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        if adapter_config.get("init_lora_weights") is not True:
            adapter_config["init_lora_weights"] = True
            adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")

    if method in ("rlpo", "rlmo"):
        base_dir = None
    else:
        base_dir = method_dir / "base"
        base_model = peft_model.base_model.unload()
        base_model.save_pretrained(str(base_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(base_dir))

    (method_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return adapter_dir, base_dir


def notes_for_method(method: str) -> list[str]:
    if method in ("rlpo", "rlmo"):
        return [
            f"{method.upper()}: orthonormal SVD directions with zero B; delta=0.",
            "Use the original pretrained model as actor_rollout_ref.model.path.",
            "Use this bundle adapter as actor_rollout_ref.model.lora_adapter_path.",
            "Do not enable lora_residual_base for this method.",
        ]
    return [
        f"{method}: SVD initialization modifies the base weights.",
        "Use base and adapter from the same bundle for reproduction.",
        "Set actor_rollout_ref.model.path to the bundle base directory.",
        "Set actor_rollout_ref.model.lora_adapter_path to the bundle adapter directory.",
    ]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    target_modules = parse_target_modules(args.target_modules)
    dtype = to_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)

    print(f"[INFO] Preparing methods={args.methods} for model={args.model_name_or_path}")
    print(f"[INFO] target_modules={target_modules}")
    print(f"[INFO] output_dir={output_root}")

    for method in args.methods:
        print(f"[INFO] Building {method}...")
        peft_model, stats = build_lora_model(
            method=method,
            model_name_or_path=args.model_name_or_path,
            dtype=dtype,
            device=args.device,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            pissa_init=args.pissa_init,
            milora_mode=args.milora_mode,
        )

        metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "method": method,
            "model_name_or_path": args.model_name_or_path,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_modules,
            "pissa_init": args.pissa_init,
            "milora_mode": args.milora_mode,
            "dtype": args.dtype,
            "device": args.device,
            "device_map": args.device_map,
            "notes": notes_for_method(method),
            "stats": stats,
        }

        timing = stats.get("svd_timing")
        if timing:
            peak = timing.get("cuda_peak_mem_bytes")
            peak_str = f"{peak / 1024 / 1024 / 1024:.2f} GiB" if peak is not None else "n/a"
            print(
                "[SVD] "
                f"method={method} layers={timing.get('layers', 0)} "
                f"svd_total_s={timing.get('total_s', 0.0):.3f} "
                f"wall_time_s={timing.get('wall_time_s', 0.0):.3f} "
                f"mean_s={timing.get('mean_s', 0.0):.6f} "
                f"max_s={timing.get('max_s', 0.0):.6f} "
                f"cuda_peak_mem={peak_str}"
            )

        adapter_dir, base_dir = save_bundle(
            peft_model=peft_model,
            tokenizer=tokenizer,
            output_root=output_root,
            method=method,
            rank=args.rank,
            alpha=args.lora_alpha,
            metadata=metadata,
        )
        print(f"[INFO] {method} adapter saved: {adapter_dir}")
        if base_dir is not None:
            print(f"[INFO] {method} compensated base saved: {base_dir}")
        else:
            print(f"[INFO] {method}: no base saved; use original pretrained model directly")

        del peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
