# Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR

This repository contains the code, scripts, and manifests for the DAPO 1.5B RLVR experiments in Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR.

## Scope

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Training data: DAPO math training parquet (`dapo-math-17k.parquet` in the original run)
- Validation data: AIME 2024 parquet (`aime-2024.parquet` in the original run)
- Main methods: LoRA, PiSSA, MiLoRA, RLPO, and RLMO.

## Repository Layout

- `scripts/`: DAPO 1.5B launch scripts, data downloader, adapter generator, smoke test, and reward function.
- `configs/`: DAPO trainer config used by the recipe.
- `manifests/experiments_1p5b.md`: method-to-script and hyperparameter map.
- `manifests/checkpoints_1p5b.md`: local checkpoint inventory and intended upload targets.
- `results/`: place cleaned metrics tables and figures here.
- `checkpoints/`: do not commit large files; use this for pointer files or Hugging Face links.

## Installation

Follow `INSTALL.md` to apply the required `verl` runtime patch, install dependencies, download data, generate adapters, and run a smoke test.

## Quick Start

Install patched `verl`, then prepare local data and adapters:

```bash
bash scripts/prepare_data.sh
export MODEL_BASE_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export BUNDLE_BASE=adapter_bundles/deepseek_r1_distill_qwen_1_5b
python scripts/prepare_init_adapters.py --model-name-or-path "$MODEL_BASE_PATH" --output-dir "$BUNDLE_BASE" --methods pissa milora --rank 32 --lora-alpha 32
python scripts/prepare_init_adapters.py --model-name-or-path "$MODEL_BASE_PATH" --output-dir "$BUNDLE_BASE" --methods rlpo rlmo --rank 16 --lora-alpha 32
```

Set paths explicitly before running if you keep artifacts elsewhere:

```bash
export MODEL_BASE_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B
export TRAIN_FILE=/path/to/dapo-math-17k.parquet
export TEST_FILE=/path/to/aime-2024.parquet
export BUNDLE_BASE=/path/to/adapter_bundles/deepseek_r1_distill_qwen_1_5b
export CKPT_ROOT=/path/to/save/checkpoints
```

Example:

```bash
# RLPO
SEED=42 bash scripts/run_rlpo_1p5b_paper_seed.sh

# RLMO
SEED=42 bash scripts/run_rlmo_1p5b_paper_seed.sh
```

## Patched verl Runtime

These experiments require a patched `verl`/vLLM rollout runtime for LoRA and SVD-initialized adapters. See `docs/verl_vllm_rollout_notes.md` before trying to reproduce training from a stock `verl` checkout.

## Important Reproducibility Notes

The launch scripts in `scripts/` use public defaults or environment-variable overrides for model, dataset, adapter, and checkpoint paths. Use `env.example` as the path contract for new machines.

`manifests/checkpoints_1p5b.md` intentionally records the current local checkpoint inventory so you can decide which artifacts to upload; replace those local paths with public Hugging Face Hub or Git LFS links before release.

## Checkpoints

Large checkpoints should be released outside git, e.g. Hugging Face Hub or Git LFS. Keep this repository lightweight and point to uploaded artifacts from `checkpoints/README.md`.
