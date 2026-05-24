# Installation

This guide describes how to prepare a machine to run the DAPO 1.5B experiments.

## 1. Clone This Repository

```bash
git clone <THIS_REPOSITORY_URL> geometry-preserving-orthonormal-init-rlvr
cd geometry-preserving-orthonormal-init-rlvr
```

## 2. Clone verl

Use the same `verl` revision used when the patch was exported:

```bash
git clone https://github.com/volcengine/verl.git ../verl
cd ../verl
git checkout f332fc814718b9ea7968f6d264211460d4e90fff
```

If your upstream remote differs, checkout the matching commit or manually port the patch.

## 3. Apply The Runtime Patch

```bash
git apply ../geometry-preserving-orthonormal-init-rlvr/patches/verl_rollout_adapter_sync.patch
```

The patch is required for vLLM rollout and LoRA/SVD-initialized adapter synchronization. See `docs/verl_vllm_rollout_notes.md` for details.

## 4. Create The Python Environment

Use a CUDA-enabled PyTorch environment compatible with your cluster. A typical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ../geometry-preserving-orthonormal-init-rlvr/requirements.txt
python -m pip install -e .
```

The pinned environment used Python 3.10.12, PyTorch 2.8.0 with CUDA 12.6, vLLM 0.11.0, and flash-attn 2.8.1. Depending on your CUDA driver and package index, install the matching PyTorch and flash-attn wheels separately before installing the rest of the requirements.

Optional: if you switch to reward functions that use Hugging Face Math-Verify, install `math-verify` separately. The DAPO reward path used here does not require it.

## 5. Prepare Data And Adapters

Return to this repository:

```bash
cd ../geometry-preserving-orthonormal-init-rlvr
```

Download the public parquet files:

```bash
bash scripts/prepare_data.sh
```

Generate initialization bundles from the base model. The two commands use different ranks because the released launch scripts use rank 32 for PiSSA/MiLoRA and rank 16 for RLPO/RLMO:

```bash
export MODEL_BASE_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export BUNDLE_BASE=adapter_bundles/deepseek_r1_distill_qwen_1_5b

python scripts/prepare_init_adapters.py \
  --model-name-or-path "$MODEL_BASE_PATH" \
  --output-dir "$BUNDLE_BASE" \
  --methods pissa milora \
  --rank 32 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --dtype bfloat16 \
  --device cuda

python scripts/prepare_init_adapters.py \
  --model-name-or-path "$MODEL_BASE_PATH" \
  --output-dir "$BUNDLE_BASE" \
  --methods rlpo rlmo \
  --rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --dtype bfloat16 \
  --device cuda
```

See `data/README.md` and `adapter_bundles/README.md` for override paths, prebuilt artifact downloads, and CPU or multi-GPU adapter generation.

## 6. Configure Paths

```bash
cd ../geometry-preserving-orthonormal-init-rlvr
cp env.example .env
# Edit .env for your local model, dataset, adapter, and checkpoint paths.
source .env
```

## 7. Run A Smoke Test

```bash
bash scripts/smoke_test.sh rlpo
```

By default, the smoke test checks files, imports, data schema, adapters, and shell syntax. Set `SMOKE_MODE=run` to execute a one-step training smoke test.

## 8. Run A Full Experiment

```bash
SEED=42 bash scripts/run_rlpo_1p5b_paper_seed.sh
SEED=42 bash scripts/run_rlmo_1p5b_paper_seed.sh
```

For all available methods, see `manifests/experiments_1p5b.md`.
