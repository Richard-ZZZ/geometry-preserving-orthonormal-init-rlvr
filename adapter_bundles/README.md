# Adapter Bundles

LoRA-RLPO and LoRA-RLMO are the two methods proposed in the paper. For a pretrained matrix $W_0=U\Sigma V^\top$, they use

$$
\text{LoRA-RLPO}:\quad B_0=0,\ A_0=V_r^\top,
\qquad
\text{LoRA-RLMO}:\quad B_0=0,\ A_0=V_{-r}^\top.
$$

Thus $A_0A_0^\top=I_r$ and $B_0A_0=0$. Both methods preserve the pretrained function without modifying the base model and do not apply singular-value scaling. See [`../docs/method.md`](../docs/method.md) for the exact method specification.

The reproduction scripts expect initialization adapters for PiSSA, MiLoRA, LoRA-RLPO, and LoRA-RLMO.

You do not need to upload prebuilt adapters if this repository includes the deterministic generation script and documents the exact base model, rank, alpha, dropout, target modules, and PEFT initialization settings. Uploading adapters is still useful for convenience and for byte-level artifact preservation.

Default root:

```text
adapter_bundles/deepseek_r1_distill_qwen_1_5b/
```

Expected layout:

```text
adapter_bundles/deepseek_r1_distill_qwen_1_5b/
  pissa_r32_a32/
    base/
    adapter/
  milora_r32_a32/
    base/
    adapter/
  rlpo_r16_a32/
    adapter/
  rlmo_r16_a32/
    adapter/
```

## Generate Locally

Use the same base model as the training scripts:

```bash
export MODEL_BASE_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export BUNDLE_BASE=adapter_bundles/deepseek_r1_distill_qwen_1_5b
```

Generate PiSSA and MiLoRA bundles at rank 32:

```bash
python scripts/prepare_init_adapters.py \
  --model-name-or-path "$MODEL_BASE_PATH" \
  --output-dir "$BUNDLE_BASE" \
  --methods pissa milora \
  --rank 32 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --dtype bfloat16 \
  --device cuda
```

Generate the LoRA-RLPO and LoRA-RLMO adapters at rank 16:

```bash
python scripts/prepare_init_adapters.py \
  --model-name-or-path "$MODEL_BASE_PATH" \
  --output-dir "$BUNDLE_BASE" \
  --methods rlpo rlmo \
  --rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --target-modules all-linear \
  --dtype bfloat16 \
  --device cuda
```

For multi-GPU or CPU-only initialization, use `--device-map auto` or `--device cpu`. Adapter generation performs SVD over target linear layers, so GPU generation is recommended.

## Method Usage

| Method | Required Path | Base Model Path |
|---|---|---|
| LoRA | No initialization adapter required by default. | Original pretrained model. |
| PiSSA | `pissa_r32_a32/adapter` | `pissa_r32_a32/base` |
| MiLoRA | `milora_r32_a32/adapter` | `milora_r32_a32/base` |
| LoRA-RLPO | `rlpo_r16_a32/adapter` | Original pretrained model. |
| LoRA-RLMO | `rlmo_r16_a32/adapter` | Original pretrained model. |

## Download Prebuilt Bundles

If you publish adapter artifacts separately, keep them outside git and download them into `BUNDLE_BASE`, for example:

```bash
huggingface-cli download <ORG_OR_USER>/<ADAPTER_REPO> \
  --local-dir adapter_bundles/deepseek_r1_distill_qwen_1_5b \
  --local-dir-use-symlinks False
```

Replace `<ORG_OR_USER>/<ADAPTER_REPO>` with your public artifact repository. Do not commit adapter weights directly unless you intentionally use Git LFS.

## Override Paths

You can keep adapters outside this repository and export:

```bash
export BUNDLE_BASE=/path/to/adapter_bundles/deepseek_r1_distill_qwen_1_5b
export LORA_ADAPTER_PATH=/path/to/specific/adapter
```

For PiSSA and MiLoRA scripts that use method-specific residual-base directories, also set:

```bash
export MODEL_BASE_PATH=/path/to/method/base
```
