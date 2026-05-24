# Checkpoints

Full checkpoint directories are not committed to this Git repository. Public artifacts are hosted on Hugging Face Hub:

- https://huggingface.co/RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts

Cosine-decay LoRA adapter checkpoints for the main LoRA-family experiments are available under:

- https://huggingface.co/RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts/tree/main/checkpoints

## Layout

The Hugging Face `checkpoints/` directory is organized by method, seed, and training step:

```text
checkpoints/
  lora_cosine/
    seed41/global_step_50/
    ...
    seed43/global_step_500/
  rlmo_cosine/
    seed41/global_step_50/
    ...
    seed43/global_step_500/
  rlpo_cosine/
    seed41/global_step_50/
    ...
    seed43/global_step_500/
```

Each `global_step_*` directory contains the PEFT adapter files:

```text
adapter_config.json
adapter_model.safetensors
```

Available steps for each method/seed are:

```text
50, 100, 150, 200, 250, 300, 350, 400, 450, 500
```

## Download

Download all checkpoints with:

```bash
huggingface-cli download \
  RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts \
  --include "checkpoints/**" \
  --local-dir geometry-preserving-orthonormal-init-rlvr-artifacts
```

Or with Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts",
    allow_patterns="checkpoints/**",
    local_dir="geometry-preserving-orthonormal-init-rlvr-artifacts",
)
```
