# Checkpoints

Full checkpoint directories are not committed to git. The released LoRA adapter checkpoints are hosted on Hugging Face Hub:

https://huggingface.co/RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts

## Uploaded Cosine Runs

Each directory contains PEFT adapter files (`adapter_config.json` and `adapter_model.safetensors`).

| Method | Seeds | Steps | Hugging Face path |
|---|---|---|---|
| LoRA | 41, 42, 43 | 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 | `https://huggingface.co/RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts/tree/main/checkpoints/LoRA` |
| RLPO | 41, 42, 43 | 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 | `https://huggingface.co/RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts/tree/main/checkpoints/RLPO` |
| RLMO | 41, 42, 43 | 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 | `https://huggingface.co/RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts/tree/main/checkpoints/RLMO` |

Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

Use `huggingface-cli download` to fetch a specific method, seed, and step, for example:

```bash
huggingface-cli download RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts \
  --include "checkpoints/RLPO/seed42/global_step_500/*" \
  --local-dir ./artifacts
```

See `../manifests/checkpoints_1p5b.md` for the experiment inventory.
