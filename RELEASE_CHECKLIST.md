# Release Checklist

## Must Do Before GitHub Release

- [ ] Replace cluster absolute paths with environment variables in every launch script.
- [x] Public method set fixed: `lora`, `pissa`, `milora`, `rlpo`, `rlmo`.
- [ ] Upload selected checkpoints to Hugging Face Hub or Git LFS and fill `checkpoints/README.md`.
- [ ] Upload or document adapter initialization bundles required by PiSSA/MiLoRA/RLPO/RLMO.
- [ ] Add dataset preparation instructions for `dapo-math-17k.parquet` and `aime-2024.parquet`.
- [ ] Add cleaned result tables under `results/` instead of raw cluster logs.

## Recommended Checkpoint Subset

- RLPO: seeds 41, 42, 43; include the best or final evaluated step.
- RLMO: seeds 41, 42, 43; include the best or final evaluated step.
- Baselines: LoRA, PiSSA, and MiLoRA final/best checkpoints used in the public comparison.

## Reproducibility Metadata

- [ ] Publish the patched `verl` branch or a `patches/` directory covering rollout/vLLM adapter synchronization.
- [ ] Pin verl commit hash and Python/CUDA/vLLM versions.
- [ ] Record GPU count, tensor parallel size, rollout settings, and batch size.
- [ ] Record random seeds and checkpoint selection rule.
- [ ] Keep raw logs private if they contain local paths; release cleaned CSV/JSON metrics.

