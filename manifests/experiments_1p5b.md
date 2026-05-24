# DAPO 1.5B Experiment Manifest

Only the public reproduction set is included: LoRA, PiSSA, MiLoRA, RLPO, and RLMO.

| method | script | project | default exp | base model | adapter | steps | rank/alpha | lr | train | eval |
|---|---|---|---|---|---|---:|---|---:|---|---|
| LoRA | `scripts/run_lora_1p5b_paper_seed.sh` | `LoRA-Baseline-Paper-Replication` | `lora_1p5b_paper_seed${SEED}_r16_a32_lr1e5` | `${MODEL_BASE_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}` | `${LORA_ADAPTER_PATH:-}` | 1024 | 16/32 | 1e-5 | `./data/dapo-math-17k.parquet` | `./data/aime-2024.parquet` |
| PiSSA | `scripts/run_pissa_1p5b_paper_seed.sh` | `PiSSA-Paper-Replication` | `pissa_1p5b_paper_seed${SEED}` | `./adapter_bundles/deepseek_r1_distill_qwen_1_5b/pissa_r32_a32/base` | `./adapter_bundles/deepseek_r1_distill_qwen_1_5b/pissa_r32_a32/adapter` | 1024 | 32/${LORA_RANK} | 2e-5 | `./data/dapo-math-17k.parquet` | `./data/aime-2024.parquet` |
| MiLoRA | `scripts/run_milora_1p5b_paper_seed.sh` | `MiLoRA-Paper-Replication` | `milora_1p5b_paper_seed${SEED}` | `./adapter_bundles/deepseek_r1_distill_qwen_1_5b/milora_r32_a32/base` | `./adapter_bundles/deepseek_r1_distill_qwen_1_5b/milora_r32_a32/adapter` | 1024 | 32/${LORA_RANK} | 2e-5 | `./data/dapo-math-17k.parquet` | `./data/aime-2024.parquet` |
| RLPO | `scripts/run_rlpo_1p5b_paper_seed.sh` | `RLPO-Paper-Replication` | `rlpo_1p5b_paper_seed${SEED}` | `${PRETRAINED}` | `${BUNDLE_BASE}/rlpo_r16_a32/adapter` | 1024 | 16/32 | 1e-5 | `./data/dapo-math-17k.parquet` | `./data/aime-2024.parquet` |
| RLMO | `scripts/run_rlmo_1p5b_paper_seed.sh` | `RLMO-Paper-Replication` | `rlmo_1p5b_paper_seed${SEED}` | `${PRETRAINED}` | `${BUNDLE_BASE}/rlmo_r16_a32/adapter` | 1024 | 16/32 | 1e-5 | `./data/dapo-math-17k.parquet` | `./data/aime-2024.parquet` |
