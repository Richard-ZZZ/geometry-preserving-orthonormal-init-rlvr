# Scripts

These are the DAPO 1.5B launch scripts copied from the working experiment tree.

## Method Mapping

| Public name | Local script | Notes |
|---|---|---|
| LoRA | `run_lora_1p5b_paper_seed.sh` | Standard LoRA baseline. |
| PiSSA | `run_pissa_1p5b_paper_seed.sh` | SVD-initialized baseline. |
| MiLoRA | `run_milora_1p5b_paper_seed.sh` | MiLoRA baseline. |
| RLPO | `run_rlpo_1p5b_paper_seed.sh` | Public method name. |
| RLMO | `run_rlmo_1p5b_paper_seed.sh` | Public method name. |

## Preparation Scripts

| Script | Purpose |
|---|---|
| `prepare_data.sh` | Downloads DAPO Math 17K and AIME 2024 parquet files and checks the `prompt` column. |
| `prepare_init_adapters.py` | Generates PiSSA, MiLoRA, RLPO, and RLMO initialization bundles from the base model. |
| `smoke_test.sh` | Checks imports, data schema, adapter paths, and shell syntax before full training. |

## Before Public Use

The scripts currently preserve original experiment defaults. For a clean public repo, make sure users can override:

- `MODEL_BASE_PATH` / `PRETRAINED`
- `TRAIN_FILE`
- `TEST_FILE`
- `BUNDLE_BASE` / `LORA_ADAPTER_PATH`
- `CKPT_ROOT`

Use `../env.example` as the public path contract.

