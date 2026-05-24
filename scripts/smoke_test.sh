#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:-rlpo}
MODE=${SMOKE_MODE:-check}

case "${METHOD}" in
  lora)
    RUN_SCRIPT="scripts/run_lora_1p5b_paper_seed.sh"
    ;;
  pissa)
    RUN_SCRIPT="scripts/run_pissa_1p5b_paper_seed.sh"
    ;;
  milora)
    RUN_SCRIPT="scripts/run_milora_1p5b_paper_seed.sh"
    ;;
  rlpo)
    RUN_SCRIPT="scripts/run_rlpo_1p5b_paper_seed.sh"
    ;;
  rlmo)
    RUN_SCRIPT="scripts/run_rlmo_1p5b_paper_seed.sh"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    echo "Usage: bash scripts/smoke_test.sh [lora|pissa|milora|rlpo|rlmo]" >&2
    exit 2
    ;;
esac

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

TRAIN_FILE=${TRAIN_FILE:-./data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-./data/aime-2024.parquet}
USER_MODEL_BASE_PATH=${MODEL_BASE_PATH:-}
DEFAULT_PRETRAINED_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
BUNDLE_BASE=${BUNDLE_BASE:-./adapter_bundles/deepseek_r1_distill_qwen_1_5b}
CKPT_ROOT=${CKPT_ROOT:-./checkpoints}

require_file() {
  local path=$1
  local label=$2
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    return 1
  fi
}

require_dir() {
  local path=$1
  local label=$2
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    return 1
  fi
}

check_python_imports() {
  python - <<'PY'
import importlib
modules = ["torch", "transformers", "peft", "vllm", "verl", "pandas", "pyarrow"]
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, repr(exc)))
if missing:
    for name, err in missing:
        print(f"Missing or broken import: {name}: {err}")
    raise SystemExit(1)
print("Python import check passed")
PY
}

check_data_schema() {
  python - <<PY
import pandas as pd
for path in ["${TRAIN_FILE}", "${TEST_FILE}"]:
    df = pd.read_parquet(path)
    if len(df) == 0:
        raise SystemExit(f"Empty parquet file: {path}")
    if "prompt" not in df.columns:
        raise SystemExit(f"Missing required column 'prompt' in {path}; columns={list(df.columns)}")
    print(f"Data check passed: {path}, rows={len(df)}, columns={list(df.columns)}")
PY
}

check_adapter_paths() {
  case "${METHOD}" in
    lora)
      return 0
      ;;
    pissa)
      require_dir "${USER_MODEL_BASE_PATH:-${BUNDLE_BASE}/pissa_r32_a32/base}" "PiSSA base model directory"
      require_dir "${LORA_ADAPTER_PATH:-${BUNDLE_BASE}/pissa_r32_a32/adapter}" "PiSSA adapter directory"
      ;;
    milora)
      require_dir "${USER_MODEL_BASE_PATH:-${BUNDLE_BASE}/milora_r32_a32/base}" "MiLoRA base model directory"
      require_dir "${LORA_ADAPTER_PATH:-${BUNDLE_BASE}/milora_r32_a32/adapter}" "MiLoRA adapter directory"
      ;;
    rlpo)
      require_dir "${LORA_ADAPTER_PATH:-${BUNDLE_BASE}/rlpo_r16_a32/adapter}" "RLPO adapter directory"
      ;;
    rlmo)
      require_dir "${LORA_ADAPTER_PATH:-${BUNDLE_BASE}/rlmo_r16_a32/adapter}" "RLMO adapter directory"
      ;;
  esac
}

bash -n "${RUN_SCRIPT}"
require_file "${TRAIN_FILE}" "training parquet"
require_file "${TEST_FILE}" "evaluation parquet"
check_python_imports
check_data_schema
check_adapter_paths
mkdir -p "${CKPT_ROOT}" outputs

echo "Static smoke checks passed for method=${METHOD}"

if [[ "${MODE}" == "run" ]]; then
  echo "Running a tiny training smoke test for method=${METHOD}"
  SEED=${SEED:-42} \
  EXP_NAME=${EXP_NAME:-smoke_${METHOD}_1p5b_seed${SEED:-42}} \
  LOG_FILE=${LOG_FILE:-./outputs/smoke_${METHOD}.log} \
  CKPT_ROOT="${CKPT_ROOT}" \
  TRAIN_FILE="${TRAIN_FILE}" \
  TEST_FILE="${TEST_FILE}" \
  MODEL_BASE_PATH="${USER_MODEL_BASE_PATH:-${DEFAULT_PRETRAINED_MODEL}}" \
  BUNDLE_BASE="${BUNDLE_BASE}" \
  TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1} \
  SAVE_FREQ=${SAVE_FREQ:-1} \
  TEST_FREQ=${TEST_FREQ:--1} \
  TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-1} \
  TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-1} \
  N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1} \
  MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256} \
  bash "${RUN_SCRIPT}"
else
  echo "Set SMOKE_MODE=run to execute a one-step training smoke test."
fi
