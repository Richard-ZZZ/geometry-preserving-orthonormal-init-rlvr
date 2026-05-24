#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<EOF
Usage: bash scripts/prepare_data.sh

Environment overrides:
  DATA_DIR      Output directory, default: data
  TRAIN_FILE    Train parquet path, default: DATA_DIR/dapo-math-17k.parquet
  TEST_FILE     Test parquet path, default: DATA_DIR/aime-2024.parquet
  OVERWRITE     Set to 1 to download again
  TRAIN_URL     Override train parquet URL
  TEST_URL      Override test parquet URL
EOF
  exit 0
fi

DATA_DIR=${DATA_DIR:-data}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/aime-2024.parquet}
OVERWRITE=${OVERWRITE:-0}

TRAIN_URL=${TRAIN_URL:-https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true}
TEST_URL=${TEST_URL:-https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true}

download_file() {
  local url="$1"
  local output="$2"

  mkdir -p "$(dirname "$output")"
  if [ -f "$output" ] && [ "$OVERWRITE" != "1" ]; then
    echo "Found $output; set OVERWRITE=1 to download again."
    return 0
  fi

  echo "Downloading $url"
  echo "       -> $output"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    echo "Neither curl nor wget is available." >&2
    return 1
  fi
}

download_file "$TRAIN_URL" "$TRAIN_FILE"
download_file "$TEST_URL" "$TEST_FILE"

python - <<PY
import pandas as pd
for path in ["$TRAIN_FILE", "$TEST_FILE"]:
    df = pd.read_parquet(path)
    if "prompt" not in df.columns:
        raise SystemExit(f"Missing required column prompt in {path}")
    print(f"{path}: rows={len(df)} columns={list(df.columns)}")
PY
