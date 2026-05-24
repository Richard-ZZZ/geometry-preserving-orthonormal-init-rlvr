# Data

The training scripts expect two parquet files by default:

```text
data/dapo-math-17k.parquet
data/aime-2024.parquet
```

## Download

Run the downloader from the repository root:

```bash
bash scripts/prepare_data.sh
```

The script downloads the public parquet files used by the DAPO recipe:

- Train: `BytedTsinghua-SIA/DAPO-Math-17k`, file `data/dapo-math-17k.parquet`
- Test: `BytedTsinghua-SIA/AIME-2024`, file `data/aime-2024.parquet`

You can override output paths or force a refresh:

```bash
DATA_DIR=/path/to/data bash scripts/prepare_data.sh
OVERWRITE=1 bash scripts/prepare_data.sh
TRAIN_FILE=/path/to/train.parquet TEST_FILE=/path/to/test.parquet bash scripts/prepare_data.sh
```

## Required Columns

The scripts use:

- `prompt`: model input prompt.
- Ground-truth fields consumed by `scripts/reward_score_math_dapo.py` through the DAPO reward pipeline.

`prepare_data.sh` verifies that the `prompt` column exists after download. You can inspect the full schema with:

```bash
python - <<PY
import pandas as pd
for path in ["data/dapo-math-17k.parquet", "data/aime-2024.parquet"]:
    df = pd.read_parquet(path)
    print(path, len(df), list(df.columns))
    print(df.head(1).to_dict(orient="records"))
PY
```

## External Data

You can keep the data outside this repository and export paths before running:

```bash
export TRAIN_FILE=/path/to/dapo-math-17k.parquet
export TEST_FILE=/path/to/aime-2024.parquet
```

Do not commit large parquet files unless you intentionally use Git LFS.
