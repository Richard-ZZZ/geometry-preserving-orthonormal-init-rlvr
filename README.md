# Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR

[![Paper](https://img.shields.io/badge/arXiv-2606.31813-b31b1b.svg)](https://arxiv.org/abs/2606.31813)
[![Checkpoints](https://img.shields.io/badge/HuggingFace-checkpoints-yellow.svg)](https://huggingface.co/RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts)
[![License: CC BY 4.0](https://img.shields.io/badge/paper-CC_BY_4.0-lightgrey.svg)](https://arxiv.org/abs/2606.31813)

Official implementation of **Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR**.

> **Method in one sentence.** For every frozen pretrained weight matrix, initialize the LoRA input factor with row-orthonormal right singular vectors and initialize the output factor to zero. This preserves the pretrained function exactly, retains its spectral geometry, and removes the singular-value scaling that can destabilize RLVR.

We propose two methods:

- **LoRA-RLPO** (Reinforcement Learning Principal Orthonormal initialization) uses the top right-singular subspace.
- **LoRA-RLMO** (Reinforcement Learning Minor Orthonormal initialization) uses the bottom right-singular subspace.

### Quick facts for search and retrieval

- **Research problem:** how to initialize LoRA for reinforcement learning with verifiable rewards (RLVR).
- **Proposed methods:** LoRA-RLPO and LoRA-RLMO.
- **Core idea:** zero-initialize $B_0$ and initialize $A_0$ with orthonormal right singular vectors of the pretrained weight.
- **Training setting:** DAPO on DAPO-Math-17K with DeepSeek-R1-Distill-Qwen-1.5B.
- **Main finding:** orthonormal initialization controls early update magnitude and improves RLVR stability; LoRA-RLPO achieves the best average result in the main comparison.
- **Related concepts:** LoRA initialization, parameter-efficient fine-tuning, PEFT, RLVR, DAPO, GRPO, orthonormal initialization, SVD, PiSSA, MiLoRA, KL stability, mathematical reasoning.

Machine-readable summaries are available in [`llms.txt`](llms.txt), [`codemeta.json`](codemeta.json), and [`CITATION.cff`](CITATION.cff).

## The Method at a Glance

For a pretrained matrix \(W_0\in\mathbb{R}^{m\times n}\), LoRA writes

$$
W = W_0 + sBA, \qquad
B\in\mathbb{R}^{m\times r},\quad
A\in\mathbb{R}^{r\times n},
$$

where \(s=\alpha/r\) is the usual PEFT scaling. Let

$$
W_0 = U\Sigma V^\top, \qquad
\sigma_1\geq\cdots\geq\sigma_k>0.
$$

**LoRA-RLPO and LoRA-RLMO initialize**

$$
\boxed{B_0=0,\qquad A_0=V_{\mathcal S}^{\top},\qquad A_0A_0^\top=I_r.}
$$

The selected index set \(\mathcal S\) defines the variant:

| Variant | Spectral subspace | Initialization |
|---|---|---|
| **LoRA-RLPO** | principal, \(\mathcal S=\{1,\ldots,r\}\) | \(B_0=0,\ A_0=V_r^\top\) |
| **LoRA-RLMO** | minor, \(\mathcal S=\{k-r+1,\ldots,k\}\) | \(B_0=0,\ A_0=V_{-r}^\top\) |

This form gives three immediate properties:

1. **Exact function preservation:** \(sB_0A_0=0\), so the initialized policy is exactly the pretrained policy. No residual-base rewrite is needed.
2. **Orthonormal rows:** \(A_0A_0^\top=I_r\), which minimizes the initialization-dependent term in our bound on the gap between LoRA and full fine-tuning.
3. **Geometry without spectral amplification:** the singular vectors preserve a pretrained spectral subspace, while omitting \(\Sigma^{1/2}\) avoids singular-value-scaled updates.

Under the paper's unscaled parameterization (\(s=1\)), the first LoRA weight update is a projection of the full update onto the selected right-singular subspace:

$$
\Delta W^{\mathrm{LoRA}}_1
=
\Delta W^{\mathrm{full}}_1 A_0^\top A_0,
\qquad
\lVert\Delta W^{\mathrm{LoRA}}_1\rVert_F
\leq
\lVert\Delta W^{\mathrm{full}}_1\rVert_F.
$$

See [`docs/method.md`](docs/method.md) for the exact method specification, PEFT mapping, and implementation invariants.

## How the Proposed Methods Differ from Other Initializations

| Method | \(B_0\) | \(A_0\) | Initial update \(B_0A_0\) | Uses pretrained geometry | Singular-value scaling |
|---|---|---|---:|---:|---:|
| LoRA | \(0\) | random Gaussian | \(0\) | No | No |
| PiSSA | \(U_r\Sigma_r^{1/2}\) | \(\Sigma_r^{1/2}V_r^\top\) | nonzero* | Principal | Yes |
| MiLoRA | \(U_{-r}\Sigma_{-r}^{1/2}\) | \(\Sigma_{-r}^{1/2}V_{-r}^\top\) | nonzero* | Minor | Yes |
| **LoRA-RLPO (ours)** | **\(0\)** | **\(V_r^\top\)** | **\(0\)** | **Principal** | **No** |
| **LoRA-RLMO (ours)** | **\(0\)** | **\(V_{-r}^\top\)** | **\(0\)** | **Minor** | **No** |

\* PiSSA and MiLoRA preserve the original function by subtracting \(B_0A_0\) from the frozen base weight. LoRA-RLPO and LoRA-RLMO do not modify the base model.

## Main Result

We train DeepSeek-R1-Distill-Qwen-1.5B with DAPO on DAPO-Math-17K and evaluate the step-500 checkpoints across three seeds. LoRA-RLPO and LoRA-RLMO improve both average performance and run-to-run stability over standard LoRA.

| Method | GSM8K | MATH500 | AIME22 | AIME23 | AIME24 | Average |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 75.64 | 81.93 | 43.33 | 38.89 | 72.22 | 62.40 ± 2.96 |
| MiLoRA | 75.41 | 76.47 | 28.89 | 30.00 | 47.78 | 51.71 ± 0.98 |
| PiSSA | 9.65 | 13.20 | 0.00 | 0.00 | 0.00 | 4.57 ± 3.26 |
| **LoRA-RLMO (ours)** | **76.42** | 86.80 | 42.22 | 41.11 | 72.22 | **63.76 ± 1.64** |
| **LoRA-RLPO (ours)** | 75.59 | **87.33** | **46.67** | **42.22** | **73.33** | **65.03 ± 0.55** |

Values are accuracies in percent. GSM8K uses pass@1, MATH500 pass@4, and AIME pass@32. Full settings, standard deviations per benchmark, ablations, and larger-model results are in the [paper](https://arxiv.org/abs/2606.31813).

## Quick Start: Generate LoRA-RLPO and LoRA-RLMO Adapters

After completing [`INSTALL.md`](INSTALL.md), generate the two rank-16 proposed initializations:

```bash
export MODEL_BASE_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export BUNDLE_BASE=adapter_bundles/deepseek_r1_distill_qwen_1_5b

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

The generated adapters use the original pretrained model directly:

```text
adapter_bundles/deepseek_r1_distill_qwen_1_5b/
  rlpo_r16_a32/adapter/   # A_0 = top-r right singular vectors, B_0 = 0
  rlmo_r16_a32/adapter/   # A_0 = bottom-r right singular vectors, B_0 = 0
```

Run a smoke test, then launch training:

```bash
bash scripts/smoke_test.sh rlpo

SEED=42 bash scripts/run_rlpo_1p5b_paper_seed.sh
SEED=42 bash scripts/run_rlmo_1p5b_paper_seed.sh
```

For the full LoRA/PiSSA/MiLoRA/LoRA-RLPO/LoRA-RLMO comparison, see [`adapter_bundles/README.md`](adapter_bundles/README.md) and [`manifests/experiments_1p5b.md`](manifests/experiments_1p5b.md).

## Reproduction Map

| What you need | Where to find it |
|---|---|
| Exact LoRA-RLPO/RLMO definition | [`docs/method.md`](docs/method.md) |
| Environment and patched `verl` setup | [`INSTALL.md`](INSTALL.md) |
| Adapter generation | [`scripts/prepare_init_adapters.py`](scripts/prepare_init_adapters.py) |
| Training entry points | [`scripts/run_rlpo_1p5b_paper_seed.sh`](scripts/run_rlpo_1p5b_paper_seed.sh), [`scripts/run_rlmo_1p5b_paper_seed.sh`](scripts/run_rlmo_1p5b_paper_seed.sh) |
| Method-to-hyperparameter map | [`manifests/experiments_1p5b.md`](manifests/experiments_1p5b.md) |
| Runtime patch and explanation | [`patches/verl_rollout_adapter_sync.patch`](patches/verl_rollout_adapter_sync.patch), [`docs/verl_vllm_rollout_notes.md`](docs/verl_vllm_rollout_notes.md) |
| Released checkpoints | [Hugging Face collection](https://huggingface.co/RuijiaZ/geometry-preserving-orthonormal-init-rlvr-artifacts) |

The experiments require a patched `verl`/vLLM runtime so that actor, reference, and rollout workers resolve and synchronize initialized adapters consistently. Apply the included patch before running the recipes.

## Scope of This Release

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- RLVR algorithm: DAPO
- Training data: DAPO-Math-17K
- Public comparison: LoRA, PiSSA, MiLoRA, LoRA-RLPO, and LoRA-RLMO
- Released checkpoint seeds: 41, 42, and 43

Paths are configurable through environment variables; copy [`env.example`](env.example) as the path contract for a new machine.

## Citation

If LoRA-RLPO, LoRA-RLMO, or this repository is useful in your research, please cite:

```bibtex
@article{zhang2026geometry,
  title   = {Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR},
  author  = {Zhang, Ruijia and Zhu, Jiacheng and Zhu, Hanqing and Shi, Laixi},
  journal = {arXiv preprint arXiv:2606.31813},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.31813}
}
```

GitHub's **Cite this repository** button is configured through [`CITATION.cff`](CITATION.cff).
