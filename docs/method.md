# LoRA-RLPO and LoRA-RLMO: Method Specification

This document is the implementation-level specification of **LoRA-RLPO** and **LoRA-RLMO**, the two geometry-preserving orthonormal initialization methods proposed in *Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR*.

## 1. Parameterization

For each targeted pretrained linear weight

```math
W_0\in\mathbb{R}^{m\times n},
```

we use the PEFT LoRA parameterization

```math
W = W_0 + sBA,
\qquad
B\in\mathbb{R}^{m\times r},
\quad
A\in\mathbb{R}^{r\times n},
\quad
s=\frac{\alpha}{r}.
```

The paper suppresses the scalar $s$ for clarity. It does not affect the initialization identity because both methods set $B_0=0$.

## 2. Spectral Basis

Compute the thin singular value decomposition in float32:

```math
W_0=U\Sigma V^\top,
\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_k),
\qquad
\sigma_1\geq\cdots\geq\sigma_k\geq0,
```

where $k=\min(m,n)$. The rows of $V^\top$ are orthonormal.

## 3. LoRA-RLPO and LoRA-RLMO Initialization

Select $r$ rows of $V^\top$, place them directly in $A$, and zero $B$:

```math
B_0=0_{m\times r},
\qquad
A_0=V_{\mathcal S}^{\top}.
```

The public variants differ only in the selected spectral index set:

| Public name | Script name | Selection | Exact initialization |
|---|---|---|---|
| LoRA-RLPO | `rlpo` | top-$r$ / principal | $B_0=0,\ A_0=V_r^\top$ |
| LoRA-RLMO | `rlmo` | bottom-$r$ / minor | $B_0=0,\ A_0=V_{-r}^\top$ |

No singular values are folded into either LoRA factor. In particular, LoRA-RLPO and LoRA-RLMO do **not** use $\Sigma^{1/2}$.

## 4. Required Invariants

An implementation conforms to LoRA-RLPO or LoRA-RLMO when, up to the numerical tolerance of its storage dtype, it satisfies all of the following for every initialized layer:

```math
A_0A_0^\top=I_r,
\qquad
B_0=0,
\qquad
sB_0A_0=0,
\qquad
W\rvert_{t=0}=W_0.
```

Consequences:

- The pretrained base weights remain unchanged; the proposed methods need no compensated or residual base model.
- The initialized model produces the same function as the pretrained model when adapters are enabled.
- For the first gradient step, the induced weight update lies in the selected right-singular subspace.
- Unlike PiSSA and MiLoRA, the proposed methods retain subspace information without scaling the update geometry by pretrained singular values.

## 5. Code Mapping

The reference implementation is in [`../scripts/prepare_init_adapters.py`](../scripts/prepare_init_adapters.py):

1. `select_svd_components(..., mode="max")` selects the top-$r$ right singular vectors for LoRA-RLPO.
2. `select_svd_components(..., mode="min")` selects the bottom-$r$ right singular vectors for LoRA-RLMO.
3. `initialize_orthogonal_zero_b_layer` assigns the selected rows of `Vh` to `lora_A` and creates an all-zero `lora_B`.
4. `apply_orthogonal_zero_b_init` applies the rule to every targeted linear module.

For a single layer, the implementation is equivalent to:

```python
_, _, vh = torch.linalg.svd(weight.float(), full_matrices=False)
A0 = vh[:rank]       # LoRA-RLPO
# A0 = vh[-rank:]    # LoRA-RLMO
B0 = torch.zeros(weight.shape[0], rank)
```

Use `--target-modules all-linear` to reproduce the paper's all-linear-layer setting. SVD is performed in float32 and the initialized factors are then cast to the base layer's storage dtype.

## 6. Reference Configuration

The main DAPO 1.5B experiments use:

| Setting | Value |
|---|---|
| Base model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| Target modules | all linear layers |
| Optimizer | AdamW |
| Learning rate | $10^{-5}$ |
| Training algorithm | DAPO with GRPO advantages |

The complete runtime settings and public launch scripts are indexed from the repository [`README.md`](../README.md).
