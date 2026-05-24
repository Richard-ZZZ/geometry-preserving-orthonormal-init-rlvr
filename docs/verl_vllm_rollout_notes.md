# verl / vLLM Rollout Notes

The experiments in this repository assume a patched `verl` runtime rather than a completely stock checkout. The most important changes are around LoRA/SVD-initialized adapters, rollout synchronization, and vLLM model loading.

## Why The Patch Is Needed

For PiSSA, MiLoRA, RLPO, and RLMO, the adapter initialization can be represented as an SVD-initialized LoRA adapter. Some initialization variants store a residual base weight internally. If rollout, actor log-prob, and reference/base-model log-prob do not resolve the same effective policy, PPO will compare mismatched policies.

The patch makes sure that:

- vLLM rollout serves the same effective model as the actor policy.
- LoRA adapters are loaded and requested consistently during generation.
- Base-model KL diagnostics use the correct frozen reference path for SVD-initialized adapters.
- DAPO training avoids unnecessary reference-policy log-prob calls when KL is disabled.

## Required Runtime Changes

### 1. Model/tokenizer path resolution

`verl.trainer.main_ppo` should construct tokenizer and processor through the same `HFModelConfig` path resolution used by workers. This avoids reward/dataset-side tokenization using a different path from the actor/rollout runtime.

Relevant local files:

- `verl/trainer/main_ppo.py`
- `verl/workers/config/model.py`

### 2. vLLM packed-weight loader compatibility

`verl.utils.vllm.patch` includes a vLLM Qwen2 packed-weight loader patch so vLLM can load PEFT-wrapped or base-layer-remapped Qwen-style weights correctly.

Relevant local file:

- `verl/utils/vllm/patch.py`

### 3. Rollout runtime model path resolution

The vLLM rollout server resolves the runtime model path from the model config and adapter config. For residual-base SVD initialization, rollout must not accidentally serve the residual base without the matching adapter semantics.

Relevant local files:

- `verl/utils/vllm/patch.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`

### 4. LoRA adapter synchronization into rollout

The FSDP actor/rollout worker includes custom LoRA synchronization logic so that vLLM generation uses the current adapter weights after actor updates. This is especially important for PEFT adapters and SVD-initialized methods.

Relevant local file:

- `verl/workers/fsdp_workers.py`

### 5. Correct base-model KL for SVD-initialized adapters

For SVD-initialized adapters, `disable_adapter()` can expose a residual base rather than the original pretrained base. The actor worker therefore keeps a frozen initialization/reference adapter path for base-model KL diagnostics instead of relying only on disabling adapters.

Relevant local files:

- `verl/workers/fsdp_workers.py`
- `verl/workers/actor/dp_actor.py`

### 6. DAPO reference log-prob behavior

DAPO runs here set KL-in-reward and actor KL loss to false. The trainer patch skips reference-policy log-prob unless it is actually consumed, and computes optional base-model KL lazily inside actor workers.

Relevant local file:

- `verl/trainer/ppo/ray_trainer.py`

## Script Flags That Depend On These Changes

The launch scripts use rollout-related settings such as:

- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.model.lora_adapter_path=...`
- `actor_rollout_ref.model.lora_rank=...`
- `actor_rollout_ref.model.lora_alpha=...`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`
- `actor_rollout_ref.rollout.enable_chunked_prefill=True`
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`

For RLPO and RLMO, make sure the released adapter bundles and the patched runtime agree on the adapter directory names used by the scripts.

## Release Recommendation

For reproducibility, publish one of the following with this repository:

1. A fork/branch of `verl` containing these runtime patches, or
2. A `patches/` directory with a git patch generated from the modified `verl` files.

Minimum files to include in that patch set:

- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/utils/vllm/patch.py`
- `verl/workers/config/model.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
