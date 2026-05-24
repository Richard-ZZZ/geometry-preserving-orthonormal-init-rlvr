#!/usr/bin/env bash
set -xeuo pipefail

# RLMO for verl DAPO experiments.
# Initialization: A = orthonormal minor SVD directions (Vh[-r:]), B = 0.
# delta = 0, so the base model is NOT modified; use the original pretrained model.
#
# Required env vars for multi-job launch:
#   CUDA_VISIBLE_DEVICES, SEED, EXP_NAME, LOG_FILE

SEED=${SEED:-42}
EXP_NAME=${EXP_NAME:-rlmo_1p5b_paper_seed${SEED}}
LOG_FILE=${LOG_FILE:-./outputs/${EXP_NAME}.log}

PROJECT_NAME=${PROJECT_NAME:-RLMO-Paper-Replication}
TRAIN_FILE=${TRAIN_FILE:-./data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-./data/aime-2024.parquet}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
ACTOR_LR=${ACTOR_LR:-1e-5}
ACTOR_LR_SCHEDULER_TYPE=${ACTOR_LR_SCHEDULER_TYPE:-constant}
ACTOR_LR_WARMUP_STEPS=${ACTOR_LR_WARMUP_STEPS:-0}
TARGET_MODULES=${TARGET_MODULES:-all-linear}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TEST_FREQ=${TEST_FREQ:--1}
SAVE_FREQ=${SAVE_FREQ:-50}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
ACTOR_FSDP_SIZE=${ACTOR_FSDP_SIZE:-${N_GPUS_PER_NODE}}

PRETRAINED=${PRETRAINED:-${MODEL_BASE_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}}
BUNDLE_BASE=${BUNDLE_BASE:-./adapter_bundles/deepseek_r1_distill_qwen_1_5b}

# RLMO initialization does NOT modify the base model (delta=0).
# Use original pretrained weights + orthonormal-initialized adapter.
MODEL_BASE_PATH=${MODEL_BASE_PATH:-${PRETRAINED}}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-${BUNDLE_BASE}/rlmo_r16_a32/adapter}

CKPT_ROOT=${CKPT_ROOT:-./checkpoints}
CKPT_DIR=${CKPT_DIR:-${CKPT_ROOT}/${PROJECT_NAME}/${EXP_NAME}}

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-128}
TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-32}

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

ENABLE_OVERLONG_BUFFER=True
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-4096}
OVERLONG_PENALTY_FACTOR=1.0
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1024}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}

ACTOR_PPO_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
INFER_PPO_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))

mkdir -p "$(dirname "${LOG_FILE}")" "${CKPT_DIR}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python3 -m verl.trainer.main_ppo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.truncation='left' \
  data.seed=${SEED} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.train_batch_size=${TRAIN_PROMPT_BSZ} \
  actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.model.use_remove_padding=True \
  +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN} \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.model.path="${MODEL_BASE_PATH}" \
  actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
  actor_rollout_ref.model.lora_rank=${LORA_RANK} \
  actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
  actor_rollout_ref.model.target_modules=${TARGET_MODULES} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.optim.lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS} \
  actor_rollout_ref.actor.optim.lr_scheduler_type=${ACTOR_LR_SCHEDULER_TYPE} \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
  actor_rollout_ref.actor.fsdp_config.seed=${SEED} \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.ref.fsdp_config.seed=${SEED} \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${ACTOR_FSDP_SIZE} \
  reward_model.reward_manager=dapo \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${ENABLE_OVERLONG_BUFFER} \
  +reward_model.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN} \
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR} \
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=1 \
  trainer.val_before_train=${VAL_BEFORE_TRAIN} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode=auto \
  trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
  >> "${LOG_FILE}" 2>&1
