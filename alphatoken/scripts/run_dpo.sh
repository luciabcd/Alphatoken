#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# AlphaToken DPO experiment launcher
#
# Reproduces the preference alignment setup from the paper (Sec. 5.1):
#   Stage 1: SFT warm-start on UltraChat-200K  (done separately — see run_sft.sh)
#   Stage 2: Preference optimisation on UltraFeedback  (this script)
#
# Evaluation:
#   - Preference: AlpacaEval 2, Arena-Hard v0.1
#   - Retention:  ARC-C, HellaSwag, MMLU, GSM8K
#
# Usage:
#   bash scripts/run_dpo.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
# Both policy and reference are initialised from the same SFT warm-start ckpt
SFT_WARMSTART_PATH="..."   # path to UltraChat-200K warm-started SFT model
OUTPUT_DIR="..."           # output directory for DPO model
PREF_DATA_PATH="..."       # UltraFeedback binarised preference pairs (JSONL/JSON)
VAL_SFT_DATA_PATH="..."    # SFT-format validation data for GDP scoring signals
FISHER_CACHE="..."         # optional: pre-computed Fisher cache from SFT stage

# ── AlphaToken hyper-parameters ───────────────────────────────────────────────
RHO=0.5
LAMBDA_STAB=1.5
K=3
W=32
B_VAL=32
N_FISHER=1000

# ── DPO hyper-parameters ──────────────────────────────────────────────────────
BETA=0.1

# ── Training (paper: global batch 64 pairs, max_prompt 1024, max_resp 1024, 3 epochs)
# 4 × A100 DDP layout: 4 GPUs × 2 pairs/GPU × accum 8 = 64 pairs/step.
N_GPUS=4
PER_DEVICE_BS=2
GRAD_ACCUM=8
MAX_PROMPT_LEN=1024
MAX_RESP_LEN=1024
EPOCHS=3
# lr=2e-5 for Llama-3.2-3B / Gemma-3-4B;  1e-5 for Qwen-3.5-9B
LR=2e-5
DTYPE=bfloat16

# ── Launch ────────────────────────────────────────────────────────────────────
GLOBAL_BS=$((N_GPUS * PER_DEVICE_BS * GRAD_ACCUM))
echo "=== AlphaToken DPO ==="
echo "Policy / Ref:   ${SFT_WARMSTART_PATH}"
echo "Pref data:      ${PREF_DATA_PATH}"
echo "Val SFT data:   ${VAL_SFT_DATA_PATH}"
echo "Output:         ${OUTPUT_DIR}"
echo "World size: ${N_GPUS}  (per-device bs=${PER_DEVICE_BS}, accum=${GRAD_ACCUM} -> global ${GLOBAL_BS} pairs)"
echo "ρ=${RHO}  λ=${LAMBDA_STAB}  β=${BETA}  K=${K}  W=${W}"

if [ "${N_GPUS}" -gt 1 ]; then
    LAUNCHER="torchrun --standalone --nproc_per_node=${N_GPUS}"
else
    LAUNCHER="python"
fi

${LAUNCHER} train_dpo.py \
    --model_name_or_path           "${SFT_WARMSTART_PATH}" \
    --sft_warmstart_path           "${SFT_WARMSTART_PATH}" \
    --train_data_path              "${PREF_DATA_PATH}" \
    --val_sft_data_path            "${VAL_SFT_DATA_PATH}" \
    --output_dir                   "${OUTPUT_DIR}" \
    --rho                          "${RHO}" \
    --lambda_stab                  "${LAMBDA_STAB}" \
    --K                            "${K}" \
    --W                            "${W}" \
    --B_val                        "${B_VAL}" \
    --n_fisher_samples             "${N_FISHER}" \
    --beta                         "${BETA}" \
    --learning_rate                "${LR}" \
    --num_epochs                   "${EPOCHS}" \
    --per_device_train_batch_size  "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps  "${GRAD_ACCUM}" \
    --max_prompt_length            "${MAX_PROMPT_LEN}" \
    --max_response_length          "${MAX_RESP_LEN}" \
    --warmup_ratio                 0.03 \
    --weight_decay                 0.01 \
    --max_grad_norm                1.0 \
    --dtype                        "${DTYPE}" \
    --fisher_cache_path            "${FISHER_CACHE}" \
    --logging_steps                10 \
    --save_steps                   200

echo "=== DPO training complete. Model saved to ${OUTPUT_DIR} ==="
