#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# AlphaToken SFT experiment launcher
#
# Reproduces the supervised fine-tuning setup from the paper (Sec. 5.1):
#   - Backbone: Llama-3.2-3B / Gemma-3-4B / Qwen-3.5-9B
#   - Dataset:  Magicoder (75K instruction examples)
#   - Evaluation: HumanEval (target) + ARC-C, HellaSwag, MMLU, GSM8K (retention)
#
# Usage:
#   bash scripts/run_sft.sh
#
# Adjust MODEL_PATH, DATA_PATH, OUTPUT_DIR, and GPU settings for your cluster.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (fill in before running) ───────────────────────────────────────────
MODEL_PATH="..."          # e.g. "meta-llama/Llama-3.2-3B" or local path
DATA_PATH="..."           # path to Magicoder JSONL/JSON
OUTPUT_DIR="..."          # where checkpoints and final model are saved
FISHER_CACHE="..."        # optional: path to cache diagonal Fisher (speeds up reruns)

# ── AlphaToken hyper-parameters (paper defaults, Table C.4 / App. C.4) ───────
RHO=0.5           # retained-token ratio
LAMBDA_STAB=1.5   # stability weight λ
K=3               # last K transformer layers for scoring
W=32              # causal window size
B_VAL=32          # validation scoring batch size
N_FISHER=1000     # prompts for Monte-Carlo Fisher

# ── Training hyper-parameters ─────────────────────────────────────────────────
# Paper: global batch 64, max_length 4096, 3 epochs, lr 2e-5 (Llama/Gemma).
# The trainer supports DDP via torchrun (auto-detected from LOCAL_RANK).
# 4 × A100 layout:  N_GPUS × PER_DEVICE_BS × GRAD_ACCUM = 4 × 4 × 4 = 64.
N_GPUS=4
PER_DEVICE_BS=4
GRAD_ACCUM=4
MAX_LEN=4096
EPOCHS=3
LR=2e-5
DTYPE=bfloat16

# ── Launch ────────────────────────────────────────────────────────────────────
echo "=== AlphaToken SFT ==="
echo "Model:      ${MODEL_PATH}"
echo "Data:       ${DATA_PATH}"
echo "Output:     ${OUTPUT_DIR}"
GLOBAL_BS=$((N_GPUS * PER_DEVICE_BS * GRAD_ACCUM))
echo "World size: ${N_GPUS}  (per-device bs=${PER_DEVICE_BS}, accum=${GRAD_ACCUM} -> global ${GLOBAL_BS})"
echo "ρ=${RHO}  λ=${LAMBDA_STAB}  K=${K}  W=${W}  B_val=${B_VAL}"

if [ "${N_GPUS}" -gt 1 ]; then
    LAUNCHER="torchrun --standalone --nproc_per_node=${N_GPUS}"
else
    LAUNCHER="python"
fi

${LAUNCHER} train_sft.py \
    --model_name_or_path  "${MODEL_PATH}" \
    --train_data_path     "${DATA_PATH}" \
    --output_dir          "${OUTPUT_DIR}" \
    --rho                 "${RHO}" \
    --lambda_stab         "${LAMBDA_STAB}" \
    --K                   "${K}" \
    --W                   "${W}" \
    --B_val               "${B_VAL}" \
    --n_fisher_samples    "${N_FISHER}" \
    --learning_rate       "${LR}" \
    --num_epochs          "${EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --max_length          "${MAX_LEN}" \
    --warmup_ratio        0.03 \
    --weight_decay        0.01 \
    --max_grad_norm       1.0 \
    --dtype               "${DTYPE}" \
    --fisher_cache_path   "${FISHER_CACHE}" \
    --logging_steps       10 \
    --save_steps          200

echo "=== SFT training complete. Model saved to ${OUTPUT_DIR} ==="
