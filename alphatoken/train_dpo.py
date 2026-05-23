"""
AlphaToken DPO entry point.

Usage:
    python train_dpo.py \
        --model_name_or_path  ... \
        --sft_warmstart_path  ... \
        --train_data_path     ... \
        --val_sft_data_path   ... \
        --output_dir          ...

The pipeline follows the paper's two-stage preference alignment:
  1. SFT warm-start on UltraChat-200K  (done separately, result = sft_warmstart_path)
  2. Preference optimisation on UltraFeedback  (this script)

Both the policy model (--model_name_or_path, loaded from sft_warmstart_path)
and the frozen reference model (--sft_warmstart_path) start from the same SFT
checkpoint, consistent with the paper.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dpo_trainer import AlphaDPOConfig, train_alpha_dpo


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaToken DPO Training")

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="...",
                        help="Path to policy model initialisation (SFT warm-start)")
    parser.add_argument("--sft_warmstart_path", type=str, default="...",
                        help="Path to frozen reference model (same SFT warm-start)")

    # Data
    parser.add_argument("--train_data_path", type=str, default="...",
                        help="Preference pairs JSON/JSONL "
                             "(list of {'prompt', 'chosen', 'rejected'} dicts)")
    parser.add_argument("--val_sft_data_path", type=str, default="...",
                        help="SFT-format validation data for scoring signals "
                             "(list of {'prompt', 'response'} dicts)")
    parser.add_argument("--output_dir", type=str, default="...",
                        help="Output directory for checkpoints and final model")

    # AlphaToken
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--lambda_stab", type=float, default=1.5)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--B_val", type=int, default=32)
    parser.add_argument("--n_fisher_samples", type=int, default=1000)

    # DPO
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO β temperature (default: 0.1)")

    # Training  (paper: lr=2e-5 for Llama/Gemma, 1e-5 for Qwen)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])

    # Misc
    parser.add_argument("--fisher_cache_path", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--trust_remote_code", action="store_true")

    return parser.parse_args()


def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path) as f:
        first_char = f.read(1)
    with open(path) as f:
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def main():
    args = parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Per paper §5.1: θ_ref is the UltraChat-200K warm-start checkpoint, and
    # the policy model is INITIALISED from the same checkpoint.  We enforce
    # that here so the Fisher-drift proxy V_l = F_Wl ⊙ (W_l − W_l_ref) is
    # anchored consistently — otherwise the residual r_0 = ∇L_ret(θ_ref) bound
    # (App. B.4.2) no longer applies.
    if args.model_name_or_path != args.sft_warmstart_path:
        raise ValueError(
            "AlphaDPO requires the policy and reference to be initialised from "
            "the same SFT warm-start checkpoint. "
            f"Got policy={args.model_name_or_path} vs ref={args.sft_warmstart_path}."
        )

    # ── Load tokenizer and models ─────────────────────────────────────────
    print(f"Loading policy model:    {args.model_name_or_path}")
    print(f"Loading reference model: {args.sft_warmstart_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_warmstart_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading preference data: {args.train_data_path}")
    train_data = load_data(args.train_data_path)
    print(f"  Preference pairs: {len(train_data)}")

    print(f"Loading validation SFT data: {args.val_sft_data_path}")
    val_sft_data = load_data(args.val_sft_data_path)
    print(f"  Validation examples: {len(val_sft_data)}")

    # ── Build config ──────────────────────────────────────────────────────
    config = AlphaDPOConfig(
        rho=args.rho,
        lambda_stab=args.lambda_stab,
        K=args.K,
        W=args.W,
        B_val=args.B_val,
        n_fisher_samples=args.n_fisher_samples,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        dtype=args.dtype,
        output_dir=args.output_dir,
        sft_warmstart_path=args.sft_warmstart_path,
        fisher_cache_path=args.fisher_cache_path,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

    print("\nAlphaDPO config:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    # ── Train ─────────────────────────────────────────────────────────────
    train_alpha_dpo(model, ref_model, tokenizer, train_data, val_sft_data, config)


if __name__ == "__main__":
    main()
