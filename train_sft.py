"""
AlphaToken SFT entry point.

Usage:
    python train_sft.py \
        --model_name_or_path ... \
        --train_data_path    ... \
        --output_dir         ...
"""

import argparse
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.sft_trainer import AlphaSFTConfig, train_alpha_sft


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaToken SFT Training")

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="...",
                        help="Path to pretrained model or HuggingFace model id")

    # Data
    parser.add_argument("--train_data_path", type=str, default="...",
                        help="Path to training data JSON/JSONL "
                             "(list of {'prompt', 'response'} dicts)")
    parser.add_argument("--output_dir", type=str, default="...",
                        help="Output directory for checkpoints and final model")

    # AlphaToken hyper-parameters (paper defaults)
    parser.add_argument("--rho", type=float, default=0.5,
                        help="Retained-token ratio ρ (default: 0.5)")
    parser.add_argument("--lambda_stab", type=float, default=1.5,
                        help="Stability weight λ (default: 1.5)")
    parser.add_argument("--K", type=int, default=3,
                        help="Number of last transformer layers to score (default: 3)")
    parser.add_argument("--W", type=int, default=32,
                        help="Causal window size W (default: 32)")
    parser.add_argument("--B_val", type=int, default=32,
                        help="Validation scoring batch size (default: 32)")
    parser.add_argument("--n_fisher_samples", type=int, default=1000,
                        help="Number of prompts for diagonal MC-Fisher (default: 1000)")

    # Training
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])

    # Misc
    parser.add_argument("--fisher_cache_path", type=str, default=None,
                        help="Optional path to cache/load pre-computed Fisher information")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--trust_remote_code", action="store_true")

    return parser.parse_args()


def load_data(path: str):
    """Load JSON or JSONL training data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found: {path}")
    with open(path) as f:
        first_char = f.read(1)
    with open(path) as f:
        if first_char == "[":
            return json.load(f)
        # JSONL
        return [json.loads(line) for line in f if line.strip()]


def main():
    args = parse_args()

    # ── Load model and tokenizer ──────────────────────────────────────────
    print(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=__import__("torch").bfloat16
        if args.dtype == "bfloat16" else __import__("torch").float16,
    )

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading training data: {args.train_data_path}")
    train_data = load_data(args.train_data_path)
    print(f"  Total examples: {len(train_data)}")

    # ── Build config ──────────────────────────────────────────────────────
    config = AlphaSFTConfig(
        rho=args.rho,
        lambda_stab=args.lambda_stab,
        K=args.K,
        W=args.W,
        B_val=args.B_val,
        n_fisher_samples=args.n_fisher_samples,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        dtype=args.dtype,
        output_dir=args.output_dir,
        fisher_cache_path=args.fisher_cache_path,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

    print("\nAlphaSFT config:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    # ── Train ─────────────────────────────────────────────────────────────
    train_alpha_sft(model, tokenizer, train_data, config)


if __name__ == "__main__":
    main()
