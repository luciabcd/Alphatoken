"""
AlphaSFT Trainer — value-aware supervised fine-tuning (Eq. 13 / Algorithm 1).

L_Alpha-SFT(θ) = Σ_t I[Φ(y_t) ≥ τ_ρ] · ℓ_t(θ)

Training procedure (Algorithm 1):
  1. Sample minibatch from D_train.
  2. Compute Φ_dir/cau_tgt  via Eqs. (8), (9).
  3. Refresh  V_l = F_Wl ⊙ (W_l − W_ref_l).
  4. Compute Φ_dir/cau_prx  via Eq. (11).
  5. Aggregate Φ(y_t) via Eq. (12); mask m_t = I[Φ(y_t) ≥ τ_ρ].
  6. Form L_Alpha-SFT and back-propagate for the actual parameter update.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

from .fisher import (
    compute_diagonal_fisher,
    get_scoring_layer_indices,
    get_linear_modules,
    save_fisher,
    load_fisher,
)
from .scoring import (
    AlphaTokenScorer,
    HookManager,
    LayerSignals,
    top_rho_mask,
    _causal_window_mask,
    _find_vproj_key,
    _unwrap_model,
)


# ──────────────────────────────────────────────────────────────────────────────
# Distributed-training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _is_main() -> bool:
    return _rank() == 0


def _barrier() -> None:
    if _is_dist():
        dist.barrier()


@contextlib.contextmanager
def _no_sync_ctx(model: nn.Module):
    """Skip DDP gradient-allreduce during this backward (we discard it anyway)."""
    if isinstance(model, DDP):
        with model.no_sync():
            yield
    else:
        yield


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AlphaSFTConfig:
    # AlphaToken hyper-parameters (paper defaults)
    rho: float = 0.5          # retained-token ratio
    lambda_stab: float = 1.5  # stability weight λ
    K: int = 3                # last K transformer layers for scoring
    W: int = 32               # causal window size
    B_val: int = 32           # validation batch size for scoring signals
    n_fisher_samples: int = 1000  # prompts for Monte-Carlo Fisher

    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    max_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"

    # Paths
    output_dir: str = "..."
    fisher_cache_path: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 200
    data_split_seed: int = 42   # used for random val / Fisher / train split


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize_sft_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    """
    Tokenize a prompt–response pair.
    Returns {'input_ids', 'attention_mask', 'labels', 'prompt_len'}.
    Labels are -100 for prompt tokens (masked from loss).
    """
    prompt = example.get("prompt", example.get("instruction", ""))
    response = example.get("response", example.get("output", ""))

    # Encode prompt and full sequence separately to find the split point
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_text = prompt + response
    full_ids = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    prompt_len = min(len(prompt_ids), len(full_ids))

    labels = full_ids.copy()
    for i in range(prompt_len):
        labels[i] = -100

    pad_len = max_length - len(full_ids)
    input_ids = full_ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad_len
    attn_mask = [1] * len(full_ids) + [0] * pad_len
    padded_labels = labels + [-100] * pad_len

    return {
        "input_ids": torch.tensor(input_ids[:max_length], dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask[:max_length], dtype=torch.long),
        "labels": torch.tensor(padded_labels[:max_length], dtype=torch.long),
        "prompt_len": prompt_len,
    }


class SFTDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ):
        self.samples = [
            _tokenize_sft_example(ex, tokenizer, max_length) for ex in data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Per-token cross-entropy
# ──────────────────────────────────────────────────────────────────────────────

def per_token_ce_loss(
    logits: torch.Tensor,   # (B, T, V)
    labels: torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """
    Return token-level cross-entropy losses (B, T).
    Positions with label=-100 get loss 0.
    """
    B, T, V = logits.shape
    shift_logits = logits[:, :-1].contiguous()   # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()    # (B, T-1)

    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T - 1)  # (B, T-1)

    # Pad back to T
    token_losses = F.pad(loss_flat, (0, 1), value=0.0)  # (B, T)
    return token_losses


# ──────────────────────────────────────────────────────────────────────────────
# Core scorer that handles batched sequences
# ──────────────────────────────────────────────────────────────────────────────

def _collect_flat_signals(
    hook_mgr: HookManager,
    batch_indices: List[int],
    seq_lengths: List[int],
    layer_indices: List[int],
    response_mask_2d: Optional[torch.Tensor] = None,
    response_mask_offset: int = 0,
    keep_only_response: bool = False,
) -> LayerSignals:
    """
    Extract per-sequence signals for the given batch indices.

    Returns a LayerSignals where:
      - acts/grads are 2-D tensors (N_tokens_total, d), concatenated over the
        batch members in `batch_indices`, optionally filtered to response
        positions only when `keep_only_response=True`.
      - attn_weights[l_idx] is a List[Tensor], one (T_i, T_i) head-averaged
        attention matrix PER batch member.  This preserves the per-sample
        attention required by Eq. 9.

    response_mask_2d:    (B_total, T) bool mask of response positions
                         (in the same row-order as the underlying hook tensors)
    response_mask_offset: row offset into response_mask_2d for these batch_indices
                         (e.g., B_train when collecting validation signals)
    """
    sig = LayerSignals()

    keys = list(hook_mgr.signals.acts.keys())
    for key in keys:
        li, _ = key
        if li not in layer_indices:
            continue

        raw_acts = hook_mgr.signals.acts.get(key)
        raw_grads = hook_mgr.signals.grads.get(key)
        if raw_acts is None:
            continue

        act_parts, grad_parts = [], []
        for bi, sl in zip(batch_indices, seq_lengths):
            sl_i = int(sl)
            if raw_acts.dim() == 3:
                a = raw_acts[bi, :sl_i]
                g = raw_grads[bi, :sl_i] if (raw_grads is not None and raw_grads.dim() == 3) else None
            else:
                a = raw_acts[:sl_i]
                g = raw_grads[:sl_i] if raw_grads is not None else None

            if keep_only_response and response_mask_2d is not None:
                # response_mask_2d row index for this batch member
                row = response_mask_offset + (bi - batch_indices[0])
                rm = response_mask_2d[row, :sl_i].to(a.device)
                a = a[rm]
                if g is not None:
                    g = g[rm]

            act_parts.append(a)
            if g is not None:
                grad_parts.append(g)

        if act_parts:
            sig.acts[key] = torch.cat(act_parts, dim=0).float()
        if grad_parts:
            sig.grads[key] = torch.cat(grad_parts, dim=0).float()

    # Per-sample attention weights — one matrix per batch member, NOT a single
    # (e.g., batch[0]) matrix shared across the batch.
    for l_idx, aw in hook_mgr.signals.attn_weights.items():
        if l_idx in layer_indices:
            # aw: (B, H, T, T)
            sig.attn_weights[l_idx] = [
                aw[bi].mean(dim=0).float() for bi in batch_indices
            ]

    return sig


# ──────────────────────────────────────────────────────────────────────────────
# AlphaSFT step
# ──────────────────────────────────────────────────────────────────────────────

def _save_grads(model: nn.Module) -> Dict[str, Optional[torch.Tensor]]:
    """Snapshot current param.grad state (for gradient-accumulation correctness)."""
    snap: Dict[str, Optional[torch.Tensor]] = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            snap[n] = p.grad.detach().clone() if p.grad is not None else None
    return snap


def _restore_grads(model: nn.Module, snap: Dict[str, Optional[torch.Tensor]]) -> None:
    """Restore param.grad to the snapshotted state (discarding scoring contribution)."""
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = snap.get(n)
        if g is None:
            p.grad = None
        else:
            p.grad = g


def alpha_sft_loss(
    model: PreTrainedModel,
    scorer: AlphaTokenScorer,
    hook_mgr: HookManager,
    train_batch: dict,
    val_batch: dict,
    config: AlphaSFTConfig,
    device: torch.device,
    dtype: torch.dtype,
    layer_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One AlphaSFT step. Returns (masked_loss, mean_phi).

    Procedure (Algorithm 1) — gradient-accumulation-safe:
      0. Snapshot existing param.grad   (preserves accumulation from previous
         micro-batches in the current accumulation cycle).
      1. Forward on combined (train + val) batch.
      2. Backward on combined loss (retain_graph=True) → captures δ_t and δ_v
         via hooks; param.grad is contaminated by the scoring loss.
      3. Compute Φ(y_t) and the top-ρ mask.
      4. RESTORE the snapshotted param.grad (discards scoring contribution).
      5. Backward on masked training loss (Eq. 13) → ADDS the masked gradient
         on top of the restored accumulation.
    """
    B_tr = train_batch["input_ids"].shape[0]
    B_val = val_batch["input_ids"].shape[0]

    # Build combined batch
    combined_ids = torch.cat([train_batch["input_ids"], val_batch["input_ids"]], dim=0)
    combined_mask = torch.cat([train_batch["attention_mask"], val_batch["attention_mask"]], dim=0)
    combined_labels = torch.cat([train_batch["labels"], val_batch["labels"]], dim=0)

    hook_mgr.clear()

    # ── Step 0: snapshot grad state for accumulation safety ──────────────
    grad_snapshot = _save_grads(model)

    # ── Step 1: Combined forward ─────────────────────────────────────────
    with torch.autocast(device_type=device.type, dtype=dtype):
        outputs = model(
            input_ids=combined_ids,
            attention_mask=combined_mask,
            output_attentions=True,
        )
    logits = outputs.logits  # (B_tr + B_val, T, V)

    train_token_losses = per_token_ce_loss(logits[:B_tr], combined_labels[:B_tr])
    val_token_losses = per_token_ce_loss(logits[B_tr:], combined_labels[B_tr:])

    # ── LM shift alignment ───────────────────────────────────────────────
    # `per_token_ce_loss` returns losses at NETWORK positions: loss[b, t]
    # corresponds to predicting label at position t+1.  In paper notation,
    # the per-token loss ℓ_t = −log π(y_t | y_<t) maps to network position
    # t−1.  We therefore work with `loss_resp_mask` (shifted-by-one) instead
    # of the raw label-position response mask:
    #     loss_resp_mask[b, t] := "is y_{t+1} a response token?"
    # The same mask aligns Φ (defined at network positions), the val-side
    # signal filter (V_tgt is paper's response-token set), AND the final
    # masked training loss — all in one consistent index space.
    label_resp_mask = combined_labels != -100                  # paper positions
    loss_resp_mask = torch.zeros_like(label_resp_mask)
    loss_resp_mask[:, :-1] = label_resp_mask[:, 1:]            # network positions

    train_loss_resp_mask = loss_resp_mask[:B_tr]               # (B_tr, T)
    val_loss_resp_mask = loss_resp_mask[B_tr:]                 # (B_val, T)

    # ── Step 2: Scoring backward (signals only) ──────────────────────────
    # Σ over RESPONSE-TOKEN prediction losses, aligned to network positions.
    L_score = (
        train_token_losses[train_loss_resp_mask].sum()
        + val_token_losses[val_loss_resp_mask].sum()
    )
    with _no_sync_ctx(model):
        L_score.backward(retain_graph=True)

    # ── Step 3: Extract signals (val side filtered to RESPONSE tokens) ───
    seq_lengths_tr = train_batch["attention_mask"].sum(dim=1).tolist()
    seq_lengths_val = val_batch["attention_mask"].sum(dim=1).tolist()

    train_sig = _collect_flat_signals(
        hook_mgr,
        list(range(B_tr)),
        [int(s) for s in seq_lengths_tr],
        layer_indices,
        response_mask_2d=loss_resp_mask,           # shifted mask = network positions
        response_mask_offset=0,
        keep_only_response=False,                  # training side keeps all positions
    )
    val_sig = _collect_flat_signals(
        hook_mgr,
        list(range(B_tr, B_tr + B_val)),
        [int(s) for s in seq_lengths_val],
        layer_indices,
        response_mask_2d=loss_resp_mask,
        response_mask_offset=B_tr,
        keep_only_response=True,                   # V_tgt = response-token signals only
    )

    # Flat training response mask aligned to PHI positions (network indices)
    resp_mask_tr_flat = torch.cat([
        train_loss_resp_mask[b, :int(seq_lengths_tr[b])]
        for b in range(B_tr)
    ]).to(device)

    # Compute direct (A–A) and direct-proxy (A–P) terms
    N_train_flat = resp_mask_tr_flat.shape[0]
    phi_dir_tgt = torch.zeros(N_train_flat, device=device)
    phi_cau_tgt = torch.zeros(N_train_flat, device=device)
    phi_dir_prx = torch.zeros(N_train_flat, device=device)
    phi_cau_prx = torch.zeros(N_train_flat, device=device)

    for l_idx in layer_indices:
        for key in list(train_sig.acts.keys()):
            li, _ = key
            if li != l_idx or key not in train_sig.grads:
                continue

            h_tr = train_sig.acts[key]   # (N_train, d_in)
            d_tr = train_sig.grads[key]  # (N_train, d_out)

            # Direct Target (Eq. 8) — val signals are response-token only
            if key in val_sig.acts and key in val_sig.grads:
                h_val = val_sig.acts[key]
                d_val = val_sig.grads[key]
                if h_val.shape[0] > 0:
                    delta_cross = d_tr @ d_val.T
                    h_cross = h_tr @ h_val.T
                    phi_dir_tgt += (delta_cross * h_cross).mean(dim=1)

            # Direct Retention Proxy (Eq. 11)
            V_l = scorer._refresh_V(key, device).float()
            VH = h_tr @ V_l.T
            phi_dir_prx += (VH * d_tr).sum(dim=-1)

        # Causal terms (v_proj, per-sample attention)
        vproj_key = _find_vproj_key(l_idx, train_sig)
        if vproj_key is None or vproj_key not in train_sig.grads:
            continue
        if l_idx not in train_sig.attn_weights:
            continue
        if vproj_key not in val_sig.acts or vproj_key not in val_sig.grads:
            continue

        h_val_v = val_sig.acts[vproj_key]
        d_val_v = val_sig.grads[vproj_key]
        if h_val_v.shape[0] == 0:
            continue

        V_l_v = scorer._refresh_V(vproj_key, device).float()
        per_sample_attn = train_sig.attn_weights[l_idx]  # List[(T, T)]

        offset = 0
        for b in range(B_tr):
            sl = int(seq_lengths_tr[b])
            h_v = train_sig.acts[vproj_key][offset:offset + sl]
            d_v = train_sig.grads[vproj_key][offset:offset + sl]
            A_b = per_sample_attn[b][:sl, :sl]   # ← THIS sample's own attention
            cau_tgt_b, cau_prx_b = scorer.compute_sequence_level(
                h_v, d_v, h_val_v, d_val_v, A_b, V_l_v, device
            )
            phi_cau_tgt[offset:offset + sl] = cau_tgt_b
            phi_cau_prx[offset:offset + sl] = cau_prx_b
            offset += sl

    # Composite Φ (Eq. 12)
    phi = (
        phi_dir_tgt
        + phi_cau_tgt
        + scorer.lambda_stab * (phi_dir_prx + phi_cau_prx)
    ) * resp_mask_tr_flat.float()

    # Top-ρ mask on Φ — phi and resp_mask_tr_flat already live in network
    # (=loss) positions, so the resulting mask is the LOSS mask directly.
    mask_flat = top_rho_mask(phi, config.rho, resp_mask_tr_flat)

    mask_loss = torch.zeros_like(train_token_losses, dtype=torch.bool)
    offset = 0
    for b in range(B_tr):
        sl = int(seq_lengths_tr[b])
        mask_loss[b, :sl] = mask_flat[offset:offset + sl]
        offset += sl

    n_keep = mask_loss.sum().clamp(min=1)
    masked_loss = (train_token_losses * mask_loss.float()).sum() / n_keep

    # ── Step 4: discard scoring contribution; restore accumulated state ──
    _restore_grads(model, grad_snapshot)

    # ── Step 5: scale-by-accum + masked backward ─────────────────────────
    # Standard gradient-accumulation convention: each micro-batch contributes
    # 1/N of the global-batch gradient, so global-batch lr matches paper's
    # "effective batch 64" rather than scaling by accum_steps.
    masked_loss_scaled = masked_loss / float(config.gradient_accumulation_steps)
    masked_loss_scaled.backward()

    mean_phi = phi[resp_mask_tr_flat].mean() if resp_mask_tr_flat.any() else torch.tensor(0.0)
    return masked_loss, mean_phi


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_alpha_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_data: List[dict],
    config: AlphaSFTConfig,
):
    """
    Full AlphaSFT training loop with multi-GPU (DDP) support.

    Launch with `torchrun --nproc_per_node=4 train_sft.py ...` (see scripts/).
    The script auto-detects torchrun's env vars (LOCAL_RANK, RANK, WORLD_SIZE)
    and falls back to single-GPU if they are absent.
    """
    # ── Distributed setup (auto-detected from torchrun) ───────────────────
    if "LOCAL_RANK" in os.environ and not _is_dist():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    world = _world_size()

    model = model.to(device)
    model.train()

    # ── Prepare data (RANDOM split, per paper §5.1 / App. C.4) ────────────
    # Same shuffle seed on every rank ⇒ identical val / Fisher / train split.
    rng = random.Random(config.data_split_seed)
    indices = list(range(len(train_data)))
    rng.shuffle(indices)
    shuffled = [train_data[i] for i in indices]

    n_val = config.B_val
    val_data = shuffled[:n_val]
    fisher_data = shuffled[n_val:n_val + config.n_fisher_samples]
    train_data_actual = shuffled[n_val + config.n_fisher_samples:]

    val_dataset = SFTDataset(val_data, tokenizer, config.max_length)
    train_dataset = SFTDataset(train_data_actual, tokenizer, config.max_length)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world, rank=_rank(), shuffle=True, drop_last=True
    ) if _is_dist() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=lambda b: {k: torch.stack([s[k] for s in b]) for k in b[0]},
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.B_val,
        shuffle=True,
        collate_fn=lambda b: {k: torch.stack([s[k] for s in b]) for k in b[0]},
    )
    val_iter = iter(val_loader)

    # ── Fisher (rank-0 only; others wait and load the cache) ──────────────
    layer_indices = get_scoring_layer_indices(model, config.K)
    fisher_path = config.fisher_cache_path or os.path.join(config.output_dir, "fisher.pt")
    if _is_main():
        os.makedirs(config.output_dir, exist_ok=True)
        if os.path.exists(fisher_path):
            fisher = load_fisher(fisher_path, str(device))
        else:
            fisher_prompts = [
                ex.get("prompt", ex.get("instruction", "")) for ex in fisher_data
            ]
            fisher = compute_diagonal_fisher(
                model, tokenizer, fisher_prompts,
                K=config.K, device=str(device), dtype=dtype,
                max_prompt_len=256, max_new_tokens=128,
            )
            save_fisher(fisher, fisher_path)
    _barrier()
    if not _is_main():
        fisher = load_fisher(fisher_path, str(device))

    # ── Scorer and hooks (attached to the underlying model, BEFORE DDP) ──
    scorer = AlphaTokenScorer(
        model, fisher, layer_indices, config.lambda_stab, config.W
    )
    hook_mgr = HookManager(model, layer_indices)

    # ── Wrap with DDP after hooks (DDP transparently delegates forward) ───
    if _is_dist():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,        # scoring backward may not touch every param
            broadcast_buffers=False,
        )

    # ── Optimiser and scheduler ───────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = (len(train_loader) // max(1, config.gradient_accumulation_steps)) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training ──────────────────────────────────────────────────────────
    micro_step = 0
    update_step = 0
    accum_loss = 0.0

    for epoch in range(config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            train_batch = {k: v.to(device) for k, v in batch.items()
                           if k in ("input_ids", "attention_mask", "labels")}

            try:
                val_batch_raw = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch_raw = next(val_iter)
            val_batch = {k: v.to(device) for k, v in val_batch_raw.items()
                         if k in ("input_ids", "attention_mask", "labels")}

            loss, mean_phi = alpha_sft_loss(
                model, scorer, hook_mgr, train_batch, val_batch,
                config, device, dtype, layer_indices,
            )
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

                if _is_main() and update_step % config.logging_steps == 0:
                    avg = accum_loss / (config.logging_steps * config.gradient_accumulation_steps)
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[epoch {epoch+1} update {update_step}] "
                        f"loss={avg:.4f}  mean_phi={float(mean_phi):.4f}  lr={lr:.2e}"
                    )
                    accum_loss = 0.0

                if _is_main() and update_step % config.save_steps == 0:
                    ckpt = os.path.join(config.output_dir, f"checkpoint-{update_step}")
                    _unwrap_model(model).save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"Saved checkpoint to {ckpt}")
                _barrier()

    hook_mgr.remove()

    if _is_main():
        _unwrap_model(model).save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        print(f"Training complete. Model saved to {config.output_dir}")
    _barrier()
    if _is_dist():
        dist.destroy_process_group()
