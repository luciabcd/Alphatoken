"""
AlphaDPO Trainer — value-aware preference optimisation (Eq. 14 / Algorithm 1).

L_Alpha-DPO(θ) = E[ ω_sg · Σ_t ( m_t⁺ ℓ_t⁺(θ) − m_t⁻ ℓ_t⁻(θ) ) ]

where:
    ω        = β σ(−s),  s = β (log π_θ(y⁺|x)/π_ref(y⁺|x) − log π_θ(y⁻|x)/π_ref(y⁻|x))
    ω_sg     = stop-gradient of ω (sequence-level DPO coefficient)
    m_t±     = I[Φ(y_t±) ≥ τ_ρ±]   (top-ρ mask per branch)
    ℓ_t±(θ)  = −log π_θ(y_t± | x, y_{<t}±)

Per-token error signals scale as δ_t^l± = ±ω · δ̃_t^l±  (Sec. 3.5).
Both chosen and rejected branches are scored against the same validation batch
using Eq. (12) with the appropriately signed error signals.
"""

from __future__ import annotations

import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    save_fisher,
    load_fisher,
)
from .scoring import (
    AlphaTokenScorer,
    HookManager,
    LayerSignals,
    top_rho_mask,
    _find_vproj_key,
    _unwrap_model,
)
from .sft_trainer import (
    AlphaSFTConfig,
    SFTDataset,
    per_token_ce_loss,
    _collect_flat_signals,
    _save_grads,
    _restore_grads,
    _is_dist,
    _is_main,
    _rank,
    _world_size,
    _barrier,
    _no_sync_ctx,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AlphaDPOConfig:
    # AlphaToken
    rho: float = 0.5
    lambda_stab: float = 1.5
    K: int = 3
    W: int = 32
    B_val: int = 32
    n_fisher_samples: int = 1000

    # DPO
    beta: float = 0.1

    # Training  (paper: lr=2e-5 for Llama/Gemma, 1e-5 for Qwen)
    learning_rate: float = 2e-5
    num_epochs: int = 3
    per_device_train_batch_size: int = 2  # preference pairs
    gradient_accumulation_steps: int = 32
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"

    # Paths
    output_dir: str = "..."
    sft_warmstart_path: str = "..."   # warm-start SFT checkpoint (reference model)
    fisher_cache_path: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 200
    data_split_seed: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize_pref_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_len: int,
    max_response_len: int,
) -> dict:
    """
    Tokenize a preference triplet (prompt, chosen, rejected).
    Returns tensors for chosen and rejected sequences.
    """
    prompt = example.get("prompt", example.get("instruction", ""))
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def encode_pair(response: str):
        p_ids = tokenizer.encode(prompt, add_special_tokens=True, max_length=max_prompt_len, truncation=True)
        full = tokenizer.encode(prompt + response, add_special_tokens=True,
                                max_length=max_prompt_len + max_response_len, truncation=True)
        p_len = min(len(p_ids), len(full))
        labels = full.copy()
        for i in range(p_len):
            labels[i] = -100
        pad_len = (max_prompt_len + max_response_len) - len(full)
        ids = full + [pad_id] * pad_len
        attn = [1] * len(full) + [0] * pad_len
        lbl = labels + [-100] * pad_len
        max_len = max_prompt_len + max_response_len
        return {
            "input_ids": torch.tensor(ids[:max_len], dtype=torch.long),
            "attention_mask": torch.tensor(attn[:max_len], dtype=torch.long),
            "labels": torch.tensor(lbl[:max_len], dtype=torch.long),
        }

    chosen_enc = encode_pair(chosen)
    rejected_enc = encode_pair(rejected)

    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "chosen_labels": chosen_enc["labels"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
        "rejected_labels": rejected_enc["labels"],
    }


class PrefDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_len: int,
        max_response_len: int,
    ):
        self.samples = [
            _tokenize_pref_example(ex, tokenizer, max_prompt_len, max_response_len)
            for ex in data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Log-probability utilities
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ref_logprobs(
    ref_model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute sequence-level log π_ref(y|x) = Σ_t log π_ref(y_t|x, y<t).
    Returns (B,) tensor.
    """
    outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
    token_lp = -per_token_ce_loss(outputs.logits, labels)  # (B, T)
    resp_mask = labels != -100
    # Align: per_token_ce_loss[b, t] predicts labels[b, t+1]
    loss_mask = torch.zeros_like(resp_mask)
    loss_mask[:, :-1] = resp_mask[:, 1:]
    return (token_lp * loss_mask.float()).sum(dim=-1)  # (B,)


def compute_policy_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Σ_t log π_θ(y_t|x, y<t) from policy logits.
    Returns (B,) tensor.
    """
    token_lp = -per_token_ce_loss(logits, labels)  # (B, T)
    resp_mask = labels != -100
    loss_mask = torch.zeros_like(resp_mask)
    loss_mask[:, :-1] = resp_mask[:, 1:]
    return (token_lp * loss_mask.float()).sum(dim=-1)  # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# AlphaDPO step
# ──────────────────────────────────────────────────────────────────────────────

def alpha_dpo_loss(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    scorer: AlphaTokenScorer,
    hook_mgr: HookManager,
    train_batch: dict,
    val_batch: dict,
    config: AlphaDPOConfig,
    device: torch.device,
    dtype: torch.dtype,
    layer_indices: List[int],
) -> Tuple[torch.Tensor, dict]:
    """
    One AlphaDPO step.  Returns (masked_loss, info_dict).

    The DPO gradient scales per-token errors as δ_t± = ±ω δ̃_t±.
    We capture these scaled signals and use them for GDP scoring
    (Sec. 3.5 / Eq. 14).
    """
    B = train_batch["chosen_input_ids"].shape[0]
    max_len = train_batch["chosen_input_ids"].shape[1]
    B_val = val_batch["input_ids"].shape[0]

    ch_ids = train_batch["chosen_input_ids"].to(device)
    ch_mask = train_batch["chosen_attention_mask"].to(device)
    ch_lbls = train_batch["chosen_labels"].to(device)
    rj_ids = train_batch["rejected_input_ids"].to(device)
    rj_mask = train_batch["rejected_attention_mask"].to(device)
    rj_lbls = train_batch["rejected_labels"].to(device)
    val_ids = val_batch["input_ids"].to(device)
    val_mask = val_batch["attention_mask"].to(device)
    val_lbls = val_batch["labels"].to(device)

    # ── Reference log-probs (no grad) ────────────────────────────────────
    ref_lp_ch = compute_ref_logprobs(ref_model, ch_ids, ch_mask, ch_lbls)
    ref_lp_rj = compute_ref_logprobs(ref_model, rj_ids, rj_mask, rj_lbls)

    # ── Policy forward: chosen + rejected + validation (combined batch) ───
    combined_ids = torch.cat([ch_ids, rj_ids, val_ids], dim=0)
    combined_attn = torch.cat([ch_mask, rj_mask, val_mask], dim=0)
    combined_lbls = torch.cat([ch_lbls, rj_lbls, val_lbls], dim=0)

    hook_mgr.clear()

    # Snapshot grad state for accumulation safety — scoring backward will
    # contaminate param.grad; we restore before the actual DPO backward so
    # micro-batches across `gradient_accumulation_steps` accumulate correctly.
    grad_snapshot = _save_grads(model)

    with torch.autocast(device_type=device.type, dtype=dtype):
        outputs = model(
            input_ids=combined_ids,
            attention_mask=combined_attn,
            output_attentions=True,
        )
    logits = outputs.logits  # (2B + B_val, T, V)

    lp_ch = compute_policy_logprobs(logits[:B], ch_lbls)       # (B,)
    lp_rj = compute_policy_logprobs(logits[B:2*B], rj_lbls)    # (B,)

    # DPO logit ratio s and ω = β σ(-s)   (per preference pair)
    s = config.beta * ((lp_ch - ref_lp_ch) - (lp_rj - ref_lp_rj))   # (B,)
    omega = config.beta * torch.sigmoid(-s)                            # (B,)
    omega_sg = omega.detach()                                          # stop-gradient

    # Per-token CE losses for chosen and rejected response tokens
    ch_token_loss = per_token_ce_loss(logits[:B], ch_lbls)     # (B, T)
    rj_token_loss = per_token_ce_loss(logits[B:2*B], rj_lbls)  # (B, T)
    val_token_loss = per_token_ce_loss(logits[2*B:], val_lbls)  # (B_val, T)

    # ── LM shift alignment (loss positions) ──────────────────────────────
    # loss[b, t] predicts label at t+1.  Build "loss-position" response masks
    # so Φ, val filtering, and the masked loss share one consistent index space.
    def _to_loss_mask(label_mask: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(label_mask)
        out[:, :-1] = label_mask[:, 1:]
        return out

    ch_resp_mask = ch_lbls != -100               # label positions
    rj_resp_mask = rj_lbls != -100
    val_resp_mask = val_lbls != -100
    ch_loss_mask = _to_loss_mask(ch_resp_mask)   # network / loss positions
    rj_loss_mask = _to_loss_mask(rj_resp_mask)
    val_loss_mask = _to_loss_mask(val_resp_mask)

    # ── Backward for scoring signals: scaled DPO gradient ────────────────
    # δ_t± = ±ω δ̃_t±   →   multiply per-token losses by ±ω before backward.
    # ω_sg acts as a constant multiplier so we capture the DPO-scaled δ at hooks.
    L_ch_score = (omega_sg.unsqueeze(1) * ch_token_loss * ch_loss_mask.float()).sum()
    L_rj_score = -(omega_sg.unsqueeze(1) * rj_token_loss * rj_loss_mask.float()).sum()
    L_val_score = val_token_loss[val_loss_mask].sum()

    with _no_sync_ctx(model):
        (L_ch_score + L_rj_score + L_val_score).backward(retain_graph=True)

    # ── Extract signals (val side filtered to RESPONSE tokens only) ──────
    seq_len_ch = ch_mask.sum(dim=1).tolist()
    seq_len_rj = rj_mask.sum(dim=1).tolist()
    seq_len_val = val_mask.sum(dim=1).tolist()

    combined_loss_mask = torch.cat([ch_loss_mask, rj_loss_mask, val_loss_mask], dim=0)

    ch_sig = _collect_flat_signals(
        hook_mgr, list(range(B)), [int(s) for s in seq_len_ch], layer_indices,
        response_mask_2d=combined_loss_mask, response_mask_offset=0,
        keep_only_response=False,
    )
    rj_sig = _collect_flat_signals(
        hook_mgr, list(range(B, 2*B)), [int(s) for s in seq_len_rj], layer_indices,
        response_mask_2d=combined_loss_mask, response_mask_offset=B,
        keep_only_response=False,
    )
    val_sig = _collect_flat_signals(
        hook_mgr, list(range(2*B, 2*B + B_val)), [int(s) for s in seq_len_val], layer_indices,
        response_mask_2d=combined_loss_mask, response_mask_offset=2*B,
        keep_only_response=True,
    )

    def _compute_phi(
        branch_sig: LayerSignals,
        seq_lengths: List[int],
        resp_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """GDP scoring for one branch (chosen or rejected)."""
        N_flat = sum(int(s) for s in seq_lengths)
        phi_dt = torch.zeros(N_flat, device=device)
        phi_ct = torch.zeros(N_flat, device=device)
        phi_dp = torch.zeros(N_flat, device=device)
        phi_cp = torch.zeros(N_flat, device=device)

        resp_flat = torch.cat([
            resp_mask_2d[b, :int(seq_lengths[b])]
            for b in range(len(seq_lengths))
        ]).to(device)

        for l_idx in layer_indices:
            for key in list(branch_sig.acts.keys()):
                li, name = key
                if li != l_idx or key not in branch_sig.grads:
                    continue
                h_tr = branch_sig.acts[key].float()
                d_tr = branch_sig.grads[key].float()

                if key in val_sig.acts and key in val_sig.grads:
                    h_val = val_sig.acts[key].float()
                    d_val = val_sig.grads[key].float()
                    delta_cross = d_tr @ d_val.T
                    h_cross = h_tr @ h_val.T
                    phi_dt += (delta_cross * h_cross).mean(dim=1)

                V_l = scorer._refresh_V(key, device).float()
                VH = h_tr @ V_l.T
                phi_dp += (VH * d_tr).sum(dim=-1)

            vproj_key = _find_vproj_key(l_idx, branch_sig)
            if vproj_key is None or vproj_key not in branch_sig.grads:
                continue
            if l_idx not in branch_sig.attn_weights:
                continue
            if vproj_key not in val_sig.acts or vproj_key not in val_sig.grads:
                continue

            h_val_v = val_sig.acts[vproj_key].float()
            d_val_v = val_sig.grads[vproj_key].float()
            if h_val_v.shape[0] == 0:
                continue
            V_l = scorer._refresh_V(vproj_key, device).float()
            per_sample_attn = branch_sig.attn_weights[l_idx]   # List[(T, T)]

            offset = 0
            for b, sl in enumerate(seq_lengths):
                sl = int(sl)
                h_v = branch_sig.acts[vproj_key][offset:offset + sl]
                d_v = branch_sig.grads[vproj_key][offset:offset + sl]
                A_b = per_sample_attn[b][:sl, :sl]  # per-sample attention
                ct_b, cp_b = scorer.compute_sequence_level(
                    h_v.float(), d_v.float(), h_val_v, d_val_v, A_b.float(), V_l, device
                )
                phi_ct[offset:offset + sl] = ct_b
                phi_cp[offset:offset + sl] = cp_b
                offset += sl

        phi = (
            phi_dt + phi_ct
            + scorer.lambda_stab * (phi_dp + phi_cp)
        ) * resp_flat.float()
        return phi, resp_flat

    # phi / resp_*_flat now live in loss/network positions — no extra shift.
    phi_ch, resp_ch_flat = _compute_phi(ch_sig, seq_len_ch, ch_loss_mask)
    phi_rj, resp_rj_flat = _compute_phi(rj_sig, seq_len_rj, rj_loss_mask)

    mask_ch_flat = top_rho_mask(phi_ch, config.rho, resp_ch_flat)
    mask_rj_flat = top_rho_mask(phi_rj, config.rho, resp_rj_flat)

    # Reshape flat masks to (B, T) directly as LOSS masks.
    def _flat_to_2d(mask_flat, seq_lengths, B, T):
        mask_2d = torch.zeros(B, T, dtype=torch.bool, device=device)
        offset = 0
        for b, sl in enumerate(seq_lengths):
            sl = int(sl)
            mask_2d[b, :sl] = mask_flat[offset:offset + sl]
            offset += sl
        return mask_2d

    mask_ch_loss = _flat_to_2d(mask_ch_flat, seq_len_ch, B, max_len)
    mask_rj_loss = _flat_to_2d(mask_rj_flat, seq_len_rj, B, max_len)

    # ── Masked AlphaDPO loss (Eq. 14) ─────────────────────────────────────
    ch_masked = (ch_token_loss * mask_ch_loss.float()).sum(dim=-1)   # (B,)
    rj_masked = (rj_token_loss * mask_rj_loss.float()).sum(dim=-1)   # (B,)

    n_ch = mask_ch_loss.sum(dim=-1).clamp(min=1).float()             # (B,)
    n_rj = mask_rj_loss.sum(dim=-1).clamp(min=1).float()             # (B,)

    # L_Alpha-DPO = ω_sg · (Σ_t m_t⁺ ℓ_t⁺ − Σ_t m_t⁻ ℓ_t⁻)  (per pair, then mean)
    per_pair_loss = omega_sg * (ch_masked / n_ch - rj_masked / n_rj)
    dpo_loss = per_pair_loss.mean()

    # Discard the scoring backward's contribution, restore accumulated grads,
    # then add this micro-batch's DPO gradient (scaled by 1/accum_steps) on top.
    _restore_grads(model, grad_snapshot)
    (dpo_loss / float(config.gradient_accumulation_steps)).backward()

    info = {
        "loss": dpo_loss.item(),
        "omega_mean": omega_sg.mean().item(),
        "chosen_reward": (lp_ch - ref_lp_ch).mean().item(),
        "rejected_reward": (lp_rj - ref_lp_rj).mean().item(),
        "phi_ch_mean": phi_ch[resp_ch_flat].mean().item() if resp_ch_flat.any() else 0.0,
        "phi_rj_mean": phi_rj[resp_rj_flat].mean().item() if resp_rj_flat.any() else 0.0,
    }
    return dpo_loss, info


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_alpha_dpo(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_data: List[dict],
    val_sft_data: List[dict],
    config: AlphaDPOConfig,
):
    """
    Full AlphaDPO training loop.

    model:         policy model (to be fine-tuned)
    ref_model:     frozen SFT warm-start reference model
    train_data:    list of {'prompt', 'chosen', 'rejected'} dicts
    val_sft_data:  validation examples (SFT format) for scoring signals
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

    model = model.to(device).train()
    ref_model = ref_model.to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Prepare data (random split) ───────────────────────────────────────
    # Per the paper, the preference-alignment validation subset and the Fisher
    # prompts both come from the CURRENT training distribution (UltraFeedback,
    # in SFT format).  We expect `val_sft_data` to be SFT-format triples derived
    # from the same UltraFeedback split that produced `train_data`, and
    # excluded from preference optimisation.
    rng = random.Random(config.data_split_seed)
    indices = list(range(len(val_sft_data)))
    rng.shuffle(indices)
    shuffled = [val_sft_data[i] for i in indices]

    n_val_score = config.B_val
    val_score_data = shuffled[:n_val_score]
    fisher_data = shuffled[n_val_score:n_val_score + config.n_fisher_samples]

    pref_dataset = PrefDataset(
        train_data, tokenizer, config.max_prompt_length, config.max_response_length
    )
    from .sft_trainer import SFTDataset as _SFTDataset
    val_score_dataset = _SFTDataset(
        val_score_data, tokenizer, config.max_prompt_length + config.max_response_length
    )

    train_sampler = DistributedSampler(
        pref_dataset, num_replicas=world, rank=_rank(), shuffle=True, drop_last=True
    ) if _is_dist() else None
    train_loader = DataLoader(
        pref_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=lambda b: {k: torch.stack([s[k] for s in b]) for k in b[0]},
    )
    val_loader = DataLoader(
        val_score_dataset,
        batch_size=config.B_val,
        shuffle=True,
        collate_fn=lambda b: {k: torch.stack([s[k] for s in b]) for k in b[0]},
    )
    val_iter = iter(val_loader)

    # ── Fisher (anchored at the REFERENCE model, not the policy) ─────────
    # The Fisher-drift proxy g_prx = F_ref · (θ − θ_ref) is anchored at the
    # pre-trained / warm-start reference (θ_ref).  Both F_ref AND the cached
    # W_ref weights must therefore come from `ref_model`, not the (mutable)
    # policy `model`.  We compute Fisher with ref_model, and the scorer below
    # snapshots W_ref from ref_model as well.
    layer_indices = get_scoring_layer_indices(ref_model, config.K)
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
                ref_model, tokenizer, fisher_prompts,
                K=config.K, device=str(device), dtype=dtype,
            )
            save_fisher(fisher, fisher_path)
    _barrier()
    if not _is_main():
        fisher = load_fisher(fisher_path, str(device))

    # Build the scorer so it tracks θ (policy) but anchors W_ref to ref_model.
    scorer = AlphaTokenScorer(
        model, fisher, layer_indices, config.lambda_stab, config.W
    )
    # Overwrite W_ref with the FROZEN reference model's weights.
    ref_base = _unwrap_model(ref_model)
    for l_idx in layer_indices:
        ref_layer = ref_base.model.layers[l_idx]
        for name, mod in ref_layer.named_modules():
            if isinstance(mod, nn.Linear):
                scorer.w_ref[(l_idx, name)] = mod.weight.detach().clone()

    # Hooks on the underlying model BEFORE wrapping with DDP.
    hook_mgr = HookManager(model, layer_indices)

    if _is_dist():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────
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

    for epoch in range(config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            try:
                val_raw = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_raw = next(val_iter)

            val_batch = {k: v.to(device) for k, v in val_raw.items()
                         if k in ("input_ids", "attention_mask", "labels")}

            loss, info = alpha_dpo_loss(
                model, ref_model, scorer, hook_mgr, batch, val_batch,
                config, device, dtype, layer_indices,
            )
            micro_step += 1

            if micro_step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

                if _is_main() and update_step % config.logging_steps == 0:
                    print(
                        f"[epoch {epoch+1} update {update_step}] "
                        f"loss={info['loss']:.4f}  "
                        f"chosen_rw={info['chosen_reward']:.3f}  "
                        f"rejected_rw={info['rejected_reward']:.3f}  "
                        f"omega={info['omega_mean']:.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

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
        print(f"DPO training complete. Model saved to {config.output_dir}")
    _barrier()
    if _is_dist():
        dist.destroy_process_group()
