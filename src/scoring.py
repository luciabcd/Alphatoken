"""
AlphaToken token valuation — Ghost Dot-Product (GDP) family.

Implements Equations (8), (9), (11) and (12) from the paper.

Composite token value (Eq. 12):
    Φ(y_t) = Φ_dir_tgt(y_t)             # direct  target  alignment  (A–A)
            + Φ_cau_tgt(y_t)             # causal  target  alignment  (A–A causal)
            + λ * Φ_dir_prx(y_t)         # direct  retention proxy    (A–P)
            + λ * Φ_cau_prx(y_t)         # causal  retention proxy    (A–P causal)

All four terms share one cached forward/backward pass.
The causal terms use the Value-Propagation approximation (Sec. 3.2 / App. B.3)
which retains only the dominant α·W_V channel and is tight under both
sparse and saturated attention.

References to equations use the paper's numbering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .fisher import get_linear_modules, get_scoring_layer_indices


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying causal-LM through DDP / DataParallel / FSDP wrappers."""
    inner = model
    # peel until we expose `.model.layers`
    for _ in range(4):
        if hasattr(inner, "module"):
            inner = inner.module
        else:
            break
    return inner


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSignals:
    """
    Captured per-layer, per-token activations and error signals.

    Two stages share this container:
      - HookManager.signals.attn_weights[l_idx]:        raw (B, H, T, T) tensor
      - flat signals (after _collect_flat_signals):     List[Tensor] of length B,
        each (T, T) head-averaged.  Eq. 9 uses each pair's OWN attention matrix,
        so we never reduce away the batch dimension before scoring.
    """
    acts: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)
    grads: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)
    attn_weights: Dict[int, object] = field(default_factory=dict)

    def clear(self):
        self.acts.clear()
        self.grads.clear()
        self.attn_weights.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Hook manager
# ──────────────────────────────────────────────────────────────────────────────

class HookManager:
    """
    Registers forward + backward hooks on all linear layers in the last K
    transformer layers and on attention modules (for attention weights).
    Captures (h, δ) pairs and attention weights during forward/backward.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
    ):
        self.signals = LayerSignals()
        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._register(model, layer_indices)

    def _register(self, model: nn.Module, layer_indices: List[int]):
        # Hooks must attach to the UNDERLYING modules, not the DDP wrapper,
        # otherwise they fire on the wrapper's forward (no submodule access).
        base = _unwrap_model(model)
        for l_idx in layer_indices:
            layer = base.model.layers[l_idx]

            # Linear layer hooks
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    key = (l_idx, name)
                    self._handles.append(
                        mod.register_forward_hook(self._fwd_hook(key))
                    )
                    self._handles.append(
                        mod.register_full_backward_hook(self._bwd_hook(key))
                    )

            # Attention module hook (for attention weights)
            self._handles.append(
                layer.self_attn.register_forward_hook(self._attn_hook(l_idx))
            )

    def _fwd_hook(self, key: Tuple[int, str]):
        def hook(module, inp, out):
            # inp[0]: (B, T, d_in)
            self.signals.acts[key] = inp[0].detach()
        return hook

    def _bwd_hook(self, key: Tuple[int, str]):
        def hook(module, grad_in, grad_out):
            # grad_out[0] = ∂L/∂(output) = δ,  (B, T, d_out)
            if grad_out[0] is not None:
                self.signals.grads[key] = grad_out[0].detach()
        return hook

    def _attn_hook(self, l_idx: int):
        def hook(module, inp, out):
            # Most HF models: out = (hidden_states, attn_weights, past_kv, ...)
            # Store FULL (B, H, T, T) — per-sample slicing is done downstream
            # so each batch member uses its own attention matrix in Eq. 9.
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                self.signals.attn_weights[l_idx] = out[1].detach()  # raw (B, H, T, T)
        return hook

    def clear(self):
        self.signals.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ──────────────────────────────────────────────────────────────────────────────
# GDP scoring
# ──────────────────────────────────────────────────────────────────────────────

def _causal_window_mask(T: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Build boolean mask A_mask[k, t] = 1  iff  0 < k - t <= W  (i.e., k > t, k - t <= W).
    Shape: (T, T).  Row k, column t: future position k attends back to position t.
    """
    rows = torch.arange(T, device=device).unsqueeze(1)   # (T, 1)
    cols = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
    diff = rows - cols
    return (diff > 0) & (diff <= W)  # (T, T)


def _find_vproj_key(
    layer_idx: int,
    signals: LayerSignals,
) -> Optional[Tuple[int, str]]:
    """Find the v_proj / value_proj key in signals for a given layer."""
    for key in signals.acts:
        li, name = key
        if li == layer_idx and "v_proj" in name:
            return key
    # Fallback: look for value-like names
    for key in signals.acts:
        li, name = key
        if li == layer_idx and ("value" in name.lower() or "v_proj" in name.lower()):
            return key
    return None


class AlphaTokenScorer:
    """
    Computes per-token composite values Φ(y_t) (Eq. 12) using the GDP family.

    Usage (SFT example):
        scorer = AlphaTokenScorer(model, fisher, layer_indices, lambda_stab, W, K)
        # combined forward/backward captured by hooks
        phi = scorer.compute(train_sig, val_sig, response_mask_train)
        mask = scorer.top_rho_mask(phi, rho, response_mask_train)
    """

    def __init__(
        self,
        model: nn.Module,
        fisher: Dict[Tuple[int, str], torch.Tensor],
        layer_indices: List[int],
        lambda_stab: float = 1.5,
        causal_window: int = 32,
    ):
        self.model = model
        self.fisher = fisher
        self.layer_indices = layer_indices
        self.lambda_stab = lambda_stab
        self.W = causal_window

        # Cache W_ref for each linear layer in S (captured once at init).
        # W_ref MUST come from the reference checkpoint at init time; for DPO
        # the DPO trainer overwrites these with `ref_model`'s weights.
        base = _unwrap_model(model)
        self.w_ref: Dict[Tuple[int, str], torch.Tensor] = {}
        for l_idx in layer_indices:
            layer = base.model.layers[l_idx]
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    self.w_ref[(l_idx, name)] = mod.weight.detach().clone()

    # ── per-step refresh ────────────────────────────────────────────────────

    def _refresh_V(self, key: Tuple[int, str], device: torch.device) -> torch.Tensor:
        """
        V_l = F_Wl ⊙ (W_l − W_ref_l)   (Eq. 10, refreshed each step)
        Shape: (d_out, d_in)
        """
        l_idx, name = key
        base = _unwrap_model(self.model)
        layer = base.model.layers[l_idx]
        w_curr = None
        for n, m in layer.named_modules():
            if n == name and isinstance(m, nn.Linear):
                w_curr = m.weight.detach()
                break
        if w_curr is None:
            raise KeyError(f"Layer {key} not found in model.")

        f_wl = self.fisher.get(key)
        if f_wl is None:
            return torch.zeros_like(w_curr)

        w_ref = self.w_ref[key].to(device)
        return f_wl.to(device) * (w_curr - w_ref)  # (d_out, d_in)

    # ── main scoring ────────────────────────────────────────────────────────
    #
    # NOTE: there used to be a `.compute(...)` method that aggregated direct
    # terms over a flat batch and stubbed out the causal path.  It is removed
    # because Eq. 9 requires per-sample positional structure (each batch
    # member has its OWN attention matrix and length), which the flat-batch
    # API cannot express.  The trainers in `sft_trainer.py` / `dpo_trainer.py`
    # orchestrate the four scoring terms directly using `_refresh_V` and
    # `compute_sequence_level` below — that is the supported entry point.

    def compute_sequence_level(
        self,
        h_tr_v: torch.Tensor,        # (T_train, d_in)  — v_proj input, training seq
        d_tr_v: torch.Tensor,        # (T_train, d_out) — v_proj grad, training seq
        h_val_v: torch.Tensor,       # (T_val, d_in)
        d_val_v: torch.Tensor,       # (T_val, d_out)
        A: torch.Tensor,             # (T_train, T_train) — averaged attention weights
        V_l: torch.Tensor,           # (d_out, d_in) — Fisher-drift proxy for v_proj
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal target (Eq. 9) and causal retention proxy for ONE layer,
        for a single training sequence of length T_train.

        Returns:
            phi_cau_tgt: (T_train,)
            phi_cau_prx: (T_train,)
        """
        T = h_tr_v.shape[0]
        window_mask = _causal_window_mask(T, self.W, device).float()  # (T, T)
        A_masked = A * window_mask   # (T, T)

        # ── Causal Target (Eq. 9) ────────────────────────────────────────
        # Φ_cau_tgt[t] = (1/N_val) Σ_v Σ_{k>t,k-t≤W} A[k,t] ⟨δ_k, δ_v⟩ ⟨h_t, h_v⟩
        delta_cross = d_tr_v @ d_val_v.T   # (T, N_val)
        # A_masked.T[t, k] = A_masked[k, t] → weight of future k affecting t
        weighted_delta = A_masked.T @ delta_cross  # (T, N_val)
        h_cross = h_tr_v @ h_val_v.T               # (T, N_val)
        phi_cau_tgt = (weighted_delta * h_cross).mean(dim=-1)  # (T,)

        # ── Causal Retention Proxy ───────────────────────────────────────
        # Φ_cau_prx[t] = Σ_{k>t,k-t≤W} A[k,t] (δ_k)^T V_l h_t
        VH = h_tr_v @ V_l.T                  # (T, d_out)
        # For each t, sum A[k,t]*δ_k over k in window
        A_delta = A_masked.T @ d_tr_v         # (T, d_out)
        phi_cau_prx = (VH * A_delta).sum(dim=-1)  # (T,)

        return phi_cau_tgt, phi_cau_prx


# ──────────────────────────────────────────────────────────────────────────────
# Masking
# ──────────────────────────────────────────────────────────────────────────────

def top_rho_mask(
    phi: torch.Tensor,
    rho: float,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Within-batch top-ρ threshold (Sec. 3.5).
    Returns binary mask: True for response tokens with Φ(y_t) ≥ τ_ρ.

    phi:           (N,) composite token values
    rho:           fraction of response tokens to retain (0 < ρ ≤ 1)
    response_mask: (N,) bool — True for response positions

    τ_ρ is computed over response tokens only so each batch retains exactly ρ.
    """
    phi_response = phi[response_mask]
    if phi_response.numel() == 0:
        return torch.zeros_like(response_mask)

    # quantile(1 - rho) gives the threshold such that top-ρ fraction is kept
    q = max(0.0, min(1.0, 1.0 - rho))
    tau = torch.quantile(phi_response.float(), q)
    mask = (phi >= tau) & response_mask
    return mask


def extract_flat_signals(
    batch_signals: LayerSignals,
    layer_idx: int,
    proj_name: str,
    seq_idx: int,
    seq_len: int,
    offset: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract (h, delta) for a specific sequence from a batched signal tensor.

    batch_signals: holds (B, T, d) tensors
    seq_idx:       index in batch dimension
    seq_len:       actual (unpadded) length of this sequence
    offset:        token offset within the flattened representation

    Returns (h, delta) each of shape (seq_len, d), or (None, None) if missing.
    """
    key = (layer_idx, proj_name)
    if key not in batch_signals.acts or key not in batch_signals.grads:
        return None, None

    acts = batch_signals.acts[key]
    grads = batch_signals.grads[key]

    if acts.dim() == 3:
        h = acts[seq_idx, :seq_len]
        d = grads[seq_idx, :seq_len] if grads.dim() == 3 else grads[offset:offset + seq_len]
    else:
        h = acts[offset:offset + seq_len]
        d = grads[offset:offset + seq_len]

    return h, d
