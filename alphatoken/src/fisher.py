"""
Diagonal Monte-Carlo Fisher information (Eq. 5 / Sec. 3.3).

F_ref = Diag( E_{x~X, y~p_θref(·|x)} [ ∇_θ log p_θref(y|x)^⊙2 ] )

For each linear layer W_l ∈ R^{d_out × d_in}:
    F_Wl[i,j] = E[ δ_i² · h_j² ]
where δ = ∂ℓ/∂(W h) is the output error signal, h is the layer input.
Stored as a matrix of shape (d_out, d_in).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    """Peel DDP / DataParallel wrappers."""
    inner = model
    for _ in range(4):
        if hasattr(inner, "module"):
            inner = inner.module
        else:
            break
    return inner


def get_scoring_layer_indices(model: PreTrainedModel, K: int) -> List[int]:
    """Return the indices of the last K transformer layers (the 'layer set S')."""
    base = _unwrap(model)
    n = len(base.model.layers)
    return list(range(max(0, n - K), n))


def get_linear_modules(
    model: PreTrainedModel,
    layer_indices: List[int],
) -> Dict[Tuple[int, str], nn.Linear]:
    """
    Collect all nn.Linear modules in the given transformer-layer indices.
    Excludes embed_tokens and lm_head (per the paper: 'embedding and lm_head
    layers are excluded from the scoring layer set S').
    """
    base = _unwrap(model)
    result: Dict[Tuple[int, str], nn.Linear] = {}
    for l_idx in layer_indices:
        layer = base.model.layers[l_idx]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                result[(l_idx, name)] = module
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Fisher computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_diagonal_fisher(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    K: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_prompt_len: int = 256,
    max_new_tokens: int = 128,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Compute the diagonal Monte-Carlo Fisher for all linear layers in the last K
    transformer layers.

    Algorithm:
        For each prompt x in prompts:
            1. Sample ỹ ~ p_θref(·|x)  (greedy / temperature=1.0)
            2. Compute  ∇_θ log p_θref(ỹ|x)  via backward on NLL loss
            3. Accumulate  F_Wl += (δ²)ᵀ @ (h²)  per layer

    Returns dict: {(layer_idx, proj_name): F_Wl  (float32, d_out × d_in)}
    """
    model.eval()
    layer_indices = get_scoring_layer_indices(model, K)
    linear_mods = get_linear_modules(model, layer_indices)

    # Accumulators (float32 for numerical stability)
    fisher: Dict[Tuple[int, str], torch.Tensor] = {
        key: torch.zeros(mod.weight.shape, dtype=torch.float32, device=device)
        for key, mod in linear_mods.items()
    }

    # Per-step activation / gradient buffers (cleared each sample)
    _acts: Dict[Tuple[int, str], torch.Tensor] = {}
    _grads: Dict[Tuple[int, str], torch.Tensor] = {}

    # Register hooks
    handles = []
    for key, mod in linear_mods.items():
        def _fwd(k):
            def hook(module, inp, out):
                _acts[k] = inp[0].detach().float()  # (B, T, d_in)
            return hook

        def _bwd(k):
            def hook(module, grad_in, grad_out):
                # grad_out[0] = ∂L/∂(output) = δ,  shape (B, T, d_out)
                if grad_out[0] is not None:
                    _grads[k] = grad_out[0].detach().float()
            return hook

        handles.append(mod.register_forward_hook(_fwd(key)))
        handles.append(mod.register_full_backward_hook(_bwd(key)))

    n_ok = 0
    for prompt in prompts:
        try:
            _acts.clear()
            _grads.clear()
            model.zero_grad()

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_prompt_len,
                truncation=True,
            ).to(device)

            # Sample completion from the reference model
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_len = enc["input_ids"].shape[1]
            full_ids = generated  # (1, prompt_len + response_len)

            # Labels: -100 for prompt (masked), token ids for response
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

            # Forward + backward to capture Fisher signals.
            # Paper's F_ref = E[∇ log p(y|x)^⊙2] uses the SUM-of-log-probs gradient,
            # NOT the mean-CE gradient HF returns in `out.loss`.  Hence we compute
            # the explicit sum-of-token-NLL and backward on that to preserve scale.
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
                out = model(input_ids=full_ids)
                shift_logits = out.logits[:, :-1].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                nll_sum = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

            # ∇ log p = -∇ NLL; F uses squared gradients so sign doesn't matter.
            nll_sum.backward()

            # Accumulate F_Wl += (δ²)ᵀ @ (h²)  (summed over tokens in sequence)
            for key in list(linear_mods.keys()):
                if key not in _acts or key not in _grads:
                    continue
                h = _acts[key]      # (1, T, d_in) → (T, d_in)
                delta = _grads[key]  # (1, T, d_out) → (T, d_out)
                h_flat = h.reshape(-1, h.shape[-1])
                d_flat = delta.reshape(-1, delta.shape[-1])
                # (d_out, T) @ (T, d_in) → element-wise-squared outer product sum
                fisher[key].add_((d_flat ** 2).T @ (h_flat ** 2))

            n_ok += 1

        except Exception as exc:
            print(f"[Fisher] skipping prompt ({exc})")
        finally:
            model.zero_grad()

    for h in handles:
        h.remove()

    if n_ok == 0:
        raise RuntimeError("Fisher computation: no prompts succeeded.")

    for key in fisher:
        fisher[key] /= n_ok
        fisher[key] = fisher[key].to(dtype)

    print(f"[Fisher] computed on {n_ok}/{len(prompts)} prompts, {len(fisher)} layers.")
    return fisher


def save_fisher(fisher: Dict, path: str) -> None:
    torch.save(fisher, path)
    print(f"[Fisher] saved to {path}")


def load_fisher(path: str, device: str = "cuda") -> Dict:
    fisher = torch.load(path, map_location=device)
    print(f"[Fisher] loaded from {path}")
    return fisher
