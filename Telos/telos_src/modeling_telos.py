#!/usr/bin/env python3
"""Telos (`Telos-Small`, 50.1M Parameters): Standalone Time-Series Joint-Embedding Predictive Architecture (JEPA).

Architecture highlights:
  1. Causal per-patch Welford mean/std normalization with `asinh` compression and
     `advancing_block_frames` multi-patch horizon refinement.
  2. 12-block Two-Stream `[Enc ; Dec]` Transformer (`d_model=512`, `num_heads=8`,
     `[time, time, time, time_variate] x 3`) with `SinCosRoPE`, `sdpa_sigmoid` gated
     time/variate attention, U-Net skip connections, Value-Residual attention, and `FusedSwiGLU`.
  3. 2-block Horizon-Conditioned `JEPALatentPredictor` predicting future patch representations
     in latent space across up to `32` patches (`1,024` steps) in a single forward shot.
  4. 21-knot monotone `QuantileHead` (`0.01 .. 0.99`) with full support for univariate,
     multivariate, past covariates, and known-future covariates.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

try:
    from .configuration_telos import QUANTILE_KNOTS_21, TelosConfig
except ImportError:
    from configuration_telos import QUANTILE_KNOTS_21, TelosConfig  # type: ignore

ROLE_TARGET = 0
ROLE_PAST_COV = 1
ROLE_KNOWN_COV = 2

LAYER_KIND_AXES: Dict[str, Tuple[bool, ...]] = {
    "time": (True,),
    "variate": (False,),
    "time_variate": (True, False),
}


# ==============================================================================
# 1. Causal Welford & Asinh Scaler Primitives
# ==============================================================================


def _fp32_io(*dtypes: torch.dtype) -> Tuple[torch.dtype, torch.dtype]:
    out_dtype = functools.reduce(torch.promote_types, dtypes)
    return out_dtype, torch.promote_types(out_dtype, torch.float32)


def compute_causal_statistics(
    data: Tensor,
    observation_mask: Tensor,
    minimum_scale: float,
    stabilize_with_global: bool,
    scale_factor_exponent: float = 10.0,
) -> Tuple[Tensor, Tensor]:
    weights = torch.ones_like(data) * observation_mask
    with torch.no_grad():
        x = data.to(torch.float64)
        w = weights.to(torch.float64)
        weighted = w * x
        cum_w = torch.cumsum(w, dim=-1)
        cum_x = torch.cumsum(weighted, dim=-1)
        denominator = cum_w.clamp_min(1.0)
        means = cum_x / denominator
        shifted = torch.zeros_like(means)
        shifted[..., 1:] = means[..., :-1]
        increment = (x - shifted) * (x - means) * w
        m2 = torch.cumsum(increment, dim=-1)
        variance = m2 / torch.clamp(denominator - 1.0, min=1.0)
        scales = torch.sqrt(variance).clamp(min=minimum_scale)
        if stabilize_with_global:
            lo, hi = 10.0 ** (-scale_factor_exponent), 10.0 ** scale_factor_exponent
            global_den = (weights * observation_mask).sum(-1, keepdim=True).clamp_min(1.0)
            global_means = torch.nan_to_num(weighted.sum(-1, keepdim=True) / global_den)
            global_var = (((x - global_means) * weights * observation_mask) ** 2).sum(-1, keepdim=True) / global_den
            global_scale = torch.sqrt(global_var + minimum_scale).expand_as(scales)
            floor = torch.max(torch.tensor(minimum_scale, device=scales.device, dtype=scales.dtype), global_scale * lo)
            scales = torch.clamp(scales, min=floor, max=global_scale * hi)
        return means.to(data.dtype), scales.to(data.dtype)


def causal_patch_loc_scale(
    data: Tensor,
    observation_mask: Tensor,
    patch_size: int,
    minimum_scale: float,
    stabilize_with_global: bool,
) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        means, scales = compute_causal_statistics(data, observation_mask, minimum_scale, stabilize_with_global)
        patch_means = means.unfold(-1, patch_size, patch_size)[..., -1]
        patch_scales = scales.unfold(-1, patch_size, patch_size)[..., -1]
        return patch_means.repeat_interleave(patch_size, dim=-1), patch_scales.repeat_interleave(patch_size, dim=-1)


def apply_affine_asinh_masked(
    data: Tensor,
    loc: Tensor,
    scale: Tensor,
    observation_mask: Tensor,
    use_asinh: bool,
) -> Tensor:
    out_dtype, work = _fp32_io(data.dtype, loc.dtype, scale.dtype)
    scaled = (data.to(work) - loc.to(work)) / scale.to(work).clamp_min(torch.finfo(work).eps)
    scaled = scaled.masked_fill(~observation_mask, 0.0)
    if use_asinh:
        scaled = torch.where(observation_mask, scaled.asinh(), scaled)
    return scaled.to(out_dtype)


def invert_affine_asinh(x: Tensor, loc: Tensor, scale: Tensor, use_asinh: bool) -> Tensor:
    out_dtype, work = _fp32_io(x.dtype, loc.dtype, scale.dtype)
    x = x.to(work)
    if use_asinh:
        x = x.clamp(-6.0, 6.0).sinh()
    return (x * scale.to(work) + loc.to(work)).to(out_dtype)


def clamp_nonfinite(vals: Tensor) -> Tensor:
    clamped = torch.where(
        vals == float("inf"),
        torch.where(vals.isfinite(), vals, -float("inf")).amax(dim=-1, keepdim=True),
        torch.where(
            vals == -float("inf"),
            torch.where(vals.isfinite(), vals, float("inf")).amin(dim=-1, keepdim=True),
            vals,
        ),
    )
    return torch.where(clamped.isinf(), torch.zeros_like(clamped), clamped)


def denormalize_quantiles(preds: Tensor, loc: Tensor, scale: Tensor, use_asinh: bool) -> Tensor:
    return clamp_nonfinite(invert_affine_asinh(preds, loc, scale, use_asinh))


def welford_combine(
    w_a: Tensor, mean_a: Tensor, m2_a: Tensor, w_b: Tensor, mean_b: Tensor, m2_b: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    w = w_a + w_b
    w_safe = w.clamp(min=torch.finfo(w.dtype).tiny)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + delta.square() * w_a * w_b / w_safe
    is_zero = w == 0
    mean = torch.where(is_zero, torch.zeros_like(mean), mean)
    m2 = torch.where(is_zero, torch.zeros_like(m2), m2)
    return w, mean, m2


def cumulative_welford(w: Tensor, mean: Tensor, m2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    n = w.shape[1]
    d = 1
    while d < n:
        pad = (d, 0)
        w_l = F.pad(w[:, : n - d], pad)
        mean_l = F.pad(mean[:, : n - d], pad)
        m2_l = F.pad(m2[:, : n - d], pad)
        w, mean, m2 = welford_combine(w_l, mean_l, m2_l, w, mean, m2)
        d *= 2
    return w, mean, m2


def advancing_block_frames(
    context: Tensor, observation_mask: Tensor, sinh_median: Tensor, minimum_scale: float
) -> Tuple[Tensor, Tensor]:
    r, n_pred, ps = sinh_median.shape
    blank = sinh_median.new_zeros(r, n_pred, ps)
    values = torch.cat([context.unflatten(-1, (-1, ps)).to(torch.float64), blank], dim=1)
    weights = torch.cat([observation_mask.unflatten(-1, (-1, ps)).to(torch.float64), blank], dim=1)
    cnt = weights.sum(-1)
    mean = (values * weights).sum(-1) / cnt.clamp(min=1.0)
    m2 = (weights * (values - mean.unsqueeze(-1)).square()).sum(-1)
    is_zero = cnt == 0
    mean = torch.where(is_zero, torch.zeros_like(mean), mean)
    m2 = torch.where(is_zero, torch.zeros_like(m2), m2)
    cnt, mean, m2 = (s[:, -n_pred - 1] for s in cumulative_welford(cnt, mean, m2))

    offset = mean
    clip = torch.finfo(torch.float32).max ** 0.5 / 1e3
    locs, scales = [], []
    for h in range(n_pred):
        var = (m2 / (cnt - 1.0).clamp(min=1.0)).clamp(min=0.0)
        scale = var.sqrt().clamp(min=minimum_scale)
        locs.append(mean)
        scales.append(scale)
        folded = (sinh_median[:, h] * scale[:, None] + mean[:, None] - offset[:, None]).clamp(-clip, clip)
        folded = folded + offset[:, None]
        b_mean = folded.mean(-1)
        b_m2 = (folded - b_mean[:, None]).square().sum(-1)
        cnt, mean, m2 = welford_combine(cnt, mean, m2, torch.full_like(cnt, ps), b_mean, b_m2)
    return torch.stack(locs, dim=1), torch.stack(scales, dim=1)


def instance_norm_stats(values: Tensor, observation_mask: Tensor, minimum_scale: float) -> Tuple[Tensor, Tensor, Tensor]:
    orig_dtype = values.dtype
    stat_dtype = torch.promote_types(orig_dtype, torch.float32)
    x = values.to(stat_dtype)
    mask_f = observation_mask.to(stat_dtype)
    n = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    loc = (x * mask_f).sum(dim=-1, keepdim=True) / n
    var = (((x - loc) * mask_f) ** 2).sum(dim=-1, keepdim=True) / n
    scale = var.sqrt().clamp_min(minimum_scale)
    is_zero_one = ((values == 0.0) | (values == 1.0) | ~observation_mask).all(dim=-1, keepdim=True)
    is_binary = is_zero_one & observation_mask.any(dim=-1, keepdim=True)
    loc = torch.where(is_binary, torch.zeros_like(loc), loc)
    scale = torch.where(is_binary, torch.ones_like(scale), scale)
    return loc.to(orig_dtype), scale.to(orig_dtype), is_binary


def ffill_with_mask(x: Tensor, mask: Tensor) -> Tensor:
    idx = torch.arange(x.shape[-1], device=x.device).view(*([1] * (x.dim() - 1)), -1)
    valid_idx = torch.where(mask, idx, torch.zeros_like(idx)).cummax(dim=-1).values
    filled = torch.gather(x, -1, valid_idx)
    return torch.where(mask.cummax(dim=-1).values.bool(), filled, torch.zeros_like(x))


def pad_to_patch(rows: Tensor, mask: Tensor, patch_size: int) -> Tuple[Tensor, Tensor]:
    rem = rows.shape[-1] % patch_size
    if rem == 0:
        return rows, mask
    pad_len = patch_size - rem
    return F.pad(rows, (pad_len, 0), value=0.0), F.pad(mask, (pad_len, 0), value=False)


# ==============================================================================
# 2. Two-Stream Masking & RoPE Gated Transformer Layers
# ==============================================================================


@dataclass(frozen=True)
class TwoStreamMask:
    mask: Tensor
    enc_len: int


def compute_enc_key_validity(obs_per_patch: Tensor, attend_missing: bool) -> Tensor:
    if not attend_missing:
        return obs_per_patch
    started = obs_per_patch.cummax(dim=-1).values
    return obs_per_patch | started


def enc_causal_mask(key_valid: Tensor) -> Tensor:
    _, ll = key_valid.shape
    device = key_valid.device
    causal = torch.tril(torch.ones(ll, ll, dtype=torch.bool, device=device))
    eye = torch.eye(ll, dtype=torch.bool, device=device)
    return (causal[None] & key_valid[:, None, :]) | eye[None]


def build_forecast_time_mask(context_len: int, horizon: int, key_valid: Tensor, device: torch.device) -> Tensor:
    rows = key_valid.shape[0]
    enc = enc_causal_mask(key_valid)
    enc_dec = torch.zeros(rows, context_len, horizon, dtype=torch.bool, device=device)
    dec_enc = key_valid[:, None, :].expand(rows, horizon, context_len)
    dec_dec = torch.tril(torch.ones(horizon, horizon, dtype=torch.bool, device=device))[None].expand(rows, horizon, horizon)
    top = torch.cat([enc, enc_dec], dim=2)
    bottom = torch.cat([dec_enc, dec_dec], dim=2)
    return torch.cat([top, bottom], dim=1).unsqueeze(1)


def build_forecast_positions(context_len: int, horizon: int, first_obs: Tensor) -> Tensor:
    p = torch.arange(context_len + horizon, device=first_obs.device)
    return p[None, :] - first_obs[:, None]


def group_first_observed_patch(obs_per_patch: Tensor, inverse: Tensor, num_groups: int) -> Tensor:
    n = obs_per_patch.shape[-1]
    idx = inverse[:, None].expand(-1, n)
    group_obs = torch.zeros(num_groups, n, dtype=torch.float32, device=obs_per_patch.device)
    group_obs.scatter_reduce_(0, idx, obs_per_patch.float(), reduce="amax", include_self=True)
    return group_obs.argmax(dim=-1)


def build_two_stream_positions(n_patches: int, first_obs: Tensor) -> Tensor:
    """`(R, 2N)` positions: encoder and decoder token of the same patch share a time."""
    p = torch.arange(n_patches, device=first_obs.device)
    row = p[None, :] - first_obs[:, None]
    return torch.cat([row, row], dim=-1)


def compute_block_layout(min_context: Tensor, n_patches: int, block_size: int) -> Tuple[Tensor, Tensor]:
    """Per decoder position: the anchor of its block and whether it is active (`p >= s`)."""
    device = min_context.device
    p = torch.arange(n_patches, device=device)[None, :]
    s = min_context[:, None]
    active = p >= s
    offset = (p - s).clamp(min=0)
    anchor = s + block_size * (offset // block_size)
    anchor = torch.where(active, anchor, torch.zeros_like(anchor))
    return anchor, active


def build_two_stream_time_mask(
    min_context: Tensor,
    n_patches: int,
    block_size: int,
    key_valid: Tensor,
    bidirectional_rows: Optional[Tensor],
) -> Tensor:
    """`(R, 1, 2N, 2N)` time mask over a full `[Enc ; Dec]` window (the covariate path)."""
    anchor, active = compute_block_layout(min_context, n_patches, block_size)
    rows, n = active.shape
    device = active.device
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    key_col = key_valid[:, None, :]
    j = torch.arange(n, device=device).view(1, 1, n)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    enc_enc = enc_causal_mask(key_valid)
    if bidirectional_rows is not None:
        enc_bi = key_valid[:, None, :].expand(rows, n, n) | eye[None]
        enc_enc = torch.where(bidirectional_rows[:, None, None], enc_bi, enc_enc)
    enc_dec = torch.zeros(rows, n, n, dtype=torch.bool, device=device)
    dec_enc = active[:, :, None] & (j < anchor[:, :, None]) & key_col
    same_block = anchor[:, :, None] == anchor[:, None, :]
    dec_dec = same_block & causal[None] & active[:, :, None] & active[:, None, :]
    dec_dec = dec_dec | eye[None]
    top = torch.cat([enc_enc, enc_dec], dim=2)
    bottom = torch.cat([dec_enc, dec_dec], dim=2)
    return torch.cat([top, bottom], dim=1).unsqueeze(1)


def variate_firewall(
    r: int, device: torch.device, target_rows: Optional[Tensor], bidirectional_rows: Optional[Tensor]
) -> Optional[Tensor]:
    """`(R, R)` forbidden query-to-key row pairs, or None."""
    if target_rows is None and bidirectional_rows is None:
        return None
    forbid = torch.zeros(r, r, dtype=torch.bool, device=device)
    if target_rows is not None:
        forbid = forbid | ((~target_rows[:, None]) & target_rows[None, :])
    if bidirectional_rows is not None:
        forbid = forbid | (bidirectional_rows[:, None] & (~bidirectional_rows)[None, :])
    return forbid


_IDENTITY_VARIATE_MASK = "identity"


def build_two_stream_variate_mask(
    group_ids: Tensor,
    enc_key_valid: Tensor,
    n_dec: int,
    target_rows: Optional[Tensor] = None,
    bidirectional_rows: Optional[Tensor] = None,
) -> Optional[Tensor | str]:
    r, _ = enc_key_valid.shape
    if r == 1:
        return None
    device = enc_key_valid.device
    if int(torch.unique(group_ids).numel()) == r:
        # Every row is an independent univariate series -> variate attention mask is I_r
        return _IDENTITY_VARIATE_MASK
    row_eq = group_ids[:, None] == group_ids[None, :]
    dec_valid = torch.ones(r, n_dec, dtype=torch.bool, device=device)
    slot_valid = torch.cat([enc_key_valid, dec_valid], dim=1)
    key_valid = slot_valid.permute(1, 0)
    mask = row_eq[None, None] & key_valid[:, None, None, :]
    forbid = variate_firewall(r, device, target_rows, bidirectional_rows)
    if forbid is not None:
        mask = mask & ~forbid[None, None]
    eye = torch.eye(r, dtype=torch.bool, device=device)
    return mask | eye[None, None]


def target_encode_covariate(
    target_context: np.ndarray,
    past_codes: np.ndarray,
    future_codes: Optional[np.ndarray] = None,
    smooth: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Per-series smoothed mean-target encoding for categorical/discrete covariates."""
    target = np.asarray(target_context, dtype=np.float64)
    past_codes = np.asarray(past_codes, dtype=np.float64)
    observed = np.isfinite(target)
    target = np.where(observed, target, 0.0)
    item_count = float(observed.sum())
    item_mean = float(target.sum() / item_count) if item_count > 0 else 0.0

    categories = np.unique(past_codes[np.isfinite(past_codes)])
    nan_slot = len(categories)

    def _slots(codes: np.ndarray, unseen: Optional[int] = None) -> np.ndarray:
        idx = np.searchsorted(categories, np.nan_to_num(codes, nan=0.0))
        idx = np.minimum(idx, max(len(categories) - 1, 0))
        known = np.isfinite(codes) & (len(categories) > 0)
        if len(categories) > 0:
            known &= categories[idx] == codes
        return np.where(np.isnan(codes), nan_slot, np.where(known, idx, unseen if unseen is not None else 0)).astype(np.intp)

    past_slots = _slots(past_codes)
    sums = np.bincount(past_slots, weights=target * observed, minlength=nan_slot + 1)
    counts = np.bincount(past_slots, weights=observed.astype(np.float64), minlength=nan_slot + 1)
    lookup = np.append((smooth * item_mean + sums) / (smooth + counts), item_mean)

    encoded_past = lookup[past_slots].astype(np.float32)
    if future_codes is None:
        return encoded_past, None
    future_slots = _slots(np.asarray(future_codes, dtype=np.float64), unseen=nan_slot + 1)
    return encoded_past, lookup[future_slots].astype(np.float32)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.scale.float()).to(self.scale.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, use_bias: bool) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(input_dims, hidden_dims, bias=use_bias)
        self.output_layer = nn.Linear(hidden_dims, output_dims, bias=use_bias)
        self.residual_layer = nn.Linear(input_dims, output_dims, bias=use_bias)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(self.activation(self.hidden_layer(x))) + self.residual_layer(x)


class FusedSwiGLU(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, use_bias: bool = False) -> None:
        super().__init__()
        self.w12 = nn.Linear(embed_dim, 2 * hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w12(x).chunk(2, dim=-1)
        prod = (gate.float() * F.silu(up.float())).clamp(min=-32768.0, max=32768.0).to(self.w3.weight.dtype)
        return self.w3(prod)


class SinCosRoPE(nn.Module):
    inv_freq: Tensor

    def __init__(self, dim: int, base: float, rotary_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.rotary_dim = rotary_dim
        self.register_buffer("inv_freq", self._build_inv_freq(None), persistent=False)

    def _build_inv_freq(self, device: Optional[torch.device]) -> Tensor:
        inv_freq = torch.zeros(self.dim // 2, dtype=torch.float32, device=device)
        idx = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
        inv_freq[: self.rotary_dim // 2] = 1.0 / (self.base ** (idx / self.rotary_dim))
        return inv_freq

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        if self.inv_freq.is_meta or self.inv_freq.device != x.device or float(self.inv_freq[0]) != 1.0:
            self.inv_freq = self._build_inv_freq(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    cos = cos.to(dtype=q.dtype).unsqueeze(2)
    sin = sin.to(dtype=q.dtype).unsqueeze(2)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _sdpa(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
    batch, seqlen, _, _ = query.shape
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        is_causal=is_causal,
        dropout_p=0.0,
    )
    return out.transpose(1, 2).contiguous().view(batch, seqlen, -1)


class Attention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, is_time: bool, use_bias: bool, gate: str, value_residual: bool = False
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_time = is_time
        self.wQKV = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.wO = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self._gate = nn.Linear(embed_dim, embed_dim, bias=use_bias) if gate == "sdpa_sigmoid" else None
        self.v_res_logit = nn.Parameter(torch.tensor(-2.0)) if (value_residual and is_time) else None

    def _qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch, seq, _ = x.shape
        qkv = self.wQKV(x.contiguous()).reshape(batch, seq, 3, self.head_dim, self.num_heads)
        q, k, v = qkv.permute(2, 0, 1, 4, 3).unbind(dim=0)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor | TwoStreamMask | str],
        cos: Tensor,
        sin: Tensor,
        v_first: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        rows, num_patches, _ = x.shape
        inputs = x if self.is_time else x.permute(1, 0, 2)
        v_time_out: Optional[Tensor] = None
        if not self.is_time and isinstance(mask, str) and mask == _IDENTITY_VARIATE_MASK:
            qkv = self.wQKV(inputs.contiguous()).reshape(num_patches, rows, 3, self.head_dim, self.num_heads)
            v = qkv[:, :, 2].permute(0, 1, 3, 2).contiguous().view(num_patches, rows, self.embed_dim)
            out = v
        else:
            q, k, v = self._qkv(inputs)
            if self.is_time:
                v_time_out = v
                if self.v_res_logit is not None and v_first is not None:
                    lam = torch.sigmoid(self.v_res_logit).to(v.dtype)
                    v = (1.0 - lam) * v + lam * v_first.to(v.dtype)
                q, k = _apply_rotary(q, k, cos, sin)
            if isinstance(mask, TwoStreamMask):
                n = mask.enc_len
                out_enc = _sdpa(q[:, :n], k[:, :n], v[:, :n], mask.mask[:, :, :n, :n])
                out_dec = _sdpa(q[:, n:], k, v, mask.mask[:, :, n:, :])
                out = torch.cat([out_enc, out_dec], dim=1)
            else:
                out = _sdpa(q, k, v, mask if not isinstance(mask, str) else None, is_causal=self.is_time and mask is None)
        if self._gate is not None:
            out = out * self._gate(inputs).sigmoid()
        if self.is_time:
            out = out.reshape(rows, num_patches, self.embed_dim)
        else:
            out = out.reshape(num_patches, rows, self.embed_dim).permute(1, 0, 2)
        return self.wO(out), v_time_out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        kind: str,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
        value_residual: bool = False,
    ) -> None:
        super().__init__()
        axes = LAYER_KIND_AXES[kind]
        self.attn_norms = nn.ModuleList([RMSNorm(embed_dim, norm_eps) for _ in axes])
        self.attns = nn.ModuleList(
            [
                Attention(
                    embed_dim, num_heads, is_time, attn_bias,
                    time_attn_gate if is_time else variate_attn_gate,
                    value_residual=value_residual,
                )
                for is_time in axes
            ]
        )
        self.mlp_norm = RMSNorm(embed_dim, norm_eps)
        self.mlp = FusedSwiGLU(embed_dim, mlp_hidden_dim, mlp_bias)

    def forward(
        self,
        x: Tensor,
        time_mask: Optional[Tensor | TwoStreamMask],
        variate_mask: Optional[Tensor],
        cos: Tensor,
        sin: Tensor,
        v_first: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        res_dtype = self.mlp_norm.scale.dtype
        x = x.to(res_dtype)
        v_time_first = v_first
        for norm, attn in zip(self.attn_norms, self.attns):
            out, v_t = attn(norm(x), time_mask if attn.is_time else variate_mask, cos, sin, v_first=v_time_first)
            if attn.is_time and v_time_first is None:
                v_time_first = v_t
            x = x + out.to(res_dtype)
        return x + self.mlp(self.mlp_norm(x)).to(res_dtype), v_time_first


class Transformer(nn.Module):
    def __init__(self, config: TelosConfig) -> None:
        super().__init__()
        c = config
        n_layers = len(c.block_schedule)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    kind=kind,
                    embed_dim=c.d_model,
                    num_heads=c.num_heads,
                    mlp_hidden_dim=c.mlp_hidden_dim,
                    norm_eps=c.norm_eps,
                    attn_bias=c.attn_bias,
                    mlp_bias=c.mlp_bias,
                    time_attn_gate=c.time_attn_gate,
                    variate_attn_gate=c.variate_attn_gate,
                    value_residual=c.value_residual and (i > 0),
                )
                for i, kind in enumerate(c.block_schedule)
            ]
        )
        self.unet_skip_scales: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(n_layers // 2, c.d_model)) if c.unet_skip_connections else None
        )
        self.rope = SinCosRoPE(dim=c.head_dim, base=c.rope_theta, rotary_dim=c.rope_dim)
        self.out_norm = RMSNorm(c.d_model, c.norm_eps)

    def forward(
        self,
        x: Tensor,
        time_mask: Optional[Tensor | TwoStreamMask],
        variate_mask: Optional[Tensor],
        positions: Tensor,
        return_layer_states: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        cos, sin = self.rope(x, positions)
        layer_states: List[Tensor] = []
        early_skips: List[Tensor] = []
        v_first: Optional[Tensor] = None
        n_l = len(self.layers)
        half = n_l // 2
        for idx, layer in enumerate(self.layers):
            if self.unet_skip_scales is not None and idx >= n_l - half:
                skip_idx = n_l - 1 - idx
                x = x + early_skips[skip_idx] * self.unet_skip_scales[skip_idx].to(x.dtype)
            x, v_first = layer(x, time_mask, variate_mask, cos, sin, v_first=v_first)
            if self.unet_skip_scales is not None and idx < half:
                early_skips.append(x)
            if return_layer_states:
                layer_states.append(x)
        out = self.out_norm(x)
        if return_layer_states:
            return out, layer_states
        return out


# ==============================================================================
# 3. Patch Embedding & Monotone Quantile Head
# ==============================================================================


class PatchRepresentation(nn.Module):
    def __init__(self, patch_size: int, d_model: int, hidden_dims: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = ResidualBlock(2 * patch_size, hidden_dims, d_model, use_bias=True)
        self.forecast_query = nn.Parameter(torch.zeros(d_model))

    def forward(self, scaled: Tensor, observed: Tensor) -> Tensor:
        r, t = scaled.shape
        n = t // self.patch_size
        values = scaled.reshape(r, n, self.patch_size)
        missing = (~observed).reshape(r, n, self.patch_size).to(scaled.dtype)
        x = torch.cat([values, missing], dim=-1).to(self.patch_proj.hidden_layer.weight.dtype)
        return self.patch_proj(x)

    def forecast_tokens(self, rows: int, n_patches: int) -> Tensor:
        return self.forecast_query.view(1, 1, -1).expand(rows, n_patches, -1)


def monotone_quantiles(raw: Tensor, levels: Tuple[float, ...], width_factor: float) -> Tensor:
    m = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(m + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]
    widths_t = torch.tensor(widths, device=raw.device, dtype=raw.dtype)
    inc = width_factor * widths_t * F.softplus(raw)
    median = raw[..., m : m + 1]
    up = torch.cumsum(inc[..., m + 1 :], dim=-1)
    down = torch.flip(torch.cumsum(torch.flip(inc[..., :m], dims=[-1]), dim=-1), dims=[-1])
    big = torch.finfo(raw.dtype).max
    return torch.cat([median - down, median, median + up], dim=-1).clamp(min=-big, max=big)


class QuantileHead(nn.Module):
    def __init__(self, config: TelosConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_quantiles = config.num_quantiles
        self.projection = ResidualBlock(
            config.d_model,
            config.head_hidden_dim,
            config.patch_size * config.num_quantiles,
            use_bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        raw = self.projection(x)
        *leading, n_patches, _ = raw.shape
        preds = raw.reshape(*leading, n_patches, self.patch_size, self.num_quantiles)
        preds = preds.to(torch.promote_types(preds.dtype, torch.float32))
        preds = monotone_quantiles(preds, self.config.quantile_knots, self.config.quantile_width_factor)
        return preds.reshape(*leading, n_patches * self.patch_size, self.num_quantiles)


# ==============================================================================
# 4. Complete Standalone Model & Block-AR Forecaster
# ==============================================================================


class JEPALatentPredictor(nn.Module):
    """Horizon-conditioned JEPA latent predictor $P_\\phi(\\mathbf{z}^{\\text{ctx}}_{1:L}, \\mathbf{q}_{L+1:L+H}) \\to \\hat{\\mathbf{z}}_{L+1:L+H}$.

    Takes causal context representations from the encoder stream plus horizon query states and predicts
    the future target encoder representations in latent space using RoPE-conditioned gated attention + SwiGLU blocks.
    """

    def __init__(self, config: TelosConfig) -> None:
        super().__init__()
        c = config
        self.max_horizon_patches = 64
        self.horizon_step_emb = nn.Parameter(torch.randn(self.max_horizon_patches, c.d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    kind="time",
                    embed_dim=c.d_model,
                    num_heads=c.num_heads,
                    mlp_hidden_dim=c.jepa_predictor_hidden,
                    norm_eps=c.norm_eps,
                    attn_bias=c.attn_bias,
                    mlp_bias=c.mlp_bias,
                    time_attn_gate=c.time_attn_gate,
                    variate_attn_gate=c.variate_attn_gate,
                )
                for _ in range(c.jepa_predictor_layers)
            ]
        )
        self.rope = SinCosRoPE(dim=c.head_dim, base=c.rope_theta, rotary_dim=c.rope_dim)
        self.out_norm = RMSNorm(c.d_model, c.norm_eps)

    def forward(
        self, h_ctx: Tensor, h_dec: Tensor, ctx_key_valid: Tensor, first_obs: Tensor
    ) -> Tensor:
        ll = h_ctx.shape[1]
        h = h_dec.shape[1]
        step_idx = torch.arange(h, device=h_dec.device).clamp_max(self.max_horizon_patches - 1)
        q_dec = h_dec + self.horizon_step_emb[step_idx].to(h_dec.dtype)[None, :, :]
        x = torch.cat([h_ctx, q_dec], dim=1)
        time_mask = build_forecast_time_mask(ll, h, ctx_key_valid, h_ctx.device)
        mask = TwoStreamMask(mask=time_mask, enc_len=ll)
        positions = build_forecast_positions(ll, h, first_obs)
        cos, sin = self.rope(x, positions)
        for layer in self.layers:
            x, _ = layer(x, mask, None, cos, sin)
        return self.out_norm(x[:, ll:, :])

    def forward_tiled(
        self,
        h_ctx: Tensor,
        h_dec: Tensor,
        tiled_mask: TwoStreamMask,
        positions: Tensor,
        step_offsets: Tensor,
    ) -> Tensor:
        """Tiled Multi-Origin DHF forward pass: predicts future latents at ALL tiled blocks simultaneously."""
        step_clamped = step_offsets.clamp(0, self.max_horizon_patches - 1)
        q_dec = h_dec + self.horizon_step_emb[step_clamped].to(h_dec.dtype)
        x = torch.cat([h_ctx, q_dec], dim=1)
        cos, sin = self.rope(x, positions)
        for layer in self.layers:
            x, _ = layer(x, tiled_mask, None, cos, sin)
        return self.out_norm(x[:, h_ctx.shape[1] :, :])


class TelosModel(PreTrainedModel):
    """Telos-Small (50.1M) Standalone Time-Series JEPA Foundation Model."""

    config_class = TelosConfig
    base_model_prefix = "telos"

    def __init__(self, config: TelosConfig) -> None:
        super().__init__(config)
        self.config = config
        c = config
        self.representation = PatchRepresentation(c.patch_size, c.d_model, c.patch_proj_hidden)
        self.transformer = Transformer(c)
        self.jepa_predictor: Optional[JEPALatentPredictor] = (
            JEPALatentPredictor(c) if c.jepa_predictor_layers > 0 else None
        )
        self.head = QuantileHead(c)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, SinCosRoPE) and module.inv_freq.is_meta:
            module.inv_freq = module._build_inv_freq(None)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def scale(self, inputs: Tensor, observed: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        c = self.config
        loc, scale = causal_patch_loc_scale(
            inputs.unsqueeze(1), observed.unsqueeze(1), c.patch_size, c.minimum_scale, c.stabilize_with_global
        )
        loc, scale = loc.squeeze(1), scale.squeeze(1)
        scaled = apply_affine_asinh_masked(inputs, loc, scale, observed, use_asinh=c.use_asinh)
        return scaled, loc, scale

    def _known_cov_scaling(
        self, inputs: Tensor, observed: Tensor, scaled: Tensor, loc: Tensor, scale: Tensor, known_rows: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Normalize known-future covariate rows with non-causal whole-row statistics."""
        if not bool(known_rows.any()):
            return scaled, loc, scale
        loc_c, scale_c, is_binary = instance_norm_stats(inputs, observed, minimum_scale=self.config.minimum_scale)
        known_scaled = apply_affine_asinh_masked(inputs, loc_c, scale_c, observed, use_asinh=False)
        if self.config.use_asinh:
            work = torch.promote_types(known_scaled.dtype, torch.float32)
            squashed = known_scaled.to(work).asinh().to(known_scaled.dtype)
            known_scaled = torch.where(observed & ~is_binary, squashed, known_scaled)
        sel = known_rows[:, None]
        return torch.where(sel, known_scaled, scaled), torch.where(sel, loc_c, loc), torch.where(sel, scale_c, scale)

    def encode_context_only(
        self,
        inputs: Tensor,
        observed: Tensor,
        group_ids: Optional[Tensor] = None,
        return_layer_states: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """Pure causal Context / Target Encoder $E_\\theta(\\mathbf{x}_{1:N})$ with zero horizon query tokens.

        Used by the Momentum EMA Target Encoder during V-JEPA training to compute causal target patch
        representations $\\mathbf{z}^{\\text{tgt}}_{L+1:L+H} = E_{\\bar\\theta}([\\mathbf{x}_{1:L}, \\mathbf{y}_{L+1:L+H}])[:, L:]$.
        """
        c = self.config
        ps = c.patch_size
        rows, t = inputs.shape
        if group_ids is None:
            group_ids = torch.arange(rows, device=inputs.device)
        n = t // ps
        scaled, _, _ = self.scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        obs_per_patch = observed.unflatten(-1, (n, ps)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = enc_causal_mask(key_valid).unsqueeze(1)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, 0)
        unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
        positions = torch.arange(n, device=inputs.device)[None, :] - first_obs[:, None]
        return self.transformer(x_enc, time_mask, variate_mask, positions, return_layer_states=return_layer_states)

    def encode_tiled_dhf(
        self,
        full_inputs: Tensor,
        full_observed: Tensor,
        min_context: Tensor,
        block_size: int,
        group_ids: Optional[Tensor] = None,
        role_ids: Optional[Tensor] = None,
        return_layer_states: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[Tensor]]:
        """Multi-Origin Tiled Decoupled-Horizon JEPA (`Tiled-DHF JEPA`):

        Encodes causal context tokens `v_0 .. v_{N-1}` once while simultaneously predicting future latents
        across consecutive tiled blocks of length `block_size` starting at `min_context` (`s_rows`), so a
        single forward pass supervises `N - min_context` forecast patches across `(N - s) / block_size` origins!

        Returns `(h_pred_dec (R, N, D), anchor (R, N), active (R, N), loc_patch (R, N), dec_layer_states)`.
        """
        c = self.config
        ps = c.patch_size
        rows, w = full_inputs.shape
        n = w // ps
        if group_ids is None:
            group_ids = torch.arange(rows, device=full_inputs.device)

        has_cov = role_ids is not None and c.covariate_enabled and not bool((role_ids == ROLE_TARGET).all())
        target_rows = (role_ids == ROLE_TARGET) if has_cov else None
        known_rows = (role_ids == ROLE_KNOWN_COV) if has_cov else None

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        if known_rows is not None:
            scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_rows)

        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)

        obs_per_patch = full_observed.unflatten(-1, (n, ps)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        anchor, active = compute_block_layout(min_context, n, block_size)
        time_mask = build_two_stream_time_mask(min_context, n, block_size, key_valid, bidirectional_rows=known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows=target_rows, bidirectional_rows=known_rows
        )
        unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
        positions = build_two_stream_positions(n, first_obs)
        ts_mask = TwoStreamMask(mask=time_mask, enc_len=n)

        x = torch.cat([x_enc, x_dec], dim=1)
        if return_layer_states:
            h_all, l_states = self.transformer(x, ts_mask, variate_mask, positions, return_layer_states=True)
            dec_layers = [ls[:, n:, :] for ls in l_states]
        else:
            h_all = self.transformer(x, ts_mask, variate_mask, positions)
            dec_layers = []

        h_ctx, h_dec = h_all[:, :n, :], h_all[:, n:, :]
        if self.jepa_predictor is not None:
            p_idx = torch.arange(n, device=full_inputs.device)[None, :]
            step_offsets = (p_idx - anchor).clamp(min=0)
            pred_time_mask = build_two_stream_time_mask(min_context, n, block_size, key_valid, bidirectional_rows=None)
            h_dec = self.jepa_predictor.forward_tiled(
                h_ctx, h_dec, TwoStreamMask(mask=pred_time_mask, enc_len=n), positions, step_offsets
            )

        loc_p = loc.unflatten(-1, (n, ps))[..., 0]
        scale_p = scale.unflatten(-1, (n, ps))[..., 0]
        return h_dec, anchor, active, torch.stack([loc_p, scale_p], dim=-1), dec_layers

    def encode_penultimate(
        self,
        inputs: Tensor,
        observed: Tensor,
        horizon_patches: int,
        group_ids: Optional[Tensor] = None,
        role_ids: Optional[Tensor] = None,
        known_future: Optional[Tensor] = None,
        known_future_mask: Optional[Tensor] = None,
        return_layer_states: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]]:
        """Run Causal Patch Scaling + Two-Stream Transformer + Horizon-Conditioned JEPA Latent Predictor.

        Supports both context-only (`role_ids=None`) and covariate cross-attention (`ROLE_TARGET`,
        `ROLE_PAST_COV`, `ROLE_KNOWN_COV` with `known_future` values in horizon patches).
        """
        c = self.config
        ps = c.patch_size
        rows, t = inputs.shape
        if group_ids is None:
            group_ids = torch.arange(rows, device=inputs.device)
        ll, h = t // ps, horizon_patches

        has_cov = (
            role_ids is not None
            and c.covariate_enabled
            and not bool((role_ids == ROLE_TARGET).all())
        )

        if not has_cov:
            scaled, loc, scale = self.scale(inputs, observed)
            x_enc = self.representation(scaled, observed)
            x_dec = self.representation.forecast_tokens(rows, h)

            obs_per_patch = observed.unflatten(-1, (ll, ps)).any(dim=-1)
            key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
            time_mask = build_forecast_time_mask(ll, h, key_valid, inputs.device)
            variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
            unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
            first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
            positions = build_forecast_positions(ll, h, first_obs)

            x = torch.cat([x_enc, x_dec], dim=1)
            mask: Tensor | TwoStreamMask = TwoStreamMask(mask=time_mask, enc_len=ll) if c.split_context_horizon_attn else time_mask
            if return_layer_states:
                h_all, l_states = self.transformer(x, mask, variate_mask, positions, return_layer_states=True)
                dec_layer_states = [ls[:, ll:, :] for ls in l_states]
            else:
                h_all = self.transformer(x, mask, variate_mask, positions)
                dec_layer_states = []
            h_raw_dec = h_all[:, ll:, :]  # (R, H_patches, d_model)
            if self.jepa_predictor is not None:
                h_raw_dec = self.jepa_predictor(h_all[:, :ll, :], h_raw_dec, key_valid, first_obs)
            h_aligned, z_rot = h_raw_dec, h_raw_dec

            anchor_loc = loc.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
            anchor_scale = scale.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
            if return_layer_states:
                return h_aligned, z_rot, h_raw_dec, anchor_loc, anchor_scale, dec_layer_states
            return h_aligned, z_rot, h_raw_dec, anchor_loc, anchor_scale

        # Two-stream covariate window path (Enc over context + horizon, Dec over context + horizon)
        assert role_ids is not None
        horizon_len = h * ps
        if known_future is None:
            known_future = torch.zeros(rows, horizon_len, device=inputs.device, dtype=inputs.dtype)
            known_future_mask = torch.zeros(rows, horizon_len, device=inputs.device, dtype=torch.bool)
        assert known_future_mask is not None
        full_inputs = torch.cat([inputs, known_future], dim=-1)
        full_observed = torch.cat([observed, known_future_mask], dim=-1)
        n = ll + h
        target_rows = role_ids == ROLE_TARGET
        known_rows = role_ids == ROLE_KNOWN_COV

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_rows)
        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)

        obs_per_patch = full_observed.unflatten(-1, (n, ps)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        s_rows = torch.full((rows,), ll, device=full_inputs.device, dtype=torch.long)
        time_mask = build_two_stream_time_mask(s_rows, n, c.block_size, key_valid, bidirectional_rows=known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows=target_rows, bidirectional_rows=known_rows
        )
        unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
        positions = build_two_stream_positions(n, first_obs)

        x = torch.cat([x_enc, x_dec], dim=1)
        mask = TwoStreamMask(mask=time_mask, enc_len=n) if c.split_context_horizon_attn else time_mask
        if return_layer_states:
            h_all, l_states = self.transformer(x, mask, variate_mask, positions, return_layer_states=True)
            dec_layer_states = [ls[:, n + ll : 2 * n, :] for ls in l_states]
        else:
            h_all = self.transformer(x, mask, variate_mask, positions)
            dec_layer_states = []
        h_raw_dec = h_all[:, n + ll : 2 * n, :]  # (R, H_patches, d_model)
        if self.jepa_predictor is not None:
            h_raw_dec = self.jepa_predictor(h_all[:, :ll, :], h_raw_dec, key_valid[:, :ll], first_obs)
        h_aligned, z_rot = h_raw_dec, h_raw_dec

        anchor_loc = loc.unflatten(-1, (n, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = scale.unflatten(-1, (n, ps))[..., 0][:, ll - 1 : ll]
        if return_layer_states:
            return h_aligned, z_rot, h_raw_dec, anchor_loc, anchor_scale, dec_layer_states
        return h_aligned, z_rot, h_raw_dec, anchor_loc, anchor_scale

    def predict_block(
        self,
        inputs: Tensor,
        observed: Tensor,
        group_ids: Tensor,
        horizon_patches: int,
        role_ids: Optional[Tensor] = None,
        known_future: Optional[Tensor] = None,
        known_future_mask: Optional[Tensor] = None,
    ) -> Tensor:
        h_aligned, _, _, anchor_loc, anchor_scale = self.encode_penultimate(
            inputs, observed, horizon_patches, group_ids, role_ids, known_future, known_future_mask
        )
        preds_norm = self.head(h_aligned)
        delta = getattr(self, "readout_delta", None)
        if delta is not None and delta.shape[0] == preds_norm.shape[0]:
            corr = torch.einsum("rhd,rdp->rhp", h_aligned, delta.to(h_aligned.dtype)).reshape(preds_norm.shape[0], -1)
            preds_norm = preds_norm + corr[..., None].to(preds_norm.dtype)
        return denormalize_quantiles(preds_norm, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), self.config.use_asinh)

    def _refine_block(
        self, quantiles: Tensor, context: Tensor, mask: Tensor, target_rows: Optional[Tensor] = None
    ) -> Tensor:
        ps = self.config.patch_size
        n_pred = quantiles.shape[1] // ps
        if n_pred <= 1:
            return quantiles
        _, loc, scale = self.scale(context, mask)
        loc0, scale0 = loc[:, -1], scale[:, -1]
        med_idx = self.config.median_index
        median = quantiles[..., med_idx].unflatten(-1, (n_pred, ps)).to(torch.float64)
        sinh_median = (median - loc0[:, None, None]) / scale0[:, None, None]
        loc_j, scale_j = advancing_block_frames(context, mask, sinh_median, self.config.minimum_scale)
        first = torch.zeros(loc_j.shape, dtype=torch.bool, device=loc_j.device)
        first[:, 0] = True
        loc_j = torch.where(first, loc0[:, None], loc_j.to(quantiles.dtype)).repeat_interleave(ps, dim=-1)
        scale_j = torch.where(first, scale0[:, None], scale_j.to(quantiles.dtype)).repeat_interleave(ps, dim=-1)
        sinh_norm = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
        refined = clamp_nonfinite(sinh_norm * scale_j[..., None] + loc_j[..., None])
        if target_rows is not None:
            refined = torch.where(target_rows[:, None, None], refined, quantiles)
        return refined

    def _assemble_rows(
        self,
        inputs: Tensor,
        observation_mask: Tensor,
        prediction_length: int,
        past_covariates: Optional[Tensor],
        known_covariates: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], int]:
        ps = self.config.patch_size
        device = inputs.device
        b, n_tgt, ctx0 = inputs.shape
        horizon = math.ceil(prediction_length / ps) * ps

        ctx_rows = [ffill_with_mask(inputs, observation_mask)]
        ctx_masks = [observation_mask]
        roles = [ROLE_TARGET] * n_tgt

        n_past = 0
        if past_covariates is not None and past_covariates.shape[1] > 0:
            n_past = past_covariates.shape[1]
            ctx_masks.append(torch.isfinite(past_covariates))
            ctx_rows.append(torch.nan_to_num(past_covariates, nan=0.0))
            roles += [ROLE_PAST_COV] * n_past

        n_known = 0
        known_future_vals: Optional[Tensor] = None
        known_future_obs: Optional[Tensor] = None
        if known_covariates is not None and known_covariates.shape[1] > 0:
            n_known = known_covariates.shape[1]
            kc_ctx, kc_fut = known_covariates[..., :-prediction_length], known_covariates[..., -prediction_length:]
            ctx_masks.append(torch.isfinite(kc_ctx))
            ctx_rows.append(torch.nan_to_num(kc_ctx, nan=0.0))
            roles += [ROLE_KNOWN_COV] * n_known
            known_future_obs = torch.isfinite(kc_fut)
            known_future_vals = torch.nan_to_num(kc_fut, nan=0.0)

        rows = torch.cat(ctx_rows, dim=1)
        mask = torch.cat(ctx_masks, dim=1)
        if rows.shape[-1] > self.config.context_length:
            rows = rows[:, :, -self.config.context_length :]
            mask = mask[:, :, -self.config.context_length :]
        rows, mask = pad_to_patch(rows, mask, ps)
        v_all = rows.shape[1]
        if n_past == 0 and n_known == 0:
            return rows, mask, None, None, None, n_tgt

        role_ids = torch.tensor(roles, dtype=torch.long, device=device).repeat(b)
        known_future: Optional[Tensor] = None
        known_future_mask: Optional[Tensor] = None
        if n_known > 0 and known_future_vals is not None and known_future_obs is not None:
            k0 = n_tgt + n_past
            plen = known_future_vals.shape[-1]
            kf = torch.zeros(b, v_all, horizon, device=device, dtype=rows.dtype)
            kfm = torch.zeros(b, v_all, horizon, device=device, dtype=torch.bool)
            kf[:, k0 : k0 + n_known, :plen] = known_future_vals
            kfm[:, k0 : k0 + n_known, :plen] = known_future_obs
            known_future = kf.reshape(b * v_all, horizon)
            known_future_mask = kfm.reshape(b * v_all, horizon)
        return rows, mask, role_ids, known_future, known_future_mask, n_tgt

    def _forecast_raw(
        self,
        inputs: Tensor,
        mask: Tensor,
        prediction_length: int,
        past_covariates: Optional[Tensor] = None,
        known_covariates: Optional[Tensor] = None,
    ) -> Tensor:
        """`(B, V, T)` -> `(B, V, H, Q_21)` with block-AR decoding, covariates, and Welford frame refinement."""
        self.eval()
        ps = self.config.patch_size
        has_variate_op = any(kind != "time" for kind in self.config.block_schedule)
        if not has_variate_op:
            past_covariates = None
            known_covariates = None

        rows, obs_mask, role_ids, known_future, known_future_mask, n_tgt = self._assemble_rows(
            inputs, mask, prediction_length, past_covariates, known_covariates
        )
        b, v, ctx_len = rows.shape
        context = rows.reshape(b * v, ctx_len)
        obs = obs_mask.reshape(b * v, ctx_len)
        group_ids = torch.arange(b, device=inputs.device).repeat_interleave(v)

        known_rows = past_rows = target_rows = None
        if role_ids is not None:
            known_rows = role_ids == ROLE_KNOWN_COV
            past_rows = (role_ids != ROLE_TARGET) & ~known_rows
            target_rows = role_ids == ROLE_TARGET

        # Compute context min/max and last-patch scale for `quantile_real_cap_k` clamping on target rows
        tgt_rows_3d = rows[:, :n_tgt]
        tgt_mask_3d = obs_mask[:, :n_tgt]
        not_obs = ~tgt_mask_3d
        ctx_max = torch.nan_to_num(tgt_rows_3d.masked_fill(not_obs, float("-inf")).amax(-1, keepdim=True), neginf=0.0)
        ctx_min = torch.nan_to_num(tgt_rows_3d.masked_fill(not_obs, float("inf")).amin(-1, keepdim=True), posinf=0.0)
        _, scale0 = causal_patch_loc_scale(
            tgt_rows_3d, tgt_mask_3d, ps, self.config.minimum_scale, self.config.stabilize_with_global
        )
        anchor_scale = scale0[..., -1:] * float(self.config.quantile_real_cap_k)

        blocks: List[Tensor] = []
        remaining = math.ceil(prediction_length / ps)
        max_block = min(self.config.block_size, max(1, int(self.config.per_shot_horizon_patches)))
        med_idx = self.config.median_index
        offset = 0

        while remaining > 0:
            blk = min(max_block, remaining)
            blen = blk * ps
            kf_blk = known_future[:, offset : offset + blen] if known_future is not None else None
            kfm_blk = known_future_mask[:, offset : offset + blen] if known_future_mask is not None else None
            q_blk = self.predict_block(context, obs, group_ids, blk, role_ids, kf_blk, kfm_blk)
            q_blk = self._refine_block(q_blk, context, obs, target_rows)
            blocks.append(q_blk)
            remaining -= blk
            offset += blen
            if remaining <= 0:
                break
            feedback = torch.nan_to_num(q_blk[..., med_idx], nan=0.0, posinf=0.0, neginf=0.0)
            fb_mask = torch.ones_like(feedback, dtype=torch.bool)
            if role_ids is not None and known_rows is not None and past_rows is not None:
                if kf_blk is not None and kfm_blk is not None:
                    feedback = torch.where(known_rows[:, None], kf_blk, feedback)
                    fb_mask = torch.where(known_rows[:, None], kfm_blk, fb_mask)
                feedback = torch.where(past_rows[:, None], torch.zeros_like(feedback), feedback)
                fb_mask = fb_mask & ~past_rows[:, None]
            context = torch.cat([context, feedback], dim=-1)
            obs = torch.cat([obs, fb_mask], dim=-1)
            if context.shape[-1] > self.config.context_length:
                keep = max(ps, (self.config.context_length // ps) * ps)
                context = context[..., -keep:]
                obs = obs[..., -keep:]

        packed = torch.cat(blocks, dim=-2)[..., :prediction_length, :]
        quantiles = packed.reshape(b, v, prediction_length, -1)[:, :n_tgt]
        return quantiles.clamp((ctx_min - anchor_scale).unsqueeze(-1), (ctx_max + anchor_scale).unsqueeze(-1))

    @torch.inference_mode()
    def forecast(
        self,
        context: np.ndarray | Tensor,
        prediction_length: int,
        past_covariates: Optional[np.ndarray | Tensor] = None,
        known_covariates: Optional[np.ndarray | Tensor] = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
    ) -> Tensor:
        """Full zero-shot probabilistic forecast `(B, V, H, 21)` with covariates & reflection averaging."""
        device = next(self.parameters()).device
        x = torch.as_tensor(context, dtype=torch.float32, device=device)
        if x.dim() == 1:
            x = x[None, None]
        elif x.dim() == 2:
            x = x[:, None]
        mask = torch.isfinite(x)
        inputs = torch.where(mask, x, torch.zeros_like(x))

        pc_t: Optional[Tensor] = None
        if past_covariates is not None:
            pc_t = torch.as_tensor(past_covariates, dtype=torch.float32, device=device)
            if pc_t.dim() == 2:
                pc_t = pc_t[:, None]

        kc_t: Optional[Tensor] = None
        if known_covariates is not None:
            kc_t = torch.as_tensor(known_covariates, dtype=torch.float32, device=device)
            if kc_t.dim() == 2:
                kc_t = kc_t[:, None]

        def _single_view(inp: Tensor, msk: Tensor, pc: Optional[Tensor], kc: Optional[Tensor]) -> Tensor:
            q = self._forecast_raw(inp, msk, prediction_length, pc, kc)
            if symmetric_averaging:
                mir = self._forecast_raw(
                    -inp,
                    msk,
                    prediction_length,
                    -pc if pc is not None else None,
                    -kc if kc is not None else None,
                )
                q = 0.5 * (q + (-mir).flip(-1))
            return q

        quantiles = _single_view(inputs, mask, pc_t, kc_t)
        if clip_nonnegative:
            has_obs = mask.any(dim=-1)
            min_obs = torch.where(mask, inputs, torch.full_like(inputs, float("inf"))).amin(dim=-1)
            eligible = has_obs & (min_obs >= 0.0)
            quantiles = torch.where(eligible[..., None, None], quantiles.clamp_min(0.0), quantiles)
        return quantiles

    @torch.inference_mode()
    def predict_series_batch(
        self,
        series_list: List[np.ndarray],
        horizon: int,
        quantile_levels: List[float],
        batch_size: int = 128,
        past_covariates_list: Optional[List[Optional[np.ndarray]]] = None,
        known_covariates_list: Optional[List[Optional[np.ndarray]]] = None,
        max_covariates: int = 4,
    ) -> np.ndarray:
        """Standalone batch inference for arbitrary-length 1D numpy series + optional covariates.

        Returns `(N, horizon, 1 + len(quantile_levels))` where col 0 is median (0.5) and
        cols `1..` correspond to `quantile_levels`.
        """
        self.eval()
        N = len(series_list)
        out = np.zeros((N, horizon, 1 + len(quantile_levels)), dtype=np.float64)
        knots = np.asarray(self.config.quantile_knots, dtype=np.float64)
        med_idx = self.config.median_index

        has_variate_op = any(kind != "time" for kind in self.config.block_schedule)
        use_cov = has_variate_op and (past_covariates_list is not None or known_covariates_list is not None)

        # Group by similar lengths (and covariate channel counts when present) for efficient padding
        def _sort_key(i: int) -> Tuple[int, int, int]:
            c_p = 0
            if use_cov and past_covariates_list is not None and past_covariates_list[i] is not None:
                c_p = min(past_covariates_list[i].shape[1], max_covariates)
            c_k = 0
            if use_cov and known_covariates_list is not None and known_covariates_list[i] is not None:
                c_k = min(known_covariates_list[i].shape[1], max_covariates)
            return (c_p, c_k, len(series_list[i]))

        order = sorted(range(N), key=_sort_key)
        # Covariate rows share dense variate attention across the batch, so cap rows per forward pass.
        step = max(8, batch_size // 8) if use_cov else batch_size
        for start in range(0, N, step):
            idx = order[start : start + step]
            max_t = min(self.config.context_length, max(len(series_list[i]) for i in idx))
            B_cur = len(idx)
            batch_arr = np.full((B_cur, 1, max_t), np.nan, dtype=np.float32)

            c_p_max = 0
            c_k_max = 0
            if use_cov:
                if past_covariates_list is not None:
                    c_p_max = min(max_covariates, max((past_covariates_list[i].shape[1] if past_covariates_list[i] is not None else 0) for i in idx))
                if known_covariates_list is not None:
                    c_k_max = min(max_covariates, max((known_covariates_list[i].shape[1] if known_covariates_list[i] is not None else 0) for i in idx))

            pc_arr = np.full((B_cur, c_p_max, max_t), np.nan, dtype=np.float32) if c_p_max > 0 else None
            kc_arr = np.full((B_cur, c_k_max, max_t + horizon), np.nan, dtype=np.float32) if c_k_max > 0 else None

            for b_i, orig_i in enumerate(idx):
                y = np.asarray(series_list[orig_i], dtype=np.float32)
                if len(y) > max_t:
                    y = y[-max_t:]
                L_i = len(y)
                batch_arr[b_i, 0, max_t - L_i :] = y

                if pc_arr is not None and past_covariates_list is not None and past_covariates_list[orig_i] is not None:
                    xp = np.asarray(past_covariates_list[orig_i], dtype=np.float32)
                    if xp.shape[0] > L_i:
                        xp = xp[-L_i:]
                    nc = min(xp.shape[1], c_p_max)
                    pc_arr[b_i, :nc, max_t - xp.shape[0] :] = xp[:, :nc].T

                if kc_arr is not None and known_covariates_list is not None and known_covariates_list[orig_i] is not None:
                    xk = np.asarray(known_covariates_list[orig_i], dtype=np.float32)
                    # xk has shape (T_orig + horizon, C_k)
                    xk_past = xk[:-horizon]
                    xk_fut = xk[-horizon:]
                    if xk_past.shape[0] > L_i:
                        xk_past = xk_past[-L_i:]
                    nc = min(xk.shape[1], c_k_max)
                    for c_j in range(nc):
                        col_p = xk_past[:, c_j]
                        col_f = xk_fut[:, c_j]
                        # Target-encode low-cardinality discrete/categorical covariates (>2 distinct values, <=12)
                        uniq = np.unique(col_p[np.isfinite(col_p)])
                        if 2 < len(uniq) <= 12 and np.all(np.floor(uniq) == uniq):
                            col_p, col_f_enc = target_encode_covariate(y[-len(col_p):], col_p, col_f)
                            if col_f_enc is not None:
                                col_f = col_f_enc
                        kc_arr[b_i, c_j, max_t - len(col_p) : max_t] = col_p
                        kc_arr[b_i, c_j, max_t : max_t + horizon] = col_f

            q21 = self.forecast(batch_arr, horizon, past_covariates=pc_arr, known_covariates=kc_arr).double().cpu().numpy()[:, 0, :, :]
            for b_i, orig_i in enumerate(idx):
                out[orig_i, :, 0] = q21[b_i, :, med_idx]
                for k_l, lv in enumerate(quantile_levels):
                    match = [k for k, m in enumerate(knots) if abs(m - lv) < 1e-9]
                    if match:
                        out[orig_i, :, 1 + k_l] = q21[b_i, :, match[0]]
                    else:
                        for h_i in range(horizon):
                            out[orig_i, h_i, 1 + k_l] = np.interp(lv, knots, q21[b_i, h_i])
        return out



    def forward(
        self,
        context: Tensor,
        prediction_length: int = 96,
        past_covariates: Optional[Tensor] = None,
        known_covariates: Optional[Tensor] = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
    ) -> Tensor:
        """Forward alias for `forecast(...)` returning `(B, V, prediction_length, 21)` quantiles."""
        return self.forecast(
            context=context,
            prediction_length=prediction_length,
            past_covariates=past_covariates,
            known_covariates=known_covariates,
            symmetric_averaging=symmetric_averaging,
            clip_nonnegative=clip_nonnegative,
        )


TelosForForecasting = TelosModel


def save_telos_checkpoint(model: TelosModel, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": model.config.to_dict(),
        "num_parameters": model.count_parameters(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, p)


def load_telos_checkpoint(path: Path | str, device: str = "cpu") -> TelosModel:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg_dict = dict(payload["config"])
    cfg_dict["model_name"] = "Telos-Small"
    cfg = TelosConfig(**cfg_dict)
    model = TelosModel(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
