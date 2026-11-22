"""Hugging Face `PreTrainedConfig` for Telos (`Telos-Small`, 50.1M parameters)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from transformers import PretrainedConfig

QUANTILE_KNOTS_21: Tuple[float, ...] = (
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
)


class TelosConfig(PretrainedConfig):
    """Configuration class for `TelosModel` (`Telos-Small`, 50.1M Time-Series JEPA Foundation Model)."""

    model_type = "telos"

    def __init__(
        self,
        model_name: str = "Telos-Small",
        patch_size: int = 32,
        d_model: int = 512,
        num_heads: int = 8,
        mlp_hidden_dim: int = 1152,
        patch_proj_hidden: int = 1536,
        head_hidden_dim: int = 1792,
        block_schedule: Sequence[str] = (
            "time", "time", "time", "time_variate",
            "time", "time", "time", "time_variate",
            "time", "time", "time", "time_variate",
        ),
        quantile_knots: Sequence[float] = QUANTILE_KNOTS_21,
        block_size: int = 32,
        context_length: int = 4096,
        max_context_length: int = 8192,
        rope_theta: float = 10000.0,
        rope_partial_factor: float = 0.5,
        norm_eps: float = 1e-6,
        attn_bias: bool = False,
        mlp_bias: bool = False,
        time_attn_gate: str = "sdpa_sigmoid",
        variate_attn_gate: str = "sdpa_sigmoid",
        minimum_scale: float = 1e-6,
        use_asinh: bool = True,
        stabilize_with_global: bool = False,
        covariate_enabled: bool = True,
        split_context_horizon_attn: bool = True,
        attend_missing_patches: bool = True,
        quantile_width_factor: float = 3.0,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 6.0,
        jepa_predictor_layers: int = 2,
        jepa_predictor_hidden: int = 1024,
        unet_skip_connections: bool = True,
        value_residual: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.patch_proj_hidden = int(patch_proj_hidden)
        self.head_hidden_dim = int(head_hidden_dim)
        self.block_schedule: List[str] = list(block_schedule)
        self.quantile_knots: List[float] = [float(q) for q in quantile_knots]
        self.block_size = int(block_size)
        self.context_length = int(context_length)
        self.max_context_length = int(max_context_length)
        self.rope_theta = float(rope_theta)
        self.rope_partial_factor = float(rope_partial_factor)
        self.norm_eps = float(norm_eps)
        self.attn_bias = bool(attn_bias)
        self.mlp_bias = bool(mlp_bias)
        self.time_attn_gate = str(time_attn_gate)
        self.variate_attn_gate = str(variate_attn_gate)
        self.minimum_scale = float(minimum_scale)
        self.use_asinh = bool(use_asinh)
        self.stabilize_with_global = bool(stabilize_with_global)
        self.covariate_enabled = bool(covariate_enabled)
        self.split_context_horizon_attn = bool(split_context_horizon_attn)
        self.attend_missing_patches = bool(attend_missing_patches)
        self.quantile_width_factor = float(quantile_width_factor)
        self.per_shot_horizon_patches = int(per_shot_horizon_patches)
        self.quantile_real_cap_k = float(quantile_real_cap_k)
        self.jepa_predictor_layers = int(jepa_predictor_layers)
        self.jepa_predictor_hidden = int(jepa_predictor_hidden)
        self.unet_skip_connections = bool(unet_skip_connections)
        self.value_residual = bool(value_residual)

    @property
    def num_quantiles(self) -> int:
        return len(self.quantile_knots)

    @property
    def median_index(self) -> int:
        return list(self.quantile_knots).index(0.5)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def rope_dim(self) -> int:
        rd = int(self.head_dim * self.rope_partial_factor)
        return max(2, rd - (rd % 2))
