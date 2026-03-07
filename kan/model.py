import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.features import local_stencil_indices


DEFAULT_CORRECTION_HEAD_OUTPUT_MODE = "linear"
DEFAULT_CORRECTION_HEAD_OUTPUT_SCALE = 1.0
DEFAULT_USE_STENCIL_FEATURES = True
DEFAULT_STENCIL_RADIUS = 2
DEFAULT_GATE_USE_STENCIL_FEATURES = False
LEGACY_CORRECTION_HEAD_OUTPUT_MODE = "softsign_scaled"
LEGACY_CORRECTION_HEAD_OUTPUT_SCALE = 0.5


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = base_activation()
        self.scale_base = scale_base
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h) + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (self.scale_spline if self.scale_spline is not None else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.spline_weight.view(self.out_features, -1))
        return base_output + spline_output


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _get_branch_cfg(primary_cfg, fallback_cfg, key, default):
    if key in primary_cfg:
        return primary_cfg[key]
    if key in fallback_cfg:
        return fallback_cfg[key]
    return default


def resolve_model_runtime_config(
    *,
    stencil_size,
    artifact_model_cfg=None,
    fallback_model_cfg=None,
    metadata=None,
    prefer_legacy_when_missing=False,
    warn_on_legacy=False,
    source_label="model",
):
    """Resolve artifact-aware model settings without silently changing old behavior."""
    artifact_model_cfg = dict(artifact_model_cfg or {})
    fallback_model_cfg = dict(fallback_model_cfg or {})
    metadata = dict(metadata or {})

    correction_head_meta = dict(metadata.get("correction_head", {}) or {})
    feature_layout_meta = dict(metadata.get("feature_layout", {}) or {})

    correction_head_cfg = {}
    if correction_head_meta:
        correction_head_cfg.update(correction_head_meta)
    if isinstance(artifact_model_cfg.get("correction_head"), dict):
        correction_head_cfg.update(artifact_model_cfg["correction_head"])

    if "shape_output_scale" in artifact_model_cfg and "output_scale" not in correction_head_cfg:
        if warn_on_legacy:
            warnings.warn(
                f"{source_label}: model.shape_output_scale is deprecated; "
                "using legacy softsign_scaled correction head.",
                RuntimeWarning,
                stacklevel=2,
            )
        correction_head_cfg.setdefault("output_mode", LEGACY_CORRECTION_HEAD_OUTPUT_MODE)
        correction_head_cfg["output_scale"] = float(artifact_model_cfg["shape_output_scale"])
    elif (
        "shape_output_scale" in fallback_model_cfg
        and "output_scale" not in correction_head_cfg
        and not correction_head_meta
        and "correction_head" not in artifact_model_cfg
        and "correction_head" not in fallback_model_cfg
    ):
        if warn_on_legacy:
            warnings.warn(
                f"{source_label}: fallback model.shape_output_scale is deprecated; "
                "using legacy softsign_scaled correction head.",
                RuntimeWarning,
                stacklevel=2,
            )
        correction_head_cfg.setdefault("output_mode", LEGACY_CORRECTION_HEAD_OUTPUT_MODE)
        correction_head_cfg["output_scale"] = float(fallback_model_cfg["shape_output_scale"])
    elif not correction_head_cfg and prefer_legacy_when_missing:
        if warn_on_legacy:
            warnings.warn(
                f"{source_label}: correction_head metadata missing; inferring legacy softsign head.",
                RuntimeWarning,
                stacklevel=2,
            )
        correction_head_cfg = {
            "output_mode": LEGACY_CORRECTION_HEAD_OUTPUT_MODE,
            "output_scale": LEGACY_CORRECTION_HEAD_OUTPUT_SCALE,
        }
    elif isinstance(fallback_model_cfg.get("correction_head"), dict) and not correction_head_cfg:
        correction_head_cfg.update(fallback_model_cfg["correction_head"])

    correction_head_output_mode = str(
        correction_head_cfg.get("output_mode", DEFAULT_CORRECTION_HEAD_OUTPUT_MODE)
    ).lower()
    correction_head_output_scale = float(
        correction_head_cfg.get("output_scale", DEFAULT_CORRECTION_HEAD_OUTPUT_SCALE)
    )

    if feature_layout_meta:
        use_stencil_features = bool(
            feature_layout_meta.get("use_stencil_features", DEFAULT_USE_STENCIL_FEATURES)
        )
        gate_use_stencil_features = bool(
            feature_layout_meta.get("gate_use_stencil_features", DEFAULT_GATE_USE_STENCIL_FEATURES)
        )
        raw_radius = feature_layout_meta.get("stencil_radius", DEFAULT_STENCIL_RADIUS)
    elif any(
        key in artifact_model_cfg for key in ("use_stencil_features", "stencil_radius", "gate_use_stencil_features")
    ):
        use_stencil_features = bool(
            artifact_model_cfg.get("use_stencil_features", DEFAULT_USE_STENCIL_FEATURES)
        )
        gate_use_stencil_features = bool(
            artifact_model_cfg.get("gate_use_stencil_features", DEFAULT_GATE_USE_STENCIL_FEATURES)
        )
        raw_radius = artifact_model_cfg.get("stencil_radius", DEFAULT_STENCIL_RADIUS)
    elif prefer_legacy_when_missing:
        if warn_on_legacy:
            warnings.warn(
                f"{source_label}: feature_layout metadata missing; inferring legacy full-stencil correction path.",
                RuntimeWarning,
                stacklevel=2,
            )
        use_stencil_features = True
        gate_use_stencil_features = False
        raw_radius = int(stencil_size) // 2
    else:
        use_stencil_features = bool(
            _get_branch_cfg(
                artifact_model_cfg,
                fallback_model_cfg,
                "use_stencil_features",
                DEFAULT_USE_STENCIL_FEATURES,
            )
        )
        gate_use_stencil_features = bool(
            _get_branch_cfg(
                artifact_model_cfg,
                fallback_model_cfg,
                "gate_use_stencil_features",
                DEFAULT_GATE_USE_STENCIL_FEATURES,
            )
        )
        raw_radius = _get_branch_cfg(
            artifact_model_cfg,
            fallback_model_cfg,
            "stencil_radius",
            DEFAULT_STENCIL_RADIUS,
        )

    stencil_radius = int(raw_radius)

    return {
        "hidden_dim": int(_get_branch_cfg(artifact_model_cfg, fallback_model_cfg, "hidden_dim", 32)),
        "shape_grid_size": int(_get_branch_cfg(artifact_model_cfg, fallback_model_cfg, "shape_grid_size", 10)),
        "shape_spline_order": int(
            _get_branch_cfg(artifact_model_cfg, fallback_model_cfg, "shape_spline_order", 3)
        ),
        "gate_hidden_dims": tuple(_get_branch_cfg(artifact_model_cfg, fallback_model_cfg, "gate_hidden_dims", [32, 16])),
        "gate_temperature": float(
            _get_branch_cfg(artifact_model_cfg, fallback_model_cfg, "gate_temperature", 2.0)
        ),
        "gate_bias_init": float(_get_branch_cfg(artifact_model_cfg, fallback_model_cfg, "gate_bias_init", -1.0)),
        "shock_indicator_threshold": float(
            _get_branch_cfg(artifact_model_cfg, fallback_model_cfg, "shock_indicator_threshold", 0.15)
        ),
        "curvature_eps": float(_get_branch_cfg(artifact_model_cfg, fallback_model_cfg, "curvature_eps", 1e-4)),
        "correction_head_output_mode": correction_head_output_mode,
        "correction_head_output_scale": correction_head_output_scale,
        "use_stencil_features": bool(use_stencil_features),
        "stencil_radius": int(stencil_radius),
        "gate_use_stencil_features": bool(gate_use_stencil_features),
    }


class GatedKAN(nn.Module):
    def __init__(
        self,
        stencil_size=9,
        phys_dim=3,
        hidden_dim=32,
        shape_grid_size=10,
        shape_spline_order=3,
        shape_output_scale=None,
        correction_head_output_mode=DEFAULT_CORRECTION_HEAD_OUTPUT_MODE,
        correction_head_output_scale=DEFAULT_CORRECTION_HEAD_OUTPUT_SCALE,
        use_stencil_features=DEFAULT_USE_STENCIL_FEATURES,
        stencil_radius=DEFAULT_STENCIL_RADIUS,
        gate_use_stencil_features=DEFAULT_GATE_USE_STENCIL_FEATURES,
        gate_hidden_dims=(32, 16),
        gate_temperature=2.0,
        gate_bias_init=-1.0,
        shock_indicator_threshold=0.15,
        curvature_eps=1e-4,
    ):
        super(GatedKAN, self).__init__()
        self.stencil_size = int(stencil_size)
        self.phys_dim = int(phys_dim)
        self.use_stencil_features = bool(use_stencil_features)
        self.gate_use_stencil_features = bool(gate_use_stencil_features)
        self.shock_indicator_threshold = float(shock_indicator_threshold)
        self.curvature_eps = float(curvature_eps)
        if shape_output_scale is not None:
            warnings.warn(
                "model.shape_output_scale is deprecated; using legacy softsign_scaled correction head.",
                RuntimeWarning,
                stacklevel=2,
            )
            correction_head_output_mode = LEGACY_CORRECTION_HEAD_OUTPUT_MODE
            correction_head_output_scale = float(shape_output_scale)

        self.correction_head_output_mode = str(correction_head_output_mode).lower()
        self.correction_head_output_scale = float(correction_head_output_scale)
        self.correction_stencil_indices = tuple(
            local_stencil_indices(self.stencil_size, int(stencil_radius))
            if self.use_stencil_features
            else []
        )
        self.gate_stencil_indices = tuple(self.correction_stencil_indices if self.gate_use_stencil_features else [])

        correction_mask = torch.zeros(self.stencil_size, dtype=torch.float32)
        if self.use_stencil_features:
            correction_mask[list(self.correction_stencil_indices)] = 1.0
        self.register_buffer("correction_stencil_mask", correction_mask.unsqueeze(0))

        # 纠正分支显式使用局部 stencil；非局部位置置零，避免把平滑区也一并放大。
        self.shape_net = KAN(
            [self.stencil_size + self.phys_dim, hidden_dim, 1],
            grid_size=shape_grid_size,
            spline_order=shape_spline_order,
        )

        # Gate 默认仍以 physics 特征为主，只在显式开启时追加局部 stencil。
        gate_input_dim = self.phys_dim + 2 + len(self.gate_stencil_indices)
        gate_hidden_dims = list(gate_hidden_dims)
        if len(gate_hidden_dims) == 0:
            gate_hidden_dims = [16]

        gate_layers = []
        last_dim = gate_input_dim
        for h_dim in gate_hidden_dims:
            gate_layers.append(nn.Linear(last_dim, int(h_dim)))
            gate_layers.append(nn.SiLU())
            last_dim = int(h_dim)
        gate_layers.append(nn.Linear(last_dim, 1))
        self.gate_net = nn.Sequential(*gate_layers)

        self.gate_temperature = float(gate_temperature)

        if hasattr(self.gate_net[-1], 'bias') and self.gate_net[-1].bias is not None:
            nn.init.constant_(self.gate_net[-1].bias, float(gate_bias_init))

    def _select_correction_input(self, x):
        x_stencil_full = x[:, : self.stencil_size]
        x_physics = x[:, self.stencil_size :]
        masked_stencil = x_stencil_full * self.correction_stencil_mask.to(dtype=x.dtype)
        return masked_stencil, x_physics

    def _build_gate_input(self, x_physics, x_stencil_full=None):
        if self.phys_dim > 1:
            abs_ux = x_physics[:, 1:2]
        else:
            abs_ux = torch.abs(x_physics[:, 0:1])

        if self.phys_dim > 2:
            abs_uxx = x_physics[:, 2:3]
        else:
            abs_uxx = torch.zeros_like(abs_ux)

        curvature_ratio = abs_uxx / (abs_ux + self.curvature_eps)
        shock_indicator = F.relu(abs_ux - self.shock_indicator_threshold)
        gate_inputs = [x_physics, curvature_ratio, shock_indicator]
        if self.gate_use_stencil_features and x_stencil_full is not None and self.gate_stencil_indices:
            gate_inputs.append(x_stencil_full[:, list(self.gate_stencil_indices)])
        return torch.cat(gate_inputs, dim=1)

    def _apply_correction_head(self, raw_correction):
        # 中文说明：旧版 softsign*0.5 在激波区会过早饱和，这里改成可配置 head。
        if self.correction_head_output_mode == "linear":
            return raw_correction * self.correction_head_output_scale
        if self.correction_head_output_mode == "tanh_scaled":
            return torch.tanh(raw_correction) * self.correction_head_output_scale
        if self.correction_head_output_mode == "softsign_scaled":
            return F.softsign(raw_correction) * self.correction_head_output_scale
        raise ValueError(f"Unsupported correction_head output_mode: {self.correction_head_output_mode}")

    @property
    def correction_head_is_bounded(self):
        return self.correction_head_output_mode in {"tanh_scaled", "softsign_scaled"}

    def forward_components(self, x):
        x_stencil_full, x_physics = self._select_correction_input(x)
        shape_input = torch.cat([x_stencil_full, x_physics], dim=1)

        raw_correction = self.shape_net(shape_input)
        raw_correction = self._apply_correction_head(raw_correction)

        gate_logits = self.gate_net(self._build_gate_input(x_physics, x_stencil_full))
        raw_gate = torch.sigmoid(gate_logits / self.gate_temperature)
        gated_correction = raw_correction * raw_gate
        return raw_correction, raw_gate, gated_correction

    def forward(self, x):
        """训练默认路径：直接返回与训练一致的 gated correction。"""
        _, gate, gated_correction = self.forward_components(x)
        return gated_correction, gate
