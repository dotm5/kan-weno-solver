import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class GatedKAN(nn.Module):
    def __init__(
        self,
        stencil_size=9,
        phys_dim=3,
        hidden_dim=32,
        shape_grid_size=10,
        shape_spline_order=3,
        shape_output_scale=0.5,
        gate_hidden_dims=(32, 16),
        gate_temperature=2.0,
        gate_bias_init=-1.0,
        shock_indicator_threshold=0.15,
        curvature_eps=1e-4,
    ):
        super(GatedKAN, self).__init__()
        self.stencil_size = stencil_size
        self.phys_dim = phys_dim
        self.shape_output_scale = float(shape_output_scale)
        self.shock_indicator_threshold = float(shock_indicator_threshold)
        self.curvature_eps = float(curvature_eps)

        # Shape Net: full state input.
        self.shape_net = KAN(
            [stencil_size + phys_dim, hidden_dim, 1],
            grid_size=shape_grid_size,
            spline_order=shape_spline_order,
        )

        # Gate gets physics + derived gradient descriptors to avoid early saturation.
        gate_input_dim = phys_dim + 2
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

    def _build_gate_input(self, x_physics):
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
        return torch.cat([x_physics, curvature_ratio, shock_indicator], dim=1)

    def forward(self, x):
        x_physics = x[:, self.stencil_size:]

        raw_correction = self.shape_net(x)
        raw_correction = F.softsign(raw_correction) * self.shape_output_scale

        gate_logits = self.gate_net(self._build_gate_input(x_physics))
        gate = torch.sigmoid(gate_logits / self.gate_temperature)

        return raw_correction * gate, gate
