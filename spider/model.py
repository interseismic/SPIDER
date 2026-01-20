from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class EikoNet(nn.Module):
    def __init__(self, scale: float, vs: float = 3.3, vp: float = 6.0, n_hidden: int = 128, n_blocks: int = 5):
        super().__init__()
        self.scale = float(scale)
        self.vs = float(vs)
        self.vp = float(vp)
        self.n_hidden = int(n_hidden)
        self.n_blocks = int(n_blocks)

        self.activation = nn.ELU()
        self.input_layer = nn.Linear(4, self.n_hidden)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(nn.ModuleList([nn.Linear(self.n_hidden, self.n_hidden), nn.Linear(self.n_hidden, self.n_hidden)]))
        self.output_layer = nn.Linear(self.n_hidden, 1)

    def T0(self, x: torch.Tensor) -> torch.Tensor:
        scalar = torch.where(x[:, 6] >= 0.5, self.vs, self.vp).unsqueeze(1)
        dist = torch.sqrt(((x[:, 0:3] - x[:, 3:6]) ** 2).sum(dim=1)).unsqueeze(dim=1)
        return dist / scalar

    def T1(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(((x[:, 0:2] - x[:, 3:5]) ** 2).sum(dim=1)).unsqueeze(1)
        inputs = torch.cat((r, x[:, 2:3], x[:, 5:]), dim=1)
        inputs[:, :3] = inputs[:, :3] / self.scale
        out = self.input_layer(inputs)
        out = self.activation(out)
        for layer1, layer2 in self.blocks:
            residual = out
            out = layer1(out)
            out = self.activation(out)
            out = layer2(out)
            out = self.activation(out) + residual
        out = self.output_layer(out)
        return torch.abs(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.T0(x) * self.T1(x)


def load_eikonet_state_dict(
    model_file: str,
    *,
    device: torch.device | str,
    default_scale: float,
    default_vs: float = 3.3,
    default_vp: float = 6.0,
    default_n_hidden: int = 128,
    default_n_blocks: int = 5,
) -> EikoNet:
    try:
        obj = torch.load(model_file, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load EikoNet weights-only checkpoint '{model_file}'.") from e

    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict checkpoint for '{model_file}'; got {type(obj)}")

    meta: Dict[str, Any] = {}
    sd: Any = obj
    if "state_dict" in obj:
        sd = obj.get("state_dict", None)
        meta_raw = obj.get("meta", {})
        if isinstance(meta_raw, dict):
            meta = meta_raw

    if not isinstance(sd, dict):
        raise TypeError(f"Expected state_dict to be a dict in '{model_file}'; got {type(sd)}")

    scale = float(meta.get("scale", default_scale))
    vs = float(meta.get("vs", default_vs))
    vp = float(meta.get("vp", default_vp))
    n_hidden = int(meta.get("n_hidden", default_n_hidden))
    n_blocks = int(meta.get("n_blocks", default_n_blocks))

    model = EikoNet(scale=scale, vs=vs, vp=vp, n_hidden=n_hidden, n_blocks=n_blocks)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)
    return model.float()
