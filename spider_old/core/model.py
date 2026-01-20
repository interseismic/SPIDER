import torch
import torch.nn as nn
from typing import Any, Dict

class EikoNet(nn.Module):
    def __init__(self, scale, vs=3.3, vp=6.0, n_hidden=128, n_blocks=5):
        super(EikoNet, self).__init__()
        self.scale = scale
        self.vs = vs
        self.vp = vp
        self.n_hidden = int(n_hidden)
        self.n_blocks = int(n_blocks)
        
        self.activation = nn.ELU()
        
        # Input layer (spatial 4D -> hidden)
        self.input_layer = nn.Linear(4, self.n_hidden)
        
        # Residual blocks
        # Each block contains 2 linear layers and a residual connection
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(nn.ModuleList([
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.Linear(self.n_hidden, self.n_hidden)
            ]))
            
        # Output layer
        self.output_layer = nn.Linear(self.n_hidden, 1)

    def T0(self, x):
        # Analytical T0 computation
        # x: [batch, 7] where 0-2 is source, 3-5 is receiver, 6 is phase
        # phase >= 0.5 is S-wave (vs), else P-wave (vp)
        scalar = torch.where(x[:, 6] >= 0.5, self.vs, self.vp).unsqueeze(1)
        dist = torch.sqrt(((x[:, 0:3] - x[:, 3:6])**2).sum(dim=1)).unsqueeze(dim=1)
        return dist / scalar

    def T1(self, x):
        # Neural network correction
        # Preprocess input: relative coordinates and phase
        r = torch.sqrt(((x[:, 0:2] - x[:, 3:5])**2).sum(dim=1)).unsqueeze(1)
        
        # Inputs: r, src_z, rec_z, phase (4 inputs)
        inputs = torch.cat((r, x[:, 2:3], x[:, 5:]), dim=1)
        
        # Scale coordinates (first 3 columns: r, src_z, rec_z)
        # Note: This matches the legacy behavior where x[:,:3] /= scale
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
        out = torch.abs(out)
        
        return out

    def forward(self, x):
        return self.T0(x) * self.T1(x)

    def EikonalPDE(self, x):
        x.requires_grad_()
        T = self.forward(x)
        dT_dx = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        s_rec = (dT_dx[:, 3:6]**2).sum(dim=1).sqrt()
        return s_rec


def _looks_like_legacy_eikonet_state_dict(sd: Dict[str, Any]) -> bool:
    try:
        keys = list(sd.keys())
    except Exception:
        return False
    return any(k.startswith("linear1.") or k.startswith("linear_out.") for k in keys)


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
    """Load an EikoNet model from a *weights-only* checkpoint.

    Supported checkpoint formats:
    - Raw state_dict: {"input_layer.weight": ..., ...}
    - Wrapped dict: {"state_dict": <state_dict>, "meta": {...}}

    Legacy pickled `torch.nn.Module` checkpoints are intentionally not supported.
    Use `scripts/migrate_legacy_eikonet_checkpoint.py` to convert them.
    """
    try:
        obj = torch.load(model_file, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load EikoNet weights-only checkpoint '{model_file}'. "
            "If this is a legacy pickled model, convert it with "
            "`python scripts/migrate_legacy_eikonet_checkpoint.py --in <old.pt> --out <new_state_dict.pt>`."
        ) from e

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
    if _looks_like_legacy_eikonet_state_dict(sd):
        raise RuntimeError(
            f"Legacy EikoNet parameter names detected in '{model_file}'. "
            "Convert it with `scripts/migrate_legacy_eikonet_checkpoint.py`."
        )

    scale = float(meta.get("scale", default_scale))
    vs = float(meta.get("vs", default_vs))
    vp = float(meta.get("vp", default_vp))
    n_hidden = int(meta.get("n_hidden", default_n_hidden))
    n_blocks = int(meta.get("n_blocks", default_n_blocks))

    model = EikoNet(scale=scale, vs=vs, vp=vp, n_hidden=n_hidden, n_blocks=n_blocks)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)
    model = model.float()
    return model
