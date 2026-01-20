from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Optional

import torch

from .config import load_params, validate_and_materialize
from .pipeline import locate_map, sample_from_bundle
from .utils.console import info, warn


def _device_from_id(device_id: int) -> torch.device:
    try:
        device_id = int(device_id)
    except Exception:
        device_id = -1
    if device_id < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    warn("CUDA is not available; running on CPU.", section="RUN")
    return torch.device("cpu")


def _load_params(path: str) -> dict:
    params = load_params(path)
    return validate_and_materialize(params)


def _cmd_locate_map(args: argparse.Namespace) -> int:
    params = _load_params(args.params)
    device_id = int(args.device) if args.device is not None else int(params["devices"][0])
    bundle_out = getattr(args, "bundle_out", None)
    locate_map(params=params, device_id=device_id, bundle_out=bundle_out)
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    params = _load_params(args.params)
    device_id = int(args.device) if args.device is not None else int(params["devices"][0])
    bundle_path = getattr(args, "bundle", None)
    if not bundle_path:
        ckpt_dir = str(params.get("checkpoint_dir", "")) or "."
        bundle_path = os.path.join(ckpt_dir, "phase2_bundle.pth")
    sample_from_bundle(params=params, device_id=device_id, bundle_path=bundle_path)
    return 0


def _cmd_sample_multi(args: argparse.Namespace) -> int:
    with open(args.params, "r") as f:
        base_params = json.load(f)
    params = validate_and_materialize(base_params)
    devs = None
    if getattr(args, "devices", None):
        devs = [int(x.strip()) for x in str(args.devices).split(",") if str(x).strip()]
    else:
        devs = [int(x) for x in params.get("devices", [])]
    if not devs:
        raise ValueError("sample-multi requires devices (either --devices or inference.compute.devices in params).")
    bundle_path = str(getattr(args, "bundle", "") or "").strip()
    if not bundle_path:
        ckpt_dir = str(params.get("checkpoint_dir", "")) or "."
        bundle_path = os.path.join(ckpt_dir, "phase2_bundle.pth")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Phase-2 bundle not found: {bundle_path}")

    n_chains = int(args.chains) if getattr(args, "chains", None) is not None else int(len(devs))
    n_chains = max(1, n_chains)
    seed0 = int(getattr(args, "seed0", 0))
    out_dir = str(getattr(args, "out_dir", "") or os.path.dirname(os.path.abspath(args.params)) or ".")
    os.makedirs(out_dir, exist_ok=True)

    procs = []
    for ci in range(n_chains):
        dev = int(devs[ci % len(devs)])
        p = json.loads(json.dumps(base_params))
        p.setdefault("inference", {})
        p["inference"].setdefault("compute", {})
        p["inference"]["compute"]["devices"] = [dev]
        p.setdefault("runtime", {})
        p["runtime"]["seed"] = int(seed0 + 1000003 * ci)

        p.setdefault("io", {})
        samp = str(p["io"].get("samples_outfile", os.path.join(out_dir, "SPIDER_samples.h5")))
        root, ext = os.path.splitext(samp)
        if not ext:
            ext = ".h5"
        p["io"]["samples_outfile"] = f"{root}_chain{ci}{ext}"
        ck = str(p["io"].get("checkpoint_dir", os.path.join(out_dir, "checkpoints")))
        p["io"]["checkpoint_dir"] = os.path.join(ck, f"chain{ci}")

        fd, tmp_path = tempfile.mkstemp(prefix=f"spider_chain{ci}_", suffix=".json", dir=out_dir)
        os.close(fd)
        with open(tmp_path, "w") as f:
            json.dump(p, f, indent=4)
        cmd = [sys.executable, "-m", "spider", "sample", tmp_path, "--device", str(dev), "--bundle", str(bundle_path)]
        print(f"[sample-multi] chain={ci} device={dev} seed={p['runtime']['seed']} samples={p['io']['samples_outfile']}")
        if getattr(args, "dry_run", False):
            continue
        procs.append(subprocess.Popen(cmd))

    if getattr(args, "dry_run", False):
        return 0
    rc = 0
    for pr in procs:
        r = pr.wait()
        rc = rc if rc != 0 else int(r)
    return int(rc)


def _cmd_validate(args: argparse.Namespace) -> int:
    _ = _load_params(args.params)
    info("Config validation passed.", section="CONFIG")
    return 0


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog or "spider", description="Clean SPIDER command-line interface")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    p_map = subparsers.add_parser("locate-map", help="Run Phase 1 (MAP) only and dump a Phase-2 bundle")
    p_map.add_argument("params", help="Path to parameter JSON file")
    p_map.add_argument("--device", type=int, default=None, help="CUDA device id to use")
    p_map.add_argument("--bundle-out", type=str, default=None, help="Output path for Phase-2 bundle")
    p_map.set_defaults(func=_cmd_locate_map)

    p_samp = subparsers.add_parser("sample", help="Run Phase 2–4 (sampling) starting from a Phase-2 bundle")
    p_samp.add_argument("params", help="Path to parameter JSON file")
    p_samp.add_argument("--device", type=int, default=None, help="CUDA device id to use")
    p_samp.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle")
    p_samp.set_defaults(func=_cmd_sample)

    p_sm = subparsers.add_parser("sample-multi", help="Launch multiple independent sampling chains")
    p_sm.add_argument("params", help="Path to parameter JSON file")
    p_sm.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle")
    p_sm.add_argument("--chains", type=int, default=None, help="Number of chains to run")
    p_sm.add_argument("--devices", type=str, default=None, help="Comma-separated CUDA device ids")
    p_sm.add_argument("--seed0", type=int, default=0, help="Base seed for chain RNG offsets")
    p_sm.add_argument("--out-dir", type=str, default=None, help="Directory to write chain param files")
    p_sm.add_argument("--dry-run", action="store_true", help="Print what would run, but do not start processes")
    p_sm.set_defaults(func=_cmd_sample_multi)

    p_val = subparsers.add_parser("validate", help="Validate a config against the clean schema")
    p_val.add_argument("params", help="Path to parameter JSON file")
    p_val.set_defaults(func=_cmd_validate)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
