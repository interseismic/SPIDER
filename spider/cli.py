import argparse
import json
import os
import sys
import subprocess
import tempfile
from typing import Optional

import torch
import torch.distributed as dist
import numpy as np
import polars as pl
from pyproj import Proj

from spider.core import prepare_input_dfs
from spider.core.locate import locate_all, locate_map, locate_sample_from_bundle
from spider.core.model import load_eikonet_state_dict
from spider.core.modeling import compute_travel_times
from spider.io.synth import synth_initial_catalog_from_truth
from spider.io.phase_bundle import load_phase2_bundle
from spider.utils import init_wandb_if_enabled
from spider.utils.console import info, warn
from spider.core.data import prepare_stations
from spider.core.analyze_resid import analyze_resid_from_bundle
from spider.core.config_schema import (
	validate_and_materialize_block1,
	validate_and_materialize_block2,
	validate_and_materialize_block3,
	validate_and_materialize_block4,
	validate_and_materialize_block5,
)
from spider.core.priors_config import validate_and_materialize_priors


def _device_from_id(device_id: int) -> torch.device:
	"""
	Map an internal device id to a torch.device.

	Convention:
	  - device_id >= 0 => CUDA device id (if CUDA available), else CPU fallback
	  - device_id < 0  => force CPU
	"""
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


def _parse_device_entry(x) -> int:
	"""
	Parse a device entry from config into our canonical int representation:
	  - CUDA device ids: non-negative ints
	  - CPU: -1 (accepts -1 or 'cpu' or 'cuda:N' or numeric strings)
	"""
	if x is None:
		raise ValueError("Device entry is null")
	if isinstance(x, int):
		return int(x)
	s = str(x).strip().lower()
	if s == "cpu":
		return -1
	if s.startswith("cuda:"):
		s2 = s.split("cuda:", 1)[1].strip()
		if not s2.isdigit():
			raise ValueError(f"Invalid device specifier {x!r} (expected 'cuda:<int>')")
		return int(s2)
	# Accept numeric strings
	if s.lstrip("-").isdigit():
		return int(s)
	raise ValueError(f"Invalid device specifier {x!r} (supported: int, -1/'cpu', 'cuda:<int>')")


def _load_model(params: dict, device: int | str) -> torch.nn.Module:
	"""
	Load and place the neural network model as configured by 'model_file'.
	"""
	model_file = params["model_file"]
	scale = float(params.get("scale", 1.0))
	n_hidden = int(params.get("n_hidden", 128))
	n_blocks = int(params.get("n_blocks", 5))
	return load_eikonet_state_dict(
		model_file,
		device=device,
		default_scale=scale,
		default_n_hidden=n_hidden,
		default_n_blocks=n_blocks,
	)


def _cmd_locate_full(args: argparse.Namespace) -> int:
	"""Legacy: run the full pipeline (Phase 1 + Phase 2–4)."""
	# Load params JSON
	with open(args.params, "r") as f:
		params = json.load(f)

	# Strict block1 config (nested-only; no defaults; no legacy keys)
	params = validate_and_materialize_block1(params)
	# Strict block2 config (nested-only; no defaults; no legacy keys)
	params = validate_and_materialize_block2(params)
	# Strict block3 config (nested-only; no defaults; no legacy keys)
	params = validate_and_materialize_block3(params)
	# Strict block4 config (nested-only; no defaults; no legacy keys)
	params = validate_and_materialize_block4(params)
	# Strict block5 config (nested-only; no defaults; no legacy keys)
	params = validate_and_materialize_block5(params)
	# Strict priors config (nested-only; no defaults; no legacy keys)
	params = validate_and_materialize_priors(params)

	# Device selection:
	# - `spider locate` is single-device.
	# - Multi-GPU should use `spider locate-multi` (one process per GPU).
	dev_list = list(params.get("devices", []))
	if not dev_list:
		raise ValueError("No devices configured. Set compute.devices in the config, or pass --device.")
	if args.device is None:
		if len(dev_list) != 1:
			raise ValueError(
				f"Config compute.devices has {len(dev_list)} entries but `spider locate` is single-device. "
				"Use `spider locate-multi` for multi-GPU, or pass --device to pick one GPU."
			)
		device_id = _parse_device_entry(dev_list[0])
	else:
		device_id = int(args.device)
		# Keep the materialized device list consistent with the explicit override.
		params["devices"] = [device_id]

	# Resolve to an actual torch.device (robust to CPU-only builds).
	device = _device_from_id(int(device_id))

	# Inject optional shift guard settings into params for core pipeline
	if getattr(args, "shift_guard", False):
		params["shift_guard_enable"] = True
	if getattr(args, "shift_guard_factor", None) is not None:
		params["shift_guard_factor"] = float(args.shift_guard_factor)

	# Load model
	model = _load_model(params, device)

	info("Preparing input dataset", section="DATA")
	stations, dtimes, origins = prepare_input_dfs(params, model=model, device=device)
	info(f"Dataset loaded events={origins.shape[0]} dtimes={dtimes.shape[0]}", section="DATA")

	# Initialize wandb if enabled (after we have the actual dataset info)
	if params["use_wandb"]:
		params["total_events"] = origins.shape[0]
		params["total_dtimes"] = dtimes.shape[0]
	wandb_logger = init_wandb_if_enabled(params)
	# Expose whether W&B is actually active at runtime (import/init can fail and return None).
	# Used to avoid computing expensive diagnostics that will never be logged.
	params["_wandb_runtime_enabled"] = bool(wandb_logger is not None)

	info("Running SPIDER", section="RUN")
	locate_all(params, origins, dtimes, model, device, wandb_logger)

	# Finish wandb run
	if wandb_logger:
		wandb_logger.finish()
	return 0


def _cmd_locate_map(args: argparse.Namespace) -> int:
	"""Run Phase 1 only, then dump a Phase-2 bundle."""
	with open(args.params, "r") as f:
		params = json.load(f)
	params = validate_and_materialize_block1(params)
	params = validate_and_materialize_block2(params)
	params = validate_and_materialize_block3(params)
	params = validate_and_materialize_block4(params)
	params = validate_and_materialize_block5(params)
	params = validate_and_materialize_priors(params)

	# --- Optional distributed (torchrun) mode ---
	# This is a *single-chain* multi-GPU mode (data-parallel minibatches) for Phase 1 (MAP).
	world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
	local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
	rank = int(os.environ.get("RANK", "0") or "0")
	ddp_enabled = bool(world_size > 1)

	dev_list = list(params.get("devices", []))
	if not dev_list:
		raise ValueError("No devices configured. Set compute.devices in the config, or pass --device.")

	if ddp_enabled:
		if args.device is not None:
			raise ValueError("When running under torchrun (WORLD_SIZE>1), do not pass --device. Use inference.compute.devices to map ranks to devices.")
		if len(dev_list) < int(world_size):
			raise ValueError(
				f"Distributed `spider locate-map` requires inference.compute.devices to list >= WORLD_SIZE devices. "
				f"Got devices={len(dev_list)} WORLD_SIZE={world_size}."
			)
		device_id = _parse_device_entry(dev_list[int(local_rank)])
		params["devices"] = [device_id]
	else:
		if args.device is None:
			if len(dev_list) != 1:
				raise ValueError(
					f"Config compute.devices has {len(dev_list)} entries but `spider locate-map` is single-device. "
					"Use `torchrun -m spider locate-map ...` for single-chain multi-GPU MAP, "
					"use `spider locate-map --device ...` to pick one device, or run one map per GPU manually."
				)
			device_id = _parse_device_entry(dev_list[0])
		else:
			device_id = int(args.device)
			params["devices"] = [device_id]
	device = _device_from_id(int(device_id))
	if device.type == "cuda":
		try:
			torch.cuda.set_device(device)
		except Exception:
			pass

	# Initialize torch.distributed if requested (torchrun).
	if ddp_enabled:
		try:
			backend = "nccl" if (torch.cuda.is_available() and device.type == "cuda") else "gloo"
			dist.init_process_group(backend=backend, init_method="env://")
		except Exception as e:
			raise RuntimeError(f"Failed to init torch.distributed process group (backend={backend}): {e}")
		# Expose rank info to core code for batching + IO gating.
		params["_ddp_world_size"] = int(world_size)
		params["_ddp_rank"] = int(rank)
		params["_ddp_local_rank"] = int(local_rank)

	if getattr(args, "shift_guard", False):
		params["shift_guard_enable"] = True
	if getattr(args, "shift_guard_factor", None) is not None:
		params["shift_guard_factor"] = float(args.shift_guard_factor)

	model = _load_model(params, device)
	info("Preparing input dataset", section="DATA")
	stations, dtimes, origins = prepare_input_dfs(params, model=model, device=device)
	info(f"Dataset loaded events={origins.shape[0]} dtimes={dtimes.shape[0]}", section="DATA")

	# W&B
	if params["use_wandb"]:
		params["total_events"] = origins.shape[0]
		params["total_dtimes"] = dtimes.shape[0]
	# Only rank0 logs in torchrun mode to avoid duplicate runs.
	is_main = (int(params.get("_ddp_rank", 0)) == 0) if ddp_enabled else True
	wandb_logger = init_wandb_if_enabled(params) if is_main else None
	params["_wandb_runtime_enabled"] = bool(wandb_logger is not None)

	# Default bundle output in checkpoint_dir
	bundle_out = getattr(args, "bundle_out", None)
	if not bundle_out:
		ckpt_dir = str(params.get("checkpoint_dir", params.get("io", {}).get("checkpoint_dir", "")) or "")
		if not ckpt_dir:
			ckpt_dir = os.path.dirname(os.path.abspath(args.params)) or "."
		os.makedirs(ckpt_dir, exist_ok=True)
		bundle_out = os.path.join(ckpt_dir, "phase2_bundle.pth")

	info("Running SPIDER Phase 1 (MAP) only", section="RUN")
	try:
		locate_map(params, origins, dtimes, model, device, wandb_logger, bundle_out=bundle_out)
	finally:
		# Clean shutdown for torchrun
		if ddp_enabled:
			try:
				dist.barrier()
			except Exception:
				pass
			try:
				dist.destroy_process_group()
			except Exception:
				pass

	if wandb_logger:
		wandb_logger.finish()
	return 0


def _cmd_sample(args: argparse.Namespace) -> int:
	"""Run Phase 2–4 starting from a Phase-2 bundle (skips Phase 1)."""
	with open(args.params, "r") as f:
		params = json.load(f)
	params = validate_and_materialize_block1(params)
	params = validate_and_materialize_block2(params)
	params = validate_and_materialize_block3(params)
	params = validate_and_materialize_block4(params)
	params = validate_and_materialize_block5(params)
	params = validate_and_materialize_priors(params)

	# --- Optional distributed (torchrun) mode ---
	# This is a *single-chain* multi-GPU mode (data-parallel minibatches).
	# It must not change the semantics of `spider sample-multi` (independent chains).
	world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
	local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
	rank = int(os.environ.get("RANK", "0") or "0")
	ddp_enabled = bool(world_size > 1)

	dev_list = list(params.get("devices", []))
	if not dev_list:
		raise ValueError("No devices configured. Set compute.devices in the config, or pass --device.")

	if ddp_enabled:
		if args.device is not None:
			raise ValueError("When running under torchrun (WORLD_SIZE>1), do not pass --device. Use inference.compute.devices to map ranks to GPUs.")
		if len(dev_list) < int(world_size):
			raise ValueError(
				f"Distributed `spider sample` requires inference.compute.devices to list >= WORLD_SIZE GPUs. "
				f"Got devices={len(dev_list)} WORLD_SIZE={world_size}."
			)
		device_id = _parse_device_entry(dev_list[int(local_rank)])
		# Keep the materialized device list consistent with this rank's device.
		params["devices"] = [device_id]
	else:
		if args.device is None:
			if len(dev_list) != 1:
				raise ValueError(
					f"Config compute.devices has {len(dev_list)} entries but `spider sample` is single-device. "
					"Use `spider sample-multi` for multi-GPU, or pass --device to pick one GPU."
				)
			device_id = _parse_device_entry(dev_list[0])
		else:
			device_id = int(args.device)
			params["devices"] = [device_id]

	# Resolve to an actual torch.device (robust to CPU-only builds).
	device = _device_from_id(int(device_id))
	if device.type == "cuda":
		try:
			torch.cuda.set_device(device)
		except Exception:
			pass

	# Initialize torch.distributed if requested (torchrun).
	if ddp_enabled:
		try:
			backend = "nccl" if torch.cuda.is_available() else "gloo"
			dist.init_process_group(backend=backend, init_method="env://")
		except Exception as e:
			raise RuntimeError(f"Failed to init torch.distributed process group (backend={backend}): {e}")
		# Expose rank info to core code for batching + IO gating.
		params["_ddp_world_size"] = int(world_size)
		params["_ddp_rank"] = int(rank)
		params["_ddp_local_rank"] = int(local_rank)

	if getattr(args, "shift_guard", False):
		params["shift_guard_enable"] = True
	if getattr(args, "shift_guard_factor", None) is not None:
		params["shift_guard_factor"] = float(args.shift_guard_factor)

	bundle_path = getattr(args, "bundle", None)
	if not bundle_path:
		ckpt_dir = str(params.get("checkpoint_dir", params.get("io", {}).get("checkpoint_dir", "")) or "")
		if not ckpt_dir:
			raise ValueError("Missing --bundle and could not infer checkpoint_dir for default bundle path.")
		bundle_path = os.path.join(ckpt_dir, "phase2_bundle.pth")
	if not os.path.exists(bundle_path):
		raise FileNotFoundError(
			f"Phase-2 bundle not found: {bundle_path}. Run `python -m spider locate-map <params>` first "
			"or pass --bundle PATH."
		)

	# Load bundle only to populate W&B config counts (core will load again)
	try:
		bun = load_phase2_bundle(path=bundle_path)
		if params["use_wandb"]:
			params["total_events"] = bun.origins0.shape[0]
			params["total_dtimes"] = bun.dtimes.shape[0]
	except Exception:
		pass

	# W&B: only rank0 logs in torchrun mode to avoid duplicate runs.
	is_main = (int(params.get("_ddp_rank", 0)) == 0) if ddp_enabled else True
	wandb_logger = init_wandb_if_enabled(params) if is_main else None
	params["_wandb_runtime_enabled"] = bool(wandb_logger is not None)

	model = _load_model(params, device)
	info(f"Running SPIDER sampling from bundle={bundle_path}", section="RUN")
	try:
		locate_sample_from_bundle(params=params, bundle_path=bundle_path, model=model, device=device, wandb_logger=wandb_logger)
	finally:
		# Clean shutdown for torchrun
		if ddp_enabled:
			try:
				dist.barrier()
			except Exception:
				pass
			try:
				dist.destroy_process_group()
			except Exception:
				pass

	if wandb_logger:
		wandb_logger.finish()
	return 0


def _cmd_analyze_resid(args: argparse.Namespace) -> int:
	"""Run post-Phase1 residual diagnostics from an existing Phase-2 bundle (no MAP rerun)."""
	with open(args.params, "r") as f:
		params = json.load(f)
	params = validate_and_materialize_block1(params)
	params = validate_and_materialize_block2(params)
	params = validate_and_materialize_block3(params)
	params = validate_and_materialize_block4(params)
	params = validate_and_materialize_block5(params)
	params = validate_and_materialize_priors(params)

	# Optional CLI overrides for shared_event_latent tau estimation inside analyze-resid.
	# analyze_resid_from_bundle() already calls maybe_estimate_shared_event_latent_tau_after_phase1(state=...),
	# which reads inference.diagnostics.shared_event_latent_tau_estimate.* from params.
	try:
		inf = params.setdefault("inference", {})
		if not isinstance(inf, dict):
			inf = {}
			params["inference"] = inf
		dg = inf.setdefault("diagnostics", {})
		if not isinstance(dg, dict):
			dg = {}
			inf["diagnostics"] = dg
		tau_cfg = dg.setdefault("shared_event_latent_tau_estimate", {})
		if not isinstance(tau_cfg, dict):
			tau_cfg = {}
			dg["shared_event_latent_tau_estimate"] = tau_cfg

		# Only override keys when the CLI flag was provided.
		if getattr(args, "tau_method", None) is not None:
			tau_cfg["method"] = str(getattr(args, "tau_method")).strip().lower()
		if getattr(args, "tau_apply", None) is not None:
			tau_cfg["apply"] = bool(getattr(args, "tau_apply"))
		if getattr(args, "tau_apply_scale", None) is not None:
			tau_cfg["apply_scale"] = float(getattr(args, "tau_apply_scale"))
		if getattr(args, "tau_n_rows", None) is not None:
			tau_cfg["n_rows"] = int(getattr(args, "tau_n_rows"))
		if getattr(args, "tau_seed", None) is not None:
			tau_cfg["seed"] = int(getattr(args, "tau_seed"))
		if getattr(args, "tau_batch_size", None) is not None:
			tau_cfg["batch_size"] = int(getattr(args, "tau_batch_size"))
		if getattr(args, "tau_holdout_frac", None) is not None:
			tau_cfg["holdout_frac"] = float(getattr(args, "tau_holdout_frac"))
		if getattr(args, "tau_min_edges_per_group", None) is not None:
			tau_cfg["min_edges_per_group"] = int(getattr(args, "tau_min_edges_per_group"))
		if getattr(args, "tau_max_edges_per_group", None) is not None:
			tau_cfg["max_edges_per_group"] = int(getattr(args, "tau_max_edges_per_group"))
		if getattr(args, "tau_max_groups_per_phase", None) is not None:
			tau_cfg["max_groups_per_phase"] = int(getattr(args, "tau_max_groups_per_phase"))
		if getattr(args, "tau_grid_decades", None) is not None:
			tau_cfg["grid_decades"] = float(getattr(args, "tau_grid_decades"))
		if getattr(args, "tau_grid_size", None) is not None:
			tau_cfg["grid_size"] = int(getattr(args, "tau_grid_size"))
		if getattr(args, "tau_cg_rtol", None) is not None:
			tau_cfg["cg_rtol"] = float(getattr(args, "tau_cg_rtol"))
		if getattr(args, "tau_cg_maxiter", None) is not None:
			tau_cfg["cg_maxiter"] = int(getattr(args, "tau_cg_maxiter"))
	except Exception:
		# Best-effort only; analyze-resid should still run.
		pass

	dev_list = list(params.get("devices", []))
	if not dev_list:
		raise ValueError("No devices configured. Set compute.devices in the config, or pass --device.")
	if args.device is None:
		if len(dev_list) != 1:
			raise ValueError(
				f"Config compute.devices has {len(dev_list)} entries but `spider analyze-resid` is single-device. "
				"Pass --device to pick one GPU."
			)
		device_id = _parse_device_entry(dev_list[0])
	else:
		device_id = int(args.device)
		params["devices"] = [device_id]
	device = _device_from_id(int(device_id))

	# Load model (needed to compute residuals)
	model = _load_model(params, device)

	bundle_path = getattr(args, "bundle", None)
	if not bundle_path:
		ckpt_dir = str(params.get("checkpoint_dir", params.get("io", {}).get("checkpoint_dir", "")) or "")
		if not ckpt_dir:
			raise ValueError("Missing --bundle and could not infer checkpoint_dir for default bundle path.")
		bundle_path = os.path.join(ckpt_dir, "phase2_bundle.pth")
	if not os.path.exists(str(bundle_path)):
		raise ValueError(
			f"Phase-2 bundle not found: {bundle_path}. Run `python -m spider locate-map <params>` first "
			"or pass --bundle PATH."
		)

	info("Running analyze-resid from Phase-2 bundle (no MAP rerun)", section="RUN")
	analyze_resid_from_bundle(
		params=params,
		bundle_path=str(bundle_path),
		model=model,
		device=device,
		plot_variograms=bool(getattr(args, "plot_variograms", True)),
		plot_dir=(str(getattr(args, "plot_dir")) if getattr(args, "plot_dir", None) is not None else None),
	)
	return 0


def _cmd_sample_multi(args: argparse.Namespace) -> int:
	"""
	Launch multiple independent sampling chains (one process per GPU/device), starting from the same bundle.
	"""
	with open(args.params, "r") as f:
		base_params = json.load(f)

	# Devices: either provided explicitly, or use inference.compute.devices from params.
	devs = None
	if getattr(args, "devices", None):
		devs = [_parse_device_entry(x.strip()) for x in str(args.devices).split(",") if str(x).strip()]
	else:
		try:
			devs = [_parse_device_entry(x) for x in base_params.get("inference", {}).get("compute", {}).get("devices", [])]
		except Exception:
			devs = []
	if not devs:
		raise ValueError("sample-multi requires devices (either --devices or inference.compute.devices in params).")

	bundle_path = str(getattr(args, "bundle", "") or "").strip()
	if not bundle_path:
		# Infer default bundle path from base checkpoint_dir
		try:
			p0 = validate_and_materialize_block1(json.loads(json.dumps(base_params)))
			p0 = validate_and_materialize_block2(p0)
			p0 = validate_and_materialize_block3(p0)
			p0 = validate_and_materialize_block4(p0)
			p0 = validate_and_materialize_block5(p0)
			p0 = validate_and_materialize_priors(p0)
			ckpt_dir = str(p0.get("checkpoint_dir", p0.get("io", {}).get("checkpoint_dir", "")) or "")
			if ckpt_dir:
				bundle_path = os.path.join(ckpt_dir, "phase2_bundle.pth")
		except Exception:
			bundle_path = ""
	if not bundle_path or not os.path.exists(bundle_path):
		raise FileNotFoundError(
			f"Phase-2 bundle not found: {bundle_path!r}. Run `python -m spider locate-map <params>` first or pass --bundle PATH."
		)

	n_chains = int(args.chains) if getattr(args, "chains", None) is not None else int(len(devs))
	n_chains = max(1, n_chains)
	seed0 = int(getattr(args, "seed0", 0))

	out_dir = str(getattr(args, "out_dir", "") or os.path.dirname(os.path.abspath(args.params)) or ".")
	os.makedirs(out_dir, exist_ok=True)

	procs = []
	tmp_paths = []
	chain_sample_paths = []
	merged_out_path = None
	for ci in range(n_chains):
		dev = int(devs[ci % len(devs)])
		p = json.loads(json.dumps(base_params))  # cheap deep copy

		# Override device for this chain
		p.setdefault("inference", {})
		p["inference"].setdefault("compute", {})
		p["inference"]["compute"]["devices"] = [dev]

		# Add/override runtime seed so batching RNG differs per chain
		p.setdefault("inference", {})
		p["inference"].setdefault("runtime", {})
		p["inference"]["runtime"]["seed"] = int(seed0 + 1000003 * ci)

		# Unique sample/checkpoint outputs
		p.setdefault("io", {})
		samp = str(p["io"].get("samples_outfile", os.path.join(out_dir, "SPIDER_samples.h5")))
		root, ext = os.path.splitext(samp)
		if not ext:
			ext = ".h5"
		p["io"]["samples_outfile"] = f"{root}_chain{ci}{ext}"
		chain_sample_paths.append(p["io"]["samples_outfile"])

		ck = str(p["io"].get("checkpoint_dir", os.path.join(out_dir, "checkpoints")))
		p["io"]["checkpoint_dir"] = os.path.join(ck, f"chain{ci}")

		# Unique wandb run name if enabled
		if isinstance(p.get("wandb", None), dict):
			rn = str(p["wandb"].get("run_name", "spider"))
			p["wandb"]["run_name"] = f"{rn}_chain{ci}"

		fd, tmp_path = tempfile.mkstemp(prefix=f"spider_chain{ci}_", suffix=".json", dir=out_dir)
		os.close(fd)
		with open(tmp_path, "w") as f:
			json.dump(p, f, indent=4)
		tmp_paths.append(tmp_path)

		cmd = [sys.executable, "-m", "spider", "sample", tmp_path, "--device", str(dev), "--bundle", str(bundle_path)]
		print(f"[sample-multi] chain={ci} device={dev} seed={p['inference']['runtime']['seed']} samples={p['io']['samples_outfile']}")
		if getattr(args, "dry_run", False):
			continue
		procs.append(subprocess.Popen(cmd))

	# Wait
	if not getattr(args, "dry_run", False):
		rc = 0
		for pr in procs:
			r = pr.wait()
			rc = rc if rc != 0 else int(r)
		if rc != 0:
			return int(rc)

		# Merge chain sample stores into base io.samples_outfile so notebooks/scripts keep working.
		do_merge = bool(getattr(args, "merge", True))
		if do_merge:
			try:
				from spider.io.samples import merge_samples_hdf5
				base_out = str(base_params.get("io", {}).get("samples_outfile", ""))
				if not base_out:
					base_out = os.path.join(out_dir, "SPIDER_samples.h5")
				merged_out_path = merge_samples_hdf5(out_path=base_out, in_paths=chain_sample_paths, overwrite=True)
				print(f"[sample-multi] merged chains -> {merged_out_path}")
			except Exception as e:
				print(f"[sample-multi] WARNING: merge failed: {e}")
		return int(rc)
	return 0


def _cmd_locate(args: argparse.Namespace) -> int:
	"""Alias for `sample` (kept for convenience)."""
	warn("`spider locate` is now an alias for `spider sample` (Phase 2–4 only). Use `spider locate-map` for Phase 1.", section="CLI")
	return _cmd_sample(args)


def _cmd_locate_multi(args: argparse.Namespace) -> int:
	"""Alias for `sample-multi` (kept for convenience)."""
	warn("`spider locate-multi` is now an alias for `spider sample-multi`. Use `spider locate-map` once, then sample-multi.", section="CLI")
	return _cmd_sample_multi(args)


def _cmd_locate_multi_legacy(args: argparse.Namespace) -> int:
	"""
	Launch multiple independent SPIDER locate chains (one process per GPU/device).
	This is the recommended way to use 1-8 GPUs without rewriting the sampler for multi-device autograd.
	"""
	with open(args.params, "r") as f:
		base_params = json.load(f)

	# Devices: either provided explicitly, or use compute.devices from params.
	devs = None
	if getattr(args, "devices", None):
		devs = [_parse_device_entry(x.strip()) for x in str(args.devices).split(",") if str(x).strip()]
	else:
		try:
			devs = [_parse_device_entry(x) for x in base_params.get("inference", {}).get("compute", {}).get("devices", [])]
		except Exception:
			devs = []
	if not devs:
		raise ValueError("locate-multi requires devices (either --devices or inference.compute.devices in params).")

	n_chains = int(args.chains) if getattr(args, "chains", None) is not None else int(len(devs))
	n_chains = max(1, n_chains)
	seed0 = int(getattr(args, "seed0", 0))

	# Output base
	out_dir = str(getattr(args, "out_dir", "") or os.path.dirname(os.path.abspath(args.params)) or ".")
	os.makedirs(out_dir, exist_ok=True)

	procs = []
	tmp_paths = []
	chain_sample_paths = []
	merged_out_path = None
	for ci in range(n_chains):
		dev = int(devs[ci % len(devs)])
		p = json.loads(json.dumps(base_params))  # cheap deep copy

		# Override device for this chain
		p.setdefault("inference", {})
		p["inference"].setdefault("compute", {})
		p["inference"]["compute"]["devices"] = [dev]

		# Add/override runtime seed so batching RNG differs per chain
		p.setdefault("inference", {})
		p["inference"].setdefault("runtime", {})
		p["inference"]["runtime"]["seed"] = int(seed0 + 1000003 * ci)

		# Unique sample/checkpoint outputs
		p.setdefault("io", {})
		samp = str(p["io"].get("samples_outfile", os.path.join(out_dir, "SPIDER_samples.h5")))
		root, ext = os.path.splitext(samp)
		if not ext:
			ext = ".h5"
		p["io"]["samples_outfile"] = f"{root}_chain{ci}{ext}"
		chain_sample_paths.append(p["io"]["samples_outfile"])

		ck = str(p["io"].get("checkpoint_dir", os.path.join(out_dir, "checkpoints")))
		p["io"]["checkpoint_dir"] = os.path.join(ck, f"chain{ci}")

		# Unique wandb run name if enabled
		if isinstance(p.get("wandb", None), dict):
			rn = str(p["wandb"].get("run_name", "spider"))
			p["wandb"]["run_name"] = f"{rn}_chain{ci}"

		# Write temp params file
		fd, tmp_path = tempfile.mkstemp(prefix=f"spider_chain{ci}_", suffix=".json", dir=out_dir)
		os.close(fd)
		with open(tmp_path, "w") as f:
			json.dump(p, f, indent=4)
		tmp_paths.append(tmp_path)

		cmd = [sys.executable, "-m", "spider", "locate", tmp_path, "--device", str(dev)]
		print(f"[locate-multi] chain={ci} device={dev} seed={p['inference']['runtime']['seed']} samples={p['io']['samples_outfile']}")
		if getattr(args, "dry_run", False):
			continue
		procs.append(subprocess.Popen(cmd))

	# Wait
	if not getattr(args, "dry_run", False):
		rc = 0
		for pr in procs:
			r = pr.wait()
			rc = rc if rc != 0 else int(r)
		if rc != 0:
			return int(rc)

		# Merge chain sample stores into the original io.samples_outfile so existing notebooks/scripts keep working.
		do_merge = bool(getattr(args, "merge", True))
		if do_merge:
			try:
				from spider.io.samples import merge_samples_hdf5
				base_out = str(base_params.get("io", {}).get("samples_outfile", ""))
				if not base_out:
					base_out = os.path.join(out_dir, "SPIDER_samples.h5")
				merged_out_path = merge_samples_hdf5(out_path=base_out, in_paths=chain_sample_paths, overwrite=True)
				print(f"[locate-multi] merged chains -> {merged_out_path}")
			except Exception as e:
				print(f"[locate-multi] WARNING: merge failed: {e}")
				# Don't fail the run if sampling succeeded.

		return int(rc)
	return 0


def _project_events_to_xyz(origins: pl.DataFrame, lat0: float, lon0: float) -> np.ndarray:
	projector = Proj(proj="laea", lat_0=lat0, lon_0=lon0, datum="WGS84", units="km")
	XX, YY = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())
	ZZ = origins["depth"].to_numpy()
	return np.column_stack([np.asarray(XX, dtype=float), np.asarray(YY, dtype=float), np.asarray(ZZ, dtype=float)])


def _random_field_values_xy(
	pos_ev_xy_km: np.ndarray,
	amp_seconds: float,
	length_xy_km: float,
	n_features: int,
	seed: int | None = None,
) -> np.ndarray:
	"""
	2D stationary SE-like Gaussian random field over event XY using random Fourier features:
		f(x) = sqrt(2)*amp/sqrt(M) * sum_m cos(w_m·x + b_m),
	with w_m ~ Normal(0, I / L^2) in 2D and b_m ~ Uniform(0, 2π).
	Returns (n_events,) seconds.
	"""
	if amp_seconds <= 0.0:
		return np.zeros((pos_ev_xy_km.shape[0],), dtype=np.float32)
	rng = np.random.default_rng(seed)
	M = max(1, int(n_features))
	std = 1.0 / max(length_xy_km, 1e-6)
	W = rng.normal(loc=0.0, scale=std, size=(M, 2))  # (M,2)
	b = rng.uniform(low=0.0, high=2.0 * np.pi, size=(M,))
	X = pos_ev_xy_km.astype(np.float64)  # (N,2)
	Φ = X @ W.T  # (N,M)
	Φ += b[None, :]
	vals = np.cos(Φ).sum(axis=1)
	scale = np.sqrt(2.0) * float(amp_seconds) / np.sqrt(M)
	return (scale * vals).astype(np.float32)


def _cmd_synth(args: argparse.Namespace) -> int:
	# Load params
	with open(args.params, "r") as f:
		params = json.load(f)
	# Strict block1 config for synth (no priors required)
	params = validate_and_materialize_block1(params)
	# Block 3 needed for synth (uses batch_size_warmup for chunking)
	params = validate_and_materialize_block3(params)
	# Block 5 needed for synth (devices)
	params = validate_and_materialize_block5(params)
	device_id = int(args.device) if args.device is not None else int(params["devices"][0])
	if torch.cuda.is_available():
		device = torch.device(f"cuda:{device_id}")
	else:
		warn("CUDA is not available; running synth on CPU.", section="RUN")
		device = torch.device("cpu")

	# Read synthesis configuration.
	synth_cfg = params.get("synth", None)
	if synth_cfg is None:
		raise ValueError("Missing required config block: synth (expected an object/dict).")
	if not isinstance(synth_cfg, dict):
		raise ValueError("Expected params['synth'] to be an object/dict.")

	seed = int(synth_cfg.get("seed", 42))
	rng = np.random.default_rng(seed)
	overwrite = bool(synth_cfg.get("overwrite", True))
	apply_filters = bool(synth_cfg.get("apply_filters", True))
	drop_missing_events = bool(synth_cfg.get("drop_missing_events", False))

	# Determine truth catalog source.
	# Preferred new-style key: synth.true_catalog_infile
	# Back-compat / convenience: if synth.true_catalog_outfile is provided AND already exists,
	# treat it as the truth infile (do NOT overwrite it unless you set a distinct outfile).
	true_in = synth_cfg.get("true_catalog_infile", None)
	true_out = synth_cfg.get("true_catalog_outfile", None)
	if true_in is None and true_out is not None and os.path.exists(str(true_out)):
		true_in = str(true_out)
		info(f"Using existing synth.true_catalog_outfile as truth input: {true_in}", section="SYNTH")
		# Don't clobber the input truth by default.
		# If the user really wants to write a copy of the filtered truth, they should set
		# synth.true_catalog_outfile to a *different* path.

	# If a truth infile is provided, temporarily override catalog_infile for data prep.
	# This makes all event/dtime filters and pair_station_ratio computations consistent
	# with the truth locations used to generate synthetic observations.
	_orig_catalog_infile = params.get("catalog_infile")
	if true_in is not None:
		params["catalog_infile"] = str(true_in)

	# Load dataframes using the chosen truth catalog.
	# If apply_filters=True, we reuse the full pipeline data prep (this will subset/filter).
	# If apply_filters=False (default), we do minimal prep and stream dtimes in batches
	# without subsetting/filtering by event/dtime heuristics.
	if apply_filters:
		stations, dtimes, origins = prepare_input_dfs(params, model=model, device=device)
	else:
		# Minimal, no-filter truth catalog load
		origins = pl.read_csv(params["catalog_infile"]).with_columns(
			pl.col("time").cast(pl.Utf8).str.strptime(pl.Datetime("ns"), strict=False).alias("time")
		)
		if origins["time"].null_count() > 0:
			raise ValueError("truth catalog 'time' could not be parsed as datetime for some rows")
		# Stations (only needed for projecting receiver coordinates)
		stations = pl.read_csv(params["station_file"]).unique(subset=["network", "station"])
		if "depth" in stations.columns:
			stations = stations.with_columns(pl.col("depth").fill_null(0.0).fill_nan(0.0))
		# We will stream dtimes below; keep a placeholder for type checkers.
		dtimes = None

	# Restore original catalog_infile in params (avoid side-effects)
	if _orig_catalog_infile is not None:
		params["catalog_infile"] = _orig_catalog_infile

	# Optional: write a synthetic *initial* catalog by perturbing the truth catalog with event-prior noise.
	# This is useful for end-to-end synthetic experiments:
	#   truth catalog -> (perturbed init catalog) + (synthetic dtimes from truth)
	# Then run SPIDER using init catalog and dtimes synthetic.
	init_out = synth_cfg.get("init_catalog_outfile", None)
	if init_out is not None or true_out is not None:
		# Determine event prior std (km,km,km,sec)
		std = None
		# Prefer nested priors.event.params.std if present
		try:
			std = params.get("model", {}).get("priors", {}).get("event", {}).get("params", {}).get("std", None)
		except Exception:
			std = None
		# Allow override specifically for synth
		std_override = synth_cfg.get("event_std", None)
		if std_override is not None:
			std = std_override
		if std is None:
			raise ValueError(
				"Requested synth.init_catalog_outfile and/or synth.true_catalog_outfile but could not determine event prior std. "
				"Provide model.priors.event.params.std (nested) or synth.event_std."
			)
		std = [float(x) for x in std]
		if len(std) != 4:
			raise ValueError(f"Event prior std must be length-4 [σx,σy,σz,σt]; got len={len(std)}")

		spatial_only = bool(synth_cfg.get("init_spatial_only", False))
		clip_domain = bool(synth_cfg.get("init_clip_domain", True))
		res = synth_initial_catalog_from_truth(
			origins,
			lat0=float(params["lat_min"]),
			lon0=float(params["lon_min"]),
			event_prior_std=std,
			seed=seed,
			spatial_only=spatial_only,
			clip_domain=clip_domain,
			z_min=float(params.get("z_min")) if params.get("z_min") is not None else None,
			z_max=float(params.get("z_max")) if params.get("z_max") is not None else None,
		)
		def _safe_write_csv(df: pl.DataFrame, path: str, label: str) -> None:
			if (not overwrite) and os.path.exists(path):
				raise FileExistsError(
					f"Refusing to overwrite existing {label} file: {path}. "
					"Set synth.overwrite=true to overwrite."
				)
			df.write_csv(path)

		# Only write truth_out if it's not the truth_in we just read from.
		if true_out is not None and (true_in is None or str(true_out) != str(true_in)):
			_safe_write_csv(res.truth, str(true_out), "true_catalog_outfile")
			print(f"Wrote truth catalog to {true_out} (N={res.truth.shape[0]})")
		if init_out is not None:
			_safe_write_csv(res.init, str(init_out), "init_catalog_outfile")
			print(f"Wrote synthetic initial catalog to {init_out} (N={res.init.shape[0]})")

	# Build mapping evid -> row index (normalize IDs as strings)
	evid_to_row = {str(row["evid"]): idx for idx, row in enumerate(origins.iter_rows(named=True))}

	# Build event positions (km) and X_src torch tensor
	ev_xyz = _project_events_to_xyz(origins, lat0=params["lat_min"], lon0=params["lon_min"])  # (Ne,3)
	Ne = ev_xyz.shape[0]
	ev_xy = ev_xyz[:, :2]
	X_src_np = np.zeros((Ne, 4), dtype=np.float32)
	X_src_np[:, :3] = ev_xyz.astype(np.float32)
	# X_src[:,3] (origin time offset) left at 0.0
	X_src = torch.tensor(X_src_np, dtype=torch.float32, device=device)
	dX_src = torch.zeros_like(X_src, device=device)

	# Determine dtimes input and size.
	# If apply_filters=True, dtimes is already a materialized DataFrame.
	# If apply_filters=False, stream dtimes from CSV to avoid loading 100M+ rows into RAM.
	if apply_filters:
		N = dtimes.shape[0]
	else:
		N = None

	# Load model on device
	model = _load_model(params, device)

	# Compute and write synthetic dtimes in chunks to avoid OOM.
	bs = int(params.get("batch_size_warmup", 10000))
	bs = max(1, bs)

	# Prepare per-station-phase random field cache (optional)
	amp = float(synth_cfg.get("rf_amp", 0.02))
	Lxy = float(synth_cfg.get("rf_len_xy", 10.0))
	M = int(synth_cfg.get("rf_features", 256))
	rf_cache: dict[tuple[str, str, int], np.ndarray] = {}

	# Observation noise model
	σp, σs = params.get("phase_unc", [0.05, 0.09])
	lik = str(params.get("likelihood", "huber")).strip().lower()

	out_path = synth_cfg.get("outfile", None)
	if out_path is None:
		out_path = str(params["dtime_file"]).rsplit(".", 1)[0] + "_synthetic.csv"
	out_path = str(out_path)
	if (not overwrite) and os.path.exists(out_path):
		raise FileExistsError(
			f"Refusing to overwrite existing synth outfile: {out_path}. "
			"Set synth.overwrite=true to overwrite."
		)

	def _normalize_phase(df: pl.DataFrame) -> pl.DataFrame:
		if "phase" not in df.columns:
			raise ValueError("Input dtimes must contain a 'phase' column")
		if df.schema.get("phase") in (pl.Utf8, pl.Categorical, pl.Enum):
			return df.with_columns(
				(pl.col("phase").cast(pl.Utf8).str.to_uppercase() == "S").cast(pl.Int8).alias("phase")
			)
		return df.with_columns(pl.col("phase").cast(pl.Int8).alias("phase"))

	def _map_event_indices(evid_series: pl.Series) -> np.ndarray:
		# Convert to python objects and map via dict; chunked to control memory
		out = np.empty((len(evid_series),), dtype=np.int64)
		for i, ev in enumerate(evid_series.to_list()):
			k = str(ev)
			idx = evid_to_row.get(k, None)
			if idx is None:
				if drop_missing_events:
					out[i] = -1
				else:
					raise KeyError(f"Event id {k} in dtimes not found in truth catalog")
			else:
				out[i] = int(idx)
		return out

	def _noise_for_phase(phase_int8: np.ndarray) -> np.ndarray:
		sigma_obs = np.where(phase_int8.astype(np.float32) < 0.5, float(σp), float(σs)).astype(np.float32)
		if lik in {"gaussian", "mse", "l2"}:
			return rng.normal(loc=0.0, scale=sigma_obs).astype(np.float32)
		if lik in {"laplace", "l1", "mae"}:
			return rng.laplace(loc=0.0, scale=sigma_obs).astype(np.float32)
		# Default: Gaussian for huber/unknown
		return rng.normal(loc=0.0, scale=sigma_obs).astype(np.float32)

	with torch.no_grad():
		if apply_filters:
			dt_iter = [(dtimes,)]  # single "batch"
		else:
			# Stream raw dtimes from CSV without inference-time filtering/subsetting.
			# We still need receiver coords, so we join stations per-batch.
			dt_iter = pl.read_csv_batched(
				params["dtime_file"],
				batch_size=bs,
				columns=["network", "station", "dt", "evid1", "evid2", "phase", "cc"],
			)

		first_write = True
		total_written = 0

		if apply_filters:
			# Ensure required columns exist
			if dtimes is None:
				raise RuntimeError("Internal error: dtimes is None in apply_filters=True path")
			df0 = dtimes
			df0 = _normalize_phase(df0)
			# Pre-pull columns as numpy
			X_col = df0["X"].to_numpy()
			Y_col = df0["Y"].to_numpy()
			Z_col = df0["Z"].to_numpy()
			PH_col = df0["phase"].to_numpy().astype(np.float32)
			e1_idx = _map_event_indices(df0["evid1"])
			e2_idx = _map_event_indices(df0["evid2"])

			N = df0.shape[0]
			dt_pred = np.empty((N,), dtype=np.float32)
			for j in range(0, N, bs):
				i0 = j
				i1 = min(j + bs, N)
				II_chunk = torch.tensor(
					np.column_stack([e1_idx[i0:i1], e2_idx[i0:i1]]),
					dtype=torch.int64, device=device
				)
				Y_chunk = torch.zeros((i1 - i0, 5), dtype=torch.float32, device=device)
				Y_chunk[:, 1:] = torch.tensor(
					np.column_stack([X_col[i0:i1], Y_col[i0:i1], Z_col[i0:i1], PH_col[i0:i1]]).astype(np.float32),
					device=device, dtype=torch.float32
				)
				dt_pred[i0:i1] = compute_travel_times(II_chunk, Y_chunk, X_src, dX_src, model).detach().cpu().numpy().astype(np.float32)

			dt_syn = dt_pred.astype(np.float32).copy()
			if amp > 0.0:
				net_arr = df0["network"].to_numpy()
				sta_arr = df0["station"].to_numpy()
				ph_arr = df0["phase"].to_numpy()
				combos = df0.select(["network", "station", "phase"]).unique(maintain_order=True)
				for row in combos.iter_rows(named=True):
					net = row["network"]; sta = row["station"]; ph = int(row["phase"])
					mask = (net_arr == net) & (sta_arr == sta) & (ph_arr == ph)
					if not np.any(mask):
						continue
					key = (str(net), str(sta), int(ph))
					f_ev = rf_cache.get(key)
					if f_ev is None:
						seed_sp = int(rng.integers(0, 2**31 - 1))
						f_ev = _random_field_values_xy(ev_xy, amp_seconds=amp, length_xy_km=Lxy, n_features=M, seed=seed_sp)
						rf_cache[key] = f_ev
					dt_syn[mask] += (f_ev[e2_idx[mask]] - f_ev[e1_idx[mask]]).astype(np.float32)

			dt_syn = dt_syn + _noise_for_phase(df0["phase"].to_numpy().astype(np.int8))

			# Match locate's input convention: if locate will flip dt sign on read (flip_dt_sign=true),
			# write dt with the opposite sign so that after the flip it equals the model-consistent dt.
			if bool(params.get("flip_dt_sign", False)):
				dt_syn = -dt_syn

			# Write output
			out_df = df0.select(["network", "station", "evid1", "evid2", "phase", "cc"]).with_columns(pl.Series("dt", dt_syn))
			with open(out_path, "w") as f:
				out_df.select(["dt", "network", "station", "evid1", "evid2", "phase", "cc"]).write_csv(f, include_header=True)
			print(f"Wrote synthetic differential times to {out_path} (N={out_df.shape[0]})")
			return 0

		# apply_filters=False streaming path
		reader = dt_iter
		while True:
			batches = reader.next_batches(1)
			if not batches:
				break
			df = batches[0]
			if df.is_empty():
				continue
			df = _normalize_phase(df)
			# Join receiver station coords (inner join is required to get X,Y,Z)
			df = prepare_stations(stations, df, params["lat_min"], params["lon_min"])
			if df.is_empty():
				continue

			# Map event ids to indices (and optionally drop missing)
			e1_idx = _map_event_indices(df["evid1"])
			e2_idx = _map_event_indices(df["evid2"])
			if drop_missing_events:
				mask_ok = (e1_idx >= 0) & (e2_idx >= 0)
				if not np.all(mask_ok):
					df = df.filter(pl.Series(mask_ok))
					e1_idx = e1_idx[mask_ok]
					e2_idx = e2_idx[mask_ok]
					if df.is_empty():
						continue

			# Build tensors for this batch and compute predicted dt
			II_chunk = torch.tensor(np.column_stack([e1_idx, e2_idx]), dtype=torch.int64, device=device)
			Y_chunk = torch.zeros((df.shape[0], 5), dtype=torch.float32, device=device)
			Y_chunk[:, 1:] = torch.tensor(
				np.column_stack([
					df["X"].to_numpy(),
					df["Y"].to_numpy(),
					df["Z"].to_numpy(),
					df["phase"].to_numpy().astype(np.float32),
				]).astype(np.float32),
				device=device, dtype=torch.float32,
			)
			dt_pred = compute_travel_times(II_chunk, Y_chunk, X_src, dX_src, model).detach().cpu().numpy().astype(np.float32)
			dt_syn = dt_pred.copy()

			# Optional correlated path term per station-phase
			if amp > 0.0:
				net_arr = df["network"].to_numpy()
				sta_arr = df["station"].to_numpy()
				ph_arr = df["phase"].to_numpy()
				for net, sta, ph in df.select(["network", "station", "phase"]).unique(maintain_order=False).iter_rows():
					mask = (net_arr == net) & (sta_arr == sta) & (ph_arr == ph)
					if not np.any(mask):
						continue
					key = (str(net), str(sta), int(ph))
					f_ev = rf_cache.get(key)
					if f_ev is None:
						seed_sp = int(rng.integers(0, 2**31 - 1))
						f_ev = _random_field_values_xy(ev_xy, amp_seconds=amp, length_xy_km=Lxy, n_features=M, seed=seed_sp)
						rf_cache[key] = f_ev
					dt_syn[mask] += (f_ev[e2_idx[mask]] - f_ev[e1_idx[mask]]).astype(np.float32)

			# Per-observation noise
			dt_syn = dt_syn + _noise_for_phase(df["phase"].to_numpy().astype(np.int8))

			# Match locate's input convention: if locate will flip dt sign on read (flip_dt_sign=true),
			# write dt with the opposite sign so that after the flip it equals the model-consistent dt.
			if bool(params.get("flip_dt_sign", False)):
				dt_syn = -dt_syn

			out_df = df.select(["network", "station", "evid1", "evid2", "phase", "cc"]).with_columns(pl.Series("dt", dt_syn))
			mode = "w" if first_write else "a"
			with open(out_path, mode) as f:
				out_df.select(["dt", "network", "station", "evid1", "evid2", "phase", "cc"]).write_csv(f, include_header=first_write)
			first_write = False
			total_written += out_df.shape[0]
			if total_written and (total_written % (10 * bs) == 0):
				info(f"synth wrote {total_written} dtimes rows so far", section="SYNTH")

	print(f"Wrote synthetic differential times to {out_path} (N={total_written})")
	return 0


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog=prog or "spider", description="SPIDER command-line interface")
	subparsers = parser.add_subparsers(dest="command", metavar="command")

	# locate-map subcommand
	p_map = subparsers.add_parser("locate-map", help="Run Phase 1 (MAP) only and dump a Phase-2 bundle")
	p_map.add_argument("params", help="Path to parameter JSON file")
	p_map.add_argument("--device", type=int, default=None, help="CUDA device id to use (single-device)")
	p_map.add_argument("--bundle-out", type=str, default=None, help="Output path for Phase-2 bundle (default: <checkpoint_dir>/phase2_bundle.pth)")
	p_map.add_argument("--shift-guard", action="store_true", help="Abort if any event moves > factor × prior stds; prints offending observations")
	p_map.add_argument("--shift-guard-factor", type=float, default=None, help="Factor relative to prior_event_std; default 5.0")
	p_map.set_defaults(func=_cmd_locate_map)

	# sample subcommand
	p_samp = subparsers.add_parser("sample", help="Run Phase 2–4 (sampling) starting from a Phase-2 bundle")
	p_samp.add_argument("params", help="Path to parameter JSON file")
	p_samp.add_argument("--device", type=int, default=None, help="CUDA device id to use (sample is single-device)")
	p_samp.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle (default: <checkpoint_dir>/phase2_bundle.pth)")
	p_samp.add_argument("--shift-guard", action="store_true", help="Abort if any event moves > factor × prior stds; prints offending observations")
	p_samp.add_argument("--shift-guard-factor", type=float, default=None, help="Factor relative to prior_event_std; default 5.0")
	p_samp.set_defaults(func=_cmd_sample)

	# sample-multi subcommand
	p_sm = subparsers.add_parser("sample-multi", help="Launch multiple independent sampling chains from the same Phase-2 bundle")
	p_sm.add_argument("params", help="Path to parameter JSON file")
	p_sm.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle (default: <checkpoint_dir>/phase2_bundle.pth)")
	p_sm.add_argument("--chains", type=int, default=None, help="Number of chains to run (default: number of devices)")
	p_sm.add_argument("--devices", type=str, default=None, help="Comma-separated CUDA device ids (default: inference.compute.devices from params)")
	p_sm.add_argument("--seed0", type=int, default=0, help="Base seed for chain RNG offsets (default: 0)")
	p_sm.add_argument("--out-dir", type=str, default=None, help="Directory to write chain param files (default: params directory)")
	p_sm.add_argument("--dry-run", action="store_true", help="Print what would run, but do not start processes")
	p_sm.add_argument("--no-merge", dest="merge", action="store_false", help="Do not merge chain HDF5 outputs into base io.samples_outfile")
	p_sm.set_defaults(func=_cmd_sample_multi)

	# analyze-resid subcommand
	p_ar = subparsers.add_parser("analyze-resid", help="Run residual diagnostics from an existing Phase-2 bundle (no MAP rerun)")
	p_ar.add_argument("params", help="Path to parameters JSON file")
	p_ar.add_argument("--device", type=int, default=None, help="CUDA device id to use (single-device)")
	p_ar.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle (.pth). Default: checkpoint_dir/phase2_bundle.pth")
	g_plot = p_ar.add_mutually_exclusive_group()
	g_plot.add_argument("--plot-variograms", dest="plot_variograms", action="store_true", help="Write variogram PNG plots (default)")
	g_plot.add_argument("--no-plot-variograms", dest="plot_variograms", action="store_false", help="Disable variogram plotting")
	p_ar.set_defaults(plot_variograms=True)
	p_ar.add_argument("--plot-dir", type=str, default=None, help="Directory for variogram PNGs (default: bundle directory)")

	# Optional: shared_event_latent tau estimation overrides (post-Phase1 diagnostics)
	p_tau = p_ar.add_argument_group("shared_event_latent tau estimation (diagnostics)")
	p_tau.add_argument(
		"--tau-method",
		type=str,
		default=None,
		choices=["moment", "cv"],
		help="Tau estimator for shared_event_latent: 'moment' (fast heuristic) or 'cv' (held-out predictive, slower).",
	)
	g_apply = p_tau.add_mutually_exclusive_group()
	g_apply.set_defaults(tau_apply=None)
	g_apply.add_argument("--tau-apply", dest="tau_apply", action="store_true", help="Apply estimated tau into params for this analyze-resid run.")
	g_apply.add_argument("--no-tau-apply", dest="tau_apply", action="store_false", help="Do not apply estimated tau (just print).")
	p_tau.add_argument("--tau-apply-scale", type=float, default=None, help="Scale factor applied to estimated tau before applying (e.g. 0.5 to be more conservative).")
	p_tau.add_argument("--tau-n-rows", type=int, default=None, help="Rows to subsample for tau estimation (default depends on method).")
	p_tau.add_argument("--tau-seed", type=int, default=None, help="RNG seed for tau estimation subsampling/splits.")
	p_tau.add_argument("--tau-batch-size", type=int, default=None, help="Batch size for residual evaluation during tau estimation.")
	# CV-only knobs (ignored for moment)
	p_tau.add_argument("--tau-holdout-frac", type=float, default=None, help="Holdout fraction per station-phase group for CV tau.")
	p_tau.add_argument("--tau-min-edges-per-group", type=int, default=None, help="Minimum edges per station-phase group to include in CV.")
	p_tau.add_argument("--tau-max-edges-per-group", type=int, default=None, help="Max edges per station-phase group (cap for compute).")
	p_tau.add_argument("--tau-max-groups-per-phase", type=int, default=None, help="Max station groups per phase to include in CV.")
	p_tau.add_argument("--tau-grid-decades", type=float, default=None, help="Log10 half-width for tau grid around center (e.g. 1.0 -> ×[0.1,10]).")
	p_tau.add_argument("--tau-grid-size", type=int, default=None, help="Number of tau candidates in the grid (odd recommended).")
	p_tau.add_argument("--tau-cg-rtol", type=float, default=None, help="CG relative tolerance for CV ridge solves.")
	p_tau.add_argument("--tau-cg-maxiter", type=int, default=None, help="CG max iterations for CV ridge solves.")

	p_ar.set_defaults(func=_cmd_analyze_resid)

	# locate subcommand
	p_locate = subparsers.add_parser("locate", help="Alias for `sample` (Phase 2–4). Use `locate-map` for Phase 1.")
	p_locate.add_argument("params", help="Path to parameter JSON file")
	p_locate.add_argument(
		"--device",
		type=int,
		default=None,
		help="CUDA device id to use (locate is single-device; required if compute.devices has >1)",
	)
	p_locate.add_argument("--shift-guard", action="store_true", help="Abort if any event moves > factor × prior stds; prints offending observations")
	p_locate.add_argument("--shift-guard-factor", type=float, default=None, help="Factor relative to prior_event_std; default 5.0")
	p_locate.set_defaults(func=_cmd_locate)

	# locate-multi subcommand
	p_multi = subparsers.add_parser("locate-multi", help="Alias for `sample-multi` (starts from Phase-2 bundle)")
	p_multi.add_argument("params", help="Path to parameter JSON file")
	p_multi.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle (default: <checkpoint_dir>/phase2_bundle.pth)")
	p_multi.add_argument("--chains", type=int, default=None, help="Number of chains to run (default: number of devices)")
	p_multi.add_argument("--devices", type=str, default=None, help="Comma-separated CUDA device ids (default: compute.devices from params)")
	p_multi.add_argument("--seed0", type=int, default=0, help="Base seed for chain RNG offsets (default: 0)")
	p_multi.add_argument("--out-dir", type=str, default=None, help="Directory to write chain param files (default: params directory)")
	p_multi.add_argument("--dry-run", action="store_true", help="Print what would run, but do not start processes")
	p_multi.add_argument("--no-merge", dest="merge", action="store_false", help="Do not merge chain HDF5 outputs into base io.samples_outfile")
	p_multi.set_defaults(func=_cmd_locate_multi)

	# locate-full subcommand (kept for one-shot legacy flows)
	p_full = subparsers.add_parser("locate-full", help="Run full pipeline (Phase 1 + Phase 2–4) [legacy]")
	p_full.add_argument("params", help="Path to parameter JSON file")
	p_full.add_argument("--device", type=int, default=None, help="CUDA device id to use (single-device)")
	p_full.add_argument("--shift-guard", action="store_true", help="Abort if any event moves > factor × prior stds; prints offending observations")
	p_full.add_argument("--shift-guard-factor", type=float, default=None, help="Factor relative to prior_event_std; default 5.0")
	p_full.set_defaults(func=_cmd_locate_full)

	# synth subcommand
	p_synth = subparsers.add_parser("synth", help="Generate a synthetic differential time dataset matching the current configuration")
	p_synth.add_argument("params", help="Path to parameter JSON file")
	p_synth.add_argument("--device", type=int, default=None, help="CUDA device id to use (default: compute.devices[0])")
	p_synth.set_defaults(func=_cmd_synth)

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


