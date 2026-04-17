import argparse

import json
import os
import sys
import subprocess
import tempfile
from typing import Optional

import torch
import numpy as np
import polars as pl
from pyproj import Proj

from spider.core import prepare_input_dfs
from spider.core.locate import locate_all, locate_map, locate_sample_from_bundle
from spider.core.eikonet_loader import load_eikonet_model
from spider.core.modeling import compute_travel_times
from spider.io.synth import synth_initial_catalog_from_truth
from spider.io.phase_bundle import load_phase2_bundle
from spider.utils import init_wandb_if_enabled
from spider.utils.console import info, warn

from spider.core.data import prepare_stations
from spider.core.analyze_resid import analyze_resid_from_bundle
from spider.core.config_v2 import ConfigError as ConfigV2Error
from spider.core.config_v2 import load_config as load_config_v2
from spider.core.config_v2 import load_config_file as load_config_file_v2
from spider.core.config_v2 import to_legacy_runtime_params


# Standardized stdout helper
def _log(*parts, section: str = "CLI", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)


def _load_params_json(path: str) -> dict:
	"""Load a JSON params file from disk."""
	with open(path, "r") as f:
		return json.load(f)


def _validate_params_all(params: dict, *, require_priors: bool = True, mode: Optional[str] = None) -> dict:
	"""
	Validate/materialize params for runtime commands.

	Input is strict config_v2 canonical JSON shape.
	Internally we bridge via config_v2.legacy_bridge while runtime modules
	are still being migrated away from legacy materialized keys.
	"""
	resolved_v2 = load_config_v2(params, mode=mode)
	return to_legacy_runtime_params(
		resolved_v2,
		profile="all",
		require_priors=bool(require_priors),
	)


def _validate_params_synth(params: dict) -> dict:
	"""
	Validation/materialization for `spider synth`.

	Synth intentionally does not require priors and does not require all inference blocks.
	"""
	resolved_v2 = load_config_v2(params, mode="synth")
	return to_legacy_runtime_params(resolved_v2, profile="synth", require_priors=False)


def _cmd_validate_config(args: argparse.Namespace) -> int:
	"""
	Validate a config against the canonical config_v2 schema.

	This command intentionally validates only config_v2 (no legacy compatibility layer).
	"""
	try:
		resolved = load_config_file_v2(args.params, mode=args.mode)
	except ConfigV2Error as e:
		warn(str(e), section="CFGv2")
		return 1
	except Exception as e:
		warn(f"Unexpected config load error: {e}", section="CFGv2")
		return 1

	info(f"Config v2 validation OK: {args.params}", section="CFGv2")
	if resolved.defaults_applied:
		for d in resolved.defaults_applied:
			info(f"default applied: {d}", section="CFGv2")
	else:
		info("No defaults applied.", section="CFGv2")

	if getattr(args, "print_resolved", False):
		print(json.dumps(resolved.runtime, indent=2))

	return 0


def _apply_torch_runtime_settings(params: dict) -> None:
	"""
	Apply optional torch runtime performance settings.

	This is particularly useful for `sample-multi` where each process should configure
	its own CUDA backend deterministically at startup.

	Config (optional; extra keys under inference.runtime are allowed):
	  inference:
	    runtime:
	      torch:
	        allow_tf32: bool
	        matmul_precision: "highest"|"high"|"medium"
	        compile_eikonet: bool
	        compile_mode: optional str (e.g. "default", "reduce-overhead", "max-autotune")
	        compile_backend: optional str (default torch backend)
	        compile_dynamic: optional bool
	        compile_fullgraph: optional bool
	"""
	try:
		inf = params.get("inference", None)
		rt = inf.get("runtime", None) if isinstance(inf, dict) else None
		tc = rt.get("torch", None) if isinstance(rt, dict) else None
		if not isinstance(tc, dict):
			return
		allow_tf32 = tc.get("allow_tf32", None)
		if allow_tf32 is not None:
			v = bool(allow_tf32)
			try:
				torch.backends.cuda.matmul.allow_tf32 = v  # type: ignore[attr-defined]
			except Exception:
				pass
			try:
				torch.backends.cudnn.allow_tf32 = v  # type: ignore[attr-defined]
			except Exception:
				pass
		mp = tc.get("matmul_precision", None)
		if mp is not None:
			try:
				torch.set_float32_matmul_precision(str(mp))
			except Exception:
				pass
	except Exception:
		return


def _maybe_compile_eikonet_model(params: dict, model: torch.nn.Module) -> torch.nn.Module:
	"""
	Optionally wrap EikoNet with torch.compile based on inference.runtime.torch config.
	"""
	try:
		inf = params.get("inference", None)
		rt = inf.get("runtime", None) if isinstance(inf, dict) else None
		tc = rt.get("torch", None) if isinstance(rt, dict) else None
		if not isinstance(tc, dict):
			return model
		enabled = bool(tc.get("compile_eikonet", False))
		if not enabled:
			return model
	except Exception:
		return model

	if not hasattr(torch, "compile"):
		warn("torch.compile requested but not available in this PyTorch build; continuing without compile.", section="RUN")
		params["_torch_compile_eikonet_enabled"] = False
		return model

	kwargs = {}
	try:
		if tc.get("compile_mode", None) is not None:
			kwargs["mode"] = str(tc.get("compile_mode"))
		if tc.get("compile_backend", None) is not None:
			kwargs["backend"] = str(tc.get("compile_backend"))
		if tc.get("compile_dynamic", None) is not None:
			kwargs["dynamic"] = bool(tc.get("compile_dynamic"))
		if tc.get("compile_fullgraph", None) is not None:
			kwargs["fullgraph"] = bool(tc.get("compile_fullgraph"))
	except Exception:
		pass

	try:
		model = torch.compile(model, **kwargs)  # type: ignore[attr-defined]
		params["_torch_compile_eikonet_enabled"] = True
		info(
			"Enabled torch.compile for EikoNet"
			+ (f" with options={kwargs}" if kwargs else ""),
			section="RUN",
		)
	except Exception as e:
		params["_torch_compile_eikonet_enabled"] = False
		warn(
			f"torch.compile requested for EikoNet but failed ({e}); continuing without compile.",
			section="RUN",
		)
	return model


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
	model = load_eikonet_model(params=params, device=device)
	model = _maybe_compile_eikonet_model(params=params, model=model)
	return model


def _remap_cuda_device_ids_for_visible_devices(device_ids: list[int]) -> list[int]:
	"""
	Remap *physical* CUDA ids to process-local 0..(n_visible-1) ids when CUDA_VISIBLE_DEVICES is set.

	This is common on clusters where a job is launched with CUDA_VISIBLE_DEVICES="4,5"
	but configs still list devices=[4,5]. In that case, inside the process, the correct ids are [0,1].
	"""
	try:
		ids = [int(x) for x in device_ids]
	except Exception:
		return device_ids
	cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
	if not cvd:
		return ids
	try:
		vis = [int(x.strip()) for x in cvd.split(",") if x.strip() != ""]
	except Exception:
		return ids
	if not vis:
		return ids
	# If the ids already look like local indices, don't remap.
	if all((0 <= int(d) < int(len(vis))) for d in ids):
		return ids
	mp = {int(pid): int(i) for i, pid in enumerate(vis)}
	if all(int(d) in mp for d in ids):
		return [mp[int(d)] for d in ids]
	return ids


def _maybe_enable_dataparallel(dev_list: list, *, args_device: int | None) -> tuple[bool, list[int]]:
	"""
	Return (use_dp, dp_device_ids) for torch.nn.DataParallel.

	We only enable DP when:
	- args_device is None (user didn't force single-device)
	- dev_list has >1 entries
	- all parsed device ids are CUDA (>=0)
	- torch.cuda.is_available()
	We also remap ids when CUDA_VISIBLE_DEVICES is set.
	"""
	if args_device is not None:
		return False, []
	if len(dev_list) <= 1:
		return False, []
	try:
		parsed = [_parse_device_entry(d) for d in dev_list]
	except Exception:
		return False, []
	if any(int(d) < 0 for d in parsed):
		warn(
			f"Not enabling DataParallel because inference.compute.devices contains a CPU entry: {dev_list}. "
			"Use all-CUDA ids to enable multi-GPU DataParallel, or pass --device for single-device.",
			section="RUN",
		)
		return False, []
	if not torch.cuda.is_available():
		warn("Not enabling DataParallel because CUDA is not available.", section="RUN")
		return False, []
	parsed = _remap_cuda_device_ids_for_visible_devices([int(d) for d in parsed])
	return True, parsed


def _fail_if_torchrun_env(cmd: str) -> None:
	"""
	Fail fast if the user is running under torchrun/DDP env vars.

	After switching to single-process DataParallel, launching via torchrun would start multiple
	processes, each attempting to use multiple GPUs, which is almost always wrong and can look
	like a deadlock/hang.
	"""
	try:
		ws = int(os.environ.get("WORLD_SIZE", "1") or "1")
	except Exception:
		ws = 1
	if int(ws) > 1:
		raise ValueError(
			f"`{cmd}` no longer supports torchrun/DDP (WORLD_SIZE={ws}). "
			"Run it as a normal single process: `python -m spider "
			+ str(cmd).strip()
			+ " ...` and pass explicit CLI device flags (--device / --devices)."
		)


def _cmd_locate_full(args: argparse.Namespace) -> int:
	"""Legacy: run the full pipeline (Phase 1 + Phase 2–4)."""
	_fail_if_torchrun_env("locate-full")
	params = _validate_params_all(_load_params_json(args.params), require_priors=True, mode="locate-full")
	_apply_torch_runtime_settings(params)

	if args.device is None:
		raise ValueError("`spider locate-full` requires --device <CUDA_ID> (CLI-only device selection).")
	device_id = int(args.device)
	params["devices"] = [device_id]

	# Resolve to an actual torch.device
	device = _device_from_id(int(device_id))
	if device.type == "cuda":
		try:
			torch.cuda.set_device(device)
		except Exception:
			pass

	params["_ddp_world_size"] = 1
	params["_ddp_rank"] = 0
	params["_ddp_local_rank"] = 0
	params["_use_dataparallel"] = False

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

	# Initialize wandb if enabled
	if params["use_wandb"]:
		params["total_events"] = origins.shape[0]
		params["total_dtimes"] = dtimes.shape[0]
	wandb_logger = init_wandb_if_enabled(params)
	params["_wandb_runtime_enabled"] = bool(wandb_logger is not None)

	info("Running SPIDER", section="RUN")
	locate_all(params, origins, dtimes, model, device, wandb_logger)

	# Finish wandb run
	if wandb_logger:
		wandb_logger.finish()
	return 0


def _cmd_locate_map(args: argparse.Namespace) -> int:
	"""Run Phase 1 only, then dump a Phase-2 bundle."""
	_fail_if_torchrun_env("locate-map")
	params = _validate_params_all(_load_params_json(args.params), require_priors=True, mode="locate-map")
	_apply_torch_runtime_settings(params)

	if args.device is None:
		raise ValueError("`spider locate-map` requires --device <CUDA_ID> (CLI-only device selection).")
	device_id = int(args.device)
	params["devices"] = [device_id]

	device = _device_from_id(int(device_id))
	if device.type == "cuda":
		try:
			torch.cuda.set_device(device)
		except Exception:
			pass

	params["_ddp_world_size"] = 1
	params["_ddp_rank"] = 0
	params["_ddp_local_rank"] = 0
	params["_use_dataparallel"] = False

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
	wandb_logger = init_wandb_if_enabled(params)
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
	locate_map(params, origins, dtimes, model, device, wandb_logger, bundle_out=bundle_out)

	if wandb_logger:
		wandb_logger.finish()
	return 0


def _cmd_sample(args: argparse.Namespace) -> int:
	"""Run Phase 2–4 starting from a Phase-2 bundle (skips Phase 1)."""
	_fail_if_torchrun_env("sample")
	raw_params = _load_params_json(args.params)
	params = _validate_params_all(raw_params, require_priors=True, mode="sample")
	_apply_torch_runtime_settings(params)

	if args.device is None:
		raise ValueError("`spider sample` requires --device <CUDA_ID> (CLI-only device selection).")
	device_id = int(args.device)
	params["devices"] = [device_id]

	device = _device_from_id(int(device_id))
	if device.type == "cuda":
		try:
			torch.cuda.set_device(device)
		except Exception:
			pass

	params["_ddp_world_size"] = 1
	params["_ddp_rank"] = 0
	params["_ddp_local_rank"] = 0
	params["_use_dataparallel"] = False

	if getattr(args, "shift_guard", False):
		params["shift_guard_enable"] = True
	if getattr(args, "shift_guard_factor", None) is not None:
		params["shift_guard_factor"] = float(args.shift_guard_factor)

	# Shared-event RE visibility check (helps verify activation).
	try:
		lk_groups = raw_params.get("model", {}).get("likelihoods", {}) if isinstance(raw_params, dict) else {}
		lk = lk_groups.get("sample", {}) if isinstance(lk_groups, dict) else {}
		shared_re_cfg = lk.get("shared_event_re", None) if isinstance(lk, dict) else None
		if shared_re_cfg is not None:
			enabled = bool(params.get("_shared_event_re_enabled", False))
			grouping = str(params.get("_shared_event_re_grouping", ""))
			_log(f"[shared_event_re] config present -> enabled={enabled} grouping={grouping}")
	except Exception:
		pass

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

	wandb_logger = init_wandb_if_enabled(params)
	params["_wandb_runtime_enabled"] = bool(wandb_logger is not None)

	model = _load_model(params, device)

	info(f"Running SPIDER sampling from bundle={bundle_path}", section="RUN")
	locate_sample_from_bundle(params=params, bundle_path=bundle_path, model=model, device=device, wandb_logger=wandb_logger)

	if wandb_logger:
		wandb_logger.finish()
	return 0


def _cmd_analyze_resid(args: argparse.Namespace) -> int:
	"""Run post-Phase1 residual diagnostics from an existing Phase-2 bundle (no MAP rerun)."""
	params = _validate_params_all(_load_params_json(args.params), require_priors=True, mode="analyze-resid")
	_apply_torch_runtime_settings(params)

	if args.device is None:
		raise ValueError("`spider analyze-resid` requires --device <CUDA_ID> (CLI-only device selection).")
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
		use_latest_checkpoint=bool(getattr(args, "use_latest_checkpoint", False)),
	)
	return 0


def _cmd_sample_multi(args: argparse.Namespace) -> int:
	"""
	Launch multiple independent sampling chains (one process per GPU/device), starting from the same bundle.
	"""
	with open(args.params, "r") as f:
		base_params = json.load(f)

	# Devices must be provided explicitly on CLI.
	devs = None
	if getattr(args, "devices", None):
		devs = [_parse_device_entry(x.strip()) for x in str(args.devices).split(",") if str(x).strip()]
	else:
		devs = []
	if not devs:
		raise ValueError("`spider sample-multi` requires --devices <id0,id1,...> (CLI-only device selection).")

	bundle_path = str(getattr(args, "bundle", "") or "").strip()
	if not bundle_path:
		# Infer default bundle path from base checkpoint_dir
		try:
			p0 = _validate_params_all(json.loads(json.dumps(base_params)), require_priors=True, mode="sample-multi")
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
		if isinstance(p.get("observability", None), dict) and isinstance(p["observability"].get("wandb", None), dict):
			rn = str(p["observability"]["wandb"].get("run_name", "spider"))
			p["observability"]["wandb"]["run_name"] = f"{rn}_chain{ci}"

		fd, tmp_path = tempfile.mkstemp(prefix=f"spider_chain{ci}_", suffix=".json", dir=out_dir)
		os.close(fd)
		with open(tmp_path, "w") as f:
			json.dump(p, f, indent=4)
		tmp_paths.append(tmp_path)

		cmd = [sys.executable, "-m", "spider", "sample", tmp_path, "--device", str(dev), "--bundle", str(bundle_path)]
		_log(f"[sample-multi] chain={ci} device={dev} seed={p['inference']['runtime']['seed']} samples={p['io']['samples_outfile']}")
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
				_log(f"[sample-multi] merged chains -> {merged_out_path}")
			except Exception as e:
				_log(f"[sample-multi] WARNING: merge failed: {e}")
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
	params = _validate_params_synth(_load_params_json(args.params))
	_apply_torch_runtime_settings(params)
	if args.device is None:
		raise ValueError("`spider synth` requires --device <CUDA_ID> (CLI-only device selection).")
	device_id = int(args.device)
	if torch.cuda.is_available():
		device = torch.device(f"cuda:{device_id}")
	else:
		warn("CUDA is not available; running on CPU.", section="RUN")
		device = torch.device("cpu")

	# Load model early (needed when apply_filters=True calls prepare_input_dfs(..., model=model)).
	model = _load_model(params, device)

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
			_log(f"Wrote truth catalog to {true_out} (N={res.truth.shape[0]})")
		if init_out is not None:
			_safe_write_csv(res.init, str(init_out), "init_catalog_outfile")
			_log(f"Wrote synthetic initial catalog to {init_out} (N={res.init.shape[0]})")

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
		if lik in {"student_t", "student-t", "studentt"}:
			try:
				nu = float(params.get("_student_t_nu", 4.0))
			except Exception:
				nu = 4.0
			if not (nu > 0.0):
				nu = 4.0
			return (rng.standard_t(df=nu, size=sigma_obs.shape).astype(np.float32) * sigma_obs).astype(np.float32)
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
				dt_pred[i0:i1] = compute_travel_times(II_chunk, Y_chunk, X_src, dX_src, model, params=params).detach().cpu().numpy().astype(np.float32)

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
			_log(f"Wrote synthetic differential times to {out_path} (N={out_df.shape[0]})")
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
			dt_pred = compute_travel_times(II_chunk, Y_chunk, X_src, dX_src, model, params=params).detach().cpu().numpy().astype(np.float32)
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

	_log(f"Wrote synthetic differential times to {out_path} (N={total_written})")
	return 0


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog=prog or "spider", description="SPIDER command-line interface")
	subparsers = parser.add_subparsers(dest="command", metavar="command")

	# locate-map subcommand
	p_map = subparsers.add_parser("locate-map", help="Run Phase 1 (MAP) only and dump a Phase-2 bundle")
	p_map.add_argument("params", help="Path to parameter JSON file")
	p_map.add_argument("--device", type=int, required=True, help="CUDA device id to use (required)")
	p_map.add_argument("--bundle-out", type=str, default=None, help="Output path for Phase-2 bundle (default: <checkpoint_dir>/phase2_bundle.pth)")
	p_map.add_argument("--shift-guard", action="store_true", help="Abort if any event moves > factor × prior stds; prints offending observations")
	p_map.add_argument("--shift-guard-factor", type=float, default=None, help="Factor relative to prior_event_std; default 5.0")
	p_map.set_defaults(func=_cmd_locate_map)

	# sample subcommand
	p_samp = subparsers.add_parser("sample", help="Run Phase 2–4 (sampling) starting from a Phase-2 bundle")
	p_samp.add_argument("params", help="Path to parameter JSON file")
	p_samp.add_argument("--device", type=int, required=True, help="CUDA device id to use (required)")
	p_samp.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle (default: <checkpoint_dir>/phase2_bundle.pth)")
	p_samp.add_argument("--shift-guard", action="store_true", help="Abort if any event moves > factor × prior stds; prints offending observations")
	p_samp.add_argument("--shift-guard-factor", type=float, default=None, help="Factor relative to prior_event_std; default 5.0")
	p_samp.set_defaults(func=_cmd_sample)

	# sample-multi subcommand
	p_sm = subparsers.add_parser("sample-multi", help="Launch multiple independent sampling chains from the same Phase-2 bundle")
	p_sm.add_argument("params", help="Path to parameter JSON file")
	p_sm.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle (default: <checkpoint_dir>/phase2_bundle.pth)")
	p_sm.add_argument("--chains", type=int, default=None, help="Number of chains to run (default: number of devices)")
	p_sm.add_argument("--devices", type=str, required=True, help="Comma-separated CUDA device ids (required)")
	p_sm.add_argument("--seed0", type=int, default=0, help="Base seed for chain RNG offsets (default: 0)")
	p_sm.add_argument("--out-dir", type=str, default=None, help="Directory to write chain param files (default: params directory)")
	p_sm.add_argument("--dry-run", action="store_true", help="Print what would run, but do not start processes")
	p_sm.add_argument("--no-merge", dest="merge", action="store_false", help="Do not merge chain HDF5 outputs into base io.samples_outfile")
	p_sm.set_defaults(func=_cmd_sample_multi)

	# analyze-resid subcommand
	p_ar = subparsers.add_parser("analyze-resid", help="Run residual diagnostics from an existing Phase-2 bundle (no MAP rerun)")
	p_ar.add_argument("params", help="Path to parameters JSON file")
	p_ar.add_argument("--device", type=int, required=True, help="CUDA device id to use (required)")
	p_ar.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle (.pth). Default: checkpoint_dir/phase2_bundle.pth")
	p_ar.add_argument(
		"--use-latest-checkpoint",
		action="store_true",
		help="If set, load the most recent checkpoint from checkpoint_dir and evaluate diagnostics at that state (e.g., after `spider sample`).",
	)
	g_plot = p_ar.add_mutually_exclusive_group()
	g_plot.add_argument("--plot-variograms", dest="plot_variograms", action="store_true", help="Write variogram PNG plots (default)")
	g_plot.add_argument("--no-plot-variograms", dest="plot_variograms", action="store_false", help="Disable variogram plotting")
	p_ar.set_defaults(plot_variograms=True)
	p_ar.add_argument("--plot-dir", type=str, default=None, help="Directory for variogram PNGs (default: bundle directory)")

	p_ar.set_defaults(func=_cmd_analyze_resid)

	# locate subcommand
	p_locate = subparsers.add_parser("locate", help="Alias for `sample` (Phase 2–4). Use `locate-map` for Phase 1.")
	p_locate.add_argument("params", help="Path to parameter JSON file")
	p_locate.add_argument(
		"--device",
		type=int,
		required=True,
		help="CUDA device id to use (required)",
	)
	p_locate.add_argument("--shift-guard", action="store_true", help="Abort if any event moves > factor × prior stds; prints offending observations")
	p_locate.add_argument("--shift-guard-factor", type=float, default=None, help="Factor relative to prior_event_std; default 5.0")
	p_locate.set_defaults(func=_cmd_locate)

	# locate-multi subcommand
	p_multi = subparsers.add_parser("locate-multi", help="Alias for `sample-multi` (starts from Phase-2 bundle)")
	p_multi.add_argument("params", help="Path to parameter JSON file")
	p_multi.add_argument("--bundle", type=str, default=None, help="Path to Phase-2 bundle (default: <checkpoint_dir>/phase2_bundle.pth)")
	p_multi.add_argument("--chains", type=int, default=None, help="Number of chains to run (default: number of devices)")
	p_multi.add_argument("--devices", type=str, required=True, help="Comma-separated CUDA device ids (required)")
	p_multi.add_argument("--seed0", type=int, default=0, help="Base seed for chain RNG offsets (default: 0)")
	p_multi.add_argument("--out-dir", type=str, default=None, help="Directory to write chain param files (default: params directory)")
	p_multi.add_argument("--dry-run", action="store_true", help="Print what would run, but do not start processes")
	p_multi.add_argument("--no-merge", dest="merge", action="store_false", help="Do not merge chain HDF5 outputs into base io.samples_outfile")
	p_multi.set_defaults(func=_cmd_locate_multi)

	# locate-full subcommand (kept for one-shot legacy flows)
	p_full = subparsers.add_parser("locate-full", help="Run full pipeline (Phase 1 + Phase 2–4) [legacy]")
	p_full.add_argument("params", help="Path to parameter JSON file")
	p_full.add_argument("--device", type=int, required=True, help="CUDA device id to use (required)")
	p_full.add_argument("--shift-guard", action="store_true", help="Abort if any event moves > factor × prior stds; prints offending observations")
	p_full.add_argument("--shift-guard-factor", type=float, default=None, help="Factor relative to prior_event_std; default 5.0")
	p_full.set_defaults(func=_cmd_locate_full)

	# synth subcommand
	p_synth = subparsers.add_parser("synth", help="Generate a synthetic differential time dataset matching the current configuration")
	p_synth.add_argument("params", help="Path to parameter JSON file")
	p_synth.add_argument("--device", type=int, required=True, help="CUDA device id to use (required)")
	p_synth.set_defaults(func=_cmd_synth)

	# validate-config subcommand (config_v2 only)
	p_val = subparsers.add_parser("validate-config", help="Validate params against canonical config_v2 schema (breaking, no compatibility layer)")
	p_val.add_argument("params", help="Path to parameter JSON file")
	p_val.add_argument(
		"--mode",
		type=str,
		default=None,
		choices=["locate-map", "sample", "sample-multi", "analyze-resid", "locate-full", "synth"],
		help="Optional command mode for mode-specific validation checks.",
	)
	p_val.add_argument(
		"--print-resolved",
		action="store_true",
		help="Print resolved runtime config JSON after validation.",
	)
	p_val.set_defaults(func=_cmd_validate_config)

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
