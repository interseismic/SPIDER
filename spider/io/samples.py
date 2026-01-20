from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional

import h5py
import numpy as np
import torch
from pyproj import Proj


SAMPLE_FIELDS = ["longitude", "latitude", "depth", "delta_t", "X", "Y", "Z"]


def _open_or_init_store(store_path: str, n_events: int, event_ids) -> h5py.File:
    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
    f = h5py.File(store_path, "a", libver="latest")
    f.attrs["n_events"] = int(n_events)
    try:
        f.attrs["event_ids_json"] = json.dumps(list(event_ids))
    except Exception:
        f.attrs["event_ids_json"] = json.dumps([str(i) for i in range(n_events)])
    return f


def save_map_locations(params: Dict[str, Any], origins0, X_src: torch.Tensor, dX_src: torch.Tensor, projector: Proj) -> bool:
    try:
        store_path = params.get("samples_outfile", "samples.h5")
        X_src1 = (X_src + dX_src).detach().cpu().numpy().astype("float32")
        N = X_src1.shape[0]
        lons = np.empty((N,), dtype=np.float32)
        lats = np.empty((N,), dtype=np.float32)
        for i in range(N):
            lo, la = projector(X_src1[i, 0], X_src1[i, 1], inverse=True)
            lons[i] = np.float32(lo)
            lats[i] = np.float32(la)
        deps = X_src1[:, 2].astype("float32")
        event_ids = [str(e) for e in origins0["evid"]]
        f = _open_or_init_store(store_path, N, event_ids)
        for name, arr in (("map_longitude", lons), ("map_latitude", lats), ("map_depth", deps)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=arr, dtype="float32", shape=arr.shape)
        f.attrs["map_present"] = True
        f.flush()
        f.close()
        print(f"Wrote MAP locations to {store_path} (map_longitude/map_latitude/map_depth)")
        return True
    except Exception as e:
        print(f"Warning: could not save MAP locations to HDF5: {e}")
        return False


def clear_samples_file(params: Dict[str, Any]) -> bool:
    store_path = params.get("samples_outfile", "samples.h5")
    if not os.path.exists(store_path):
        return True
    try:
        os.remove(store_path)
        return True
    except Exception:
        return False


def get_next_sample_count(params: Dict[str, Any]) -> int:
    store_path = params.get("samples_outfile", "samples.h5")
    if not os.path.exists(store_path):
        return 0
    try:
        with h5py.File(store_path, "r") as f:
            nums = []
            for name in f.keys():
                if isinstance(f[name], h5py.Group) and name.startswith("batch_"):
                    try:
                        nums.append(int(name.split("_")[1]))
                    except Exception:
                        pass
            return (max(nums) + 1) if nums else 0
    except Exception:
        return 0


def read_all_samples(
    params,
    backend: str = "numpy",
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    pin_memory: bool = False,
    thin: int = 1,
) -> Dict[str, np.ndarray] | Dict[str, torch.Tensor]:
    """Read all samples from HDF5 file with optional thinning.

    Args:
        params: Parameter dictionary containing 'samples_outfile' OR a path to params.json
        backend: 'numpy' or 'torch' for output format
        device: Target device for torch tensors
        dtype: Data type for torch tensors
        pin_memory: Whether to pin memory for torch tensors
        thin: Thinning factor - keep every nth sample (default: 1 = no thinning)
    """
    if isinstance(params, str):
        try:
            with open(params, "r") as f:
                params = json.load(f)
        except Exception as e:
            raise ValueError(f"read_all_samples: failed to read params from '{params}': {e}")

    if isinstance(params, dict) and ("samples_outfile" not in params):
        io_cfg = params.get("io", None)
        if isinstance(io_cfg, dict) and ("samples_outfile" in io_cfg):
            v = io_cfg.get("samples_outfile")
            if v is not None:
                params["samples_outfile"] = v

    store_path = params.get("samples_outfile", None)
    if not store_path:
        store_path = "samples.h5"
    if not os.path.exists(store_path):
        print(f"Samples store not found: {store_path}")
        return {}

    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")

    with h5py.File(store_path, "r") as f:
        # Optional MAP datasets at root
        map_lon_root = f["map_longitude"][:] if "map_longitude" in f else None
        map_lat_root = f["map_latitude"][:] if "map_latitude" in f else None
        map_dep_root = f["map_depth"][:] if "map_depth" in f else None

        # Collect batch groups (stable sort by parsed index)
        batches: list[tuple[int, str, Any]] = []
        for name in f.keys():
            if isinstance(f[name], h5py.Group) and name.startswith("batch_"):
                try:
                    num = int(name.split("_")[1])
                    batches.append((num, str(name), f[name]))
                except Exception:
                    pass
        if not batches:
            # MAP-only mode (no samples)
            if map_lon_root is not None and map_lat_root is not None and map_dep_root is not None:
                try:
                    ids = json.loads(f.attrs.get("event_ids_json", "[]"))
                    n_events_meta = int(f.attrs.get("n_events", 0))
                    if not ids or (n_events_meta > 0 and len(ids) != n_events_meta):
                        ids = [str(i) for i in range(n_events_meta)]
                    event_ids = np.asarray(ids, dtype=str)
                except Exception:
                    n_events_meta = int(map_lon_root.shape[0])
                    event_ids = np.asarray([str(i) for i in range(n_events_meta)], dtype=str)

                n_events = int(n_events_meta) if int(n_events_meta) > 0 else int(map_lon_root.shape[0])
                map_lon = np.asarray(map_lon_root, dtype=np.float32)[:n_events]
                map_lat = np.asarray(map_lat_root, dtype=np.float32)[:n_events]
                map_dep = np.asarray(map_dep_root, dtype=np.float32)[:n_events]
                if event_ids.shape[0] != n_events:
                    event_ids = event_ids[:n_events]

                out_map: Dict[str, Any] = {
                    "event_ids": event_ids,
                    "longitude": map_lon.reshape(n_events, 1),
                    "latitude": map_lat.reshape(n_events, 1),
                    "depth": map_dep.reshape(n_events, 1),
                    "delta_t": np.zeros((n_events, 1), dtype=np.float32),
                    "X": np.zeros((n_events, 1), dtype=np.float32),
                    "Y": np.zeros((n_events, 1), dtype=np.float32),
                    "Z": np.zeros((n_events, 1), dtype=np.float32),
                    "map_longitude": map_lon,
                    "map_latitude": map_lat,
                    "map_depth": map_dep,
                }
                print("No batch_* groups found; returning MAP-only samples (n_samples=1).")
                if str(backend).lower() == "numpy":
                    return out_map  # type: ignore[return-value]
                target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                out_torch: Dict[str, torch.Tensor] | Dict[str, np.ndarray] = {"event_ids": out_map["event_ids"]}
                for name in SAMPLE_FIELDS:
                    arr = out_map[name]
                    t = torch.from_numpy(arr).to(dtype)
                    if pin_memory and target_device != "cpu":
                        t = t.pin_memory()
                    out_torch[name] = t.to(target_device, non_blocking=True) if target_device != "cpu" else t
                for name in ("map_longitude", "map_latitude", "map_depth"):
                    arr = out_map[name]
                    t = torch.from_numpy(arr).to(dtype)
                    if pin_memory and target_device != "cpu":
                        t = t.pin_memory()
                    out_torch[name] = t.to(target_device, non_blocking=True) if target_device != "cpu" else t
                return out_torch  # type: ignore[return-value]

            print("No batches found in samples store")
            return {}
        batches.sort(key=lambda x: x[0])

        try:
            ids = json.loads(f.attrs.get("event_ids_json", "[]"))
            n_events_meta = int(f.attrs.get("n_events", 0))
            if not ids or (n_events_meta > 0 and len(ids) != n_events_meta):
                ids = [str(i) for i in range(n_events_meta)]
            event_ids = np.asarray(ids, dtype=str)
        except Exception:
            n_events_meta = batches[0][2]["longitude"].shape[0]
            event_ids = np.asarray([str(i) for i in range(n_events_meta)], dtype=str)

        # Canonical event count (skip mismatched batches by default)
        mismatch_mode = str(params.get("read_samples_mismatch_mode", "skip")).strip().lower()
        if mismatch_mode not in {"skip", "min", "error"}:
            mismatch_mode = "skip"
        n_events_per_batch: list[int] = []
        for _, _, grp in batches:
            try:
                n_events_per_batch.append(int(grp["longitude"].shape[0]))
            except Exception:
                pass
        if not n_events_per_batch:
            print("No valid batches found in samples store")
            return {}
        canonical_n_events = int(n_events_meta) if int(n_events_meta) > 0 else int(n_events_per_batch[0])

        if mismatch_mode == "min":
            n_events = int(min(n_events_per_batch))
        else:
            n_events = int(canonical_n_events)
            keep: list[tuple[int, str, Any]] = []
            dropped: list[int] = []
            for (num, name, grp), ne in zip(batches, n_events_per_batch):
                if int(ne) == int(n_events):
                    keep.append((num, name, grp))
                else:
                    dropped.append(int(ne))
            if dropped:
                msg = (
                    f"Inconsistent event counts across sample batches: kept only batches with n_events={n_events}; "
                    f"dropped batches with n_events={sorted(set(dropped))}. "
                    f"(override with read_samples_mismatch_mode='min' or 'error')"
                )
                if mismatch_mode == "error":
                    raise ValueError(msg)
                print(f"Warning: {msg}")
            batches = keep if keep else batches

        # Calculate total samples after thinning
        total_samples = 0
        batch_names: list[str] = []
        batch_sample_counts: list[int] = []
        for _, bname, grp in batches:
            try:
                sample_ds = grp["longitude"]
                n_samples_in_batch = int(sample_ds.shape[1]) if sample_ds.ndim == 2 else 1
            except Exception:
                n_samples_in_batch = 0
            if thin == 1:
                n_keep = int(n_samples_in_batch)
            else:
                n_keep = int(len(range(0, int(n_samples_in_batch), int(thin))))
            total_samples += int(n_keep)
            batch_names.append(str(bname))
            batch_sample_counts.append(int(n_keep))

        # Align event_ids
        if int(getattr(event_ids, "shape", [0])[0]) != int(n_events):
            event_ids = event_ids[:n_events]

        out: Dict[str, Any] = {
            "event_ids": event_ids,
            "longitude": np.empty((n_events, total_samples), dtype=np.float32),
            "latitude": np.empty((n_events, total_samples), dtype=np.float32),
            "depth": np.empty((n_events, total_samples), dtype=np.float32),
            "delta_t": np.empty((n_events, total_samples), dtype=np.float32),
            "X": np.empty((n_events, total_samples), dtype=np.float32),
            "Y": np.empty((n_events, total_samples), dtype=np.float32),
            "Z": np.empty((n_events, total_samples), dtype=np.float32),
        }
        try:
            out["_batch_names"] = list(batch_names)
            out["_batch_sample_counts"] = np.asarray(batch_sample_counts, dtype=np.int64)
            csum = np.cumsum(np.asarray(batch_sample_counts, dtype=np.int64))
            out["_batch_boundaries"] = csum[:-1].astype(np.int64, copy=False)
            starts = np.concatenate([np.asarray([0], dtype=np.int64), csum[:-1]]).astype(np.int64, copy=False)
            out["_batch_slices"] = {str(nm): (int(s), int(e)) for nm, s, e in zip(batch_names, starts, csum)}
        except Exception:
            pass
        if map_lon_root is not None and map_lat_root is not None and map_dep_root is not None:
            out["map_longitude"] = map_lon_root[:n_events].astype(np.float32, copy=False)
            out["map_latitude"] = map_lat_root[:n_events].astype(np.float32, copy=False)
            out["map_depth"] = map_dep_root[:n_events].astype(np.float32, copy=False)

        # Read batches
        offset = 0
        for _, _, grp in batches:
            try:
                ds0 = grp["longitude"]
                n_samples_in_batch = int(ds0.shape[1]) if ds0.ndim == 2 else 1
            except Exception:
                continue
            idx = list(range(0, int(n_samples_in_batch), int(thin)))
            keep = len(idx)
            if keep == 0:
                continue
            for name in SAMPLE_FIELDS:
                if name not in grp:
                    continue
                ds = grp[name]
                if ds.ndim == 1:
                    arr = np.asarray(ds[:], dtype=np.float32).reshape(n_events, 1)
                else:
                    arr = np.asarray(ds[:, idx], dtype=np.float32)
                out[name][:, offset:offset + keep] = arr[:, :keep]
            offset += keep

        if str(backend).lower() == "numpy":
            return out  # type: ignore[return-value]

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        out_torch: Dict[str, torch.Tensor] | Dict[str, np.ndarray] = {"event_ids": out["event_ids"]}
        for name in SAMPLE_FIELDS:
            arr = out[name]
            t = torch.from_numpy(arr).to(dtype)
            if pin_memory and target_device != "cpu":
                t = t.pin_memory()
            out_torch[name] = t.to(target_device, non_blocking=True) if target_device != "cpu" else t
        for name in ("map_longitude", "map_latitude", "map_depth"):
            if name in out:
                arr = out[name]
                t = torch.from_numpy(arr).to(dtype)
                if pin_memory and target_device != "cpu":
                    t = t.pin_memory()
                out_torch[name] = t.to(target_device, non_blocking=True) if target_device != "cpu" else t
        return out_torch  # type: ignore[return-value]


def save_samples_periodic(
    *,
    params: Dict[str, Any],
    origins0,
    X_src: torch.Tensor,
    dX_src: torch.Tensor,
    projector: Proj,
    sample_count: int,
) -> Optional[str]:
    try:
        store_path = params.get("samples_outfile", "samples.h5")
        X_src1 = (X_src + dX_src).detach().cpu().numpy().astype("float32")
        N = X_src1.shape[0]
        lons = np.empty((N,), dtype=np.float32)
        lats = np.empty((N,), dtype=np.float32)
        for i in range(N):
            lo, la = projector(X_src1[i, 0], X_src1[i, 1], inverse=True)
            lons[i] = np.float32(lo)
            lats[i] = np.float32(la)
        deps = X_src1[:, 2].astype("float32")
        dt = X_src1[:, 3].astype("float32")
        event_ids = [str(e) for e in origins0["evid"]]

        f = _open_or_init_store(store_path, N, event_ids)
        grp = f.create_group(f"batch_{int(sample_count)}")
        grp.create_dataset("longitude", data=lons, dtype="float32")
        grp.create_dataset("latitude", data=lats, dtype="float32")
        grp.create_dataset("depth", data=deps, dtype="float32")
        grp.create_dataset("delta_t", data=dt, dtype="float32")
        grp.create_dataset("X", data=X_src1[:, 0], dtype="float32")
        grp.create_dataset("Y", data=X_src1[:, 1], dtype="float32")
        grp.create_dataset("Z", data=X_src1[:, 2], dtype="float32")
        f.flush()
        f.close()
        return store_path
    except Exception as e:
        print(f"Warning: failed to write samples: {e}")
        return None
