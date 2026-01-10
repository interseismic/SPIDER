import os
from typing import Dict, Any, Optional, List

import json
import numpy as np
import torch
import h5py
from pyproj import Proj


SAMPLE_FIELDS = ['longitude', 'latitude', 'depth', 'delta_t', 'X', 'Y', 'Z']

def merge_samples_hdf5(
    *,
    out_path: str,
    in_paths: List[str],
    overwrite: bool = True,
) -> str:
    """
    Merge multiple SPIDER samples HDF5 files (each with batch_* groups) into a single output file.

    This is designed for multi-chain runs (one samples file per chain). We copy each batch group
    into the output as a new batch_{k} in increasing order (stable, deterministic).
    
    Chain identity:
    - The merged file preserves chain identity by writing provenance metadata on each output
      `batch_*` group:
        - attrs['chain_idx'] (int): index within `in_paths`
        - attrs['chain_file'] (str): basename of source file
        - attrs['source_batch'] (str): original group name in the source file (e.g. 'batch_12')
        - attrs['source_batch_idx'] (int): parsed index from source_batch, or -1 if unavailable

    Assumptions:
    - All inputs have the same event_ids_json / n_events (or are compatible).
    - Output file will be overwritten by default.
    """
    in_paths = [str(p) for p in in_paths if p]
    if not in_paths:
        raise ValueError("merge_samples_hdf5: in_paths is empty")

    if overwrite and os.path.exists(out_path):
        os.remove(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def _list_batches(f: h5py.File):
        out = []
        for name in f.keys():
            if isinstance(f[name], h5py.Group) and str(name).startswith("batch_"):
                try:
                    out.append((int(str(name).split("_")[1]), str(name)))
                except Exception:
                    continue
        out.sort(key=lambda x: x[0])
        return [nm for _, nm in out]

    # Read canonical metadata from first file
    with h5py.File(in_paths[0], "r") as f0:
        n_events = int(f0.attrs.get("n_events", 0))
        event_ids_json = str(f0.attrs.get("event_ids_json", "[]"))
        map_lon = f0["map_longitude"][:] if "map_longitude" in f0 else None
        map_lat = f0["map_latitude"][:] if "map_latitude" in f0 else None
        map_dep = f0["map_depth"][:] if "map_depth" in f0 else None

    with h5py.File(out_path, "a") as fout:
        fout.attrs["n_events"] = int(n_events)
        fout.attrs["event_ids_json"] = event_ids_json
        # Best-effort provenance at file level
        try:
            fout.attrs["n_chains"] = int(len(in_paths))
            fout.attrs["chain_files_json"] = json.dumps([os.path.basename(str(p)) for p in in_paths])
        except Exception:
            pass
        if map_lon is not None and map_lat is not None and map_dep is not None:
            for name, arr in (("map_longitude", map_lon), ("map_latitude", map_lat), ("map_depth", map_dep)):
                if name in fout:
                    del fout[name]
                fout.create_dataset(name, data=np.asarray(arr, dtype=np.float32))
            fout.attrs["map_present"] = True

        next_batch = 0
        for chain_idx, ip in enumerate(in_paths):
            with h5py.File(ip, "r") as fin:
                ne = int(fin.attrs.get("n_events", 0))
                ids = str(fin.attrs.get("event_ids_json", "[]"))
                if n_events > 0 and ne > 0 and int(ne) != int(n_events):
                    raise ValueError(f"merge_samples_hdf5: n_events mismatch {ip}: {ne} != {n_events}")
                if ids and event_ids_json and (ids != event_ids_json):
                    # Be strict: mixing different event-id orderings breaks downstream analysis.
                    raise ValueError(f"merge_samples_hdf5: event_ids_json mismatch for {ip}")

                for bname in _list_batches(fin):
                    grp_in = fin[bname]
                    grp_out = fout.create_group(f"batch_{next_batch}")
                    # Copy attrs best-effort
                    try:
                        for k, v in grp_in.attrs.items():
                            grp_out.attrs[k] = v
                    except Exception:
                        pass
                    # Add chain provenance (best-effort; must not break merge)
                    try:
                        grp_out.attrs["chain_idx"] = int(chain_idx)
                        grp_out.attrs["chain_file"] = str(os.path.basename(str(ip)))
                        grp_out.attrs["source_batch"] = str(bname)
                        try:
                            grp_out.attrs["source_batch_idx"] = int(str(bname).split("_")[1])
                        except Exception:
                            grp_out.attrs["source_batch_idx"] = int(-1)
                    except Exception:
                        pass

                    # Copy datasets
                    for dset_name in list(grp_in.keys()):
                        obj = grp_in[dset_name]
                        if not isinstance(obj, h5py.Dataset):
                            continue
                        ds_in = obj
                        # Create matching dataset in output
                        ds_out = grp_out.create_dataset(
                            dset_name,
                            shape=ds_in.shape,
                            dtype=ds_in.dtype,
                            chunks=ds_in.chunks,
                            compression=ds_in.compression,
                            compression_opts=ds_in.compression_opts,
                            shuffle=ds_in.shuffle,
                        )
                        # Chunked copy along sample dimension when 2D
                        if ds_in.ndim == 2:
                            # (n_events, n_samples)
                            n_samp = int(ds_in.shape[1])
                            step = 128
                            for j0 in range(0, n_samp, step):
                                j1 = min(j0 + step, n_samp)
                                ds_out[:, j0:j1] = ds_in[:, j0:j1]
                        else:
                            ds_out[...] = ds_in[...]
                    next_batch += 1

        fout.flush()
    return out_path


def _open_or_init_store(store_path: str, n_events: int, event_ids) -> h5py.File:
    os.makedirs(os.path.dirname(store_path) or '.', exist_ok=True)
    f = h5py.File(store_path, 'a', libver='latest')
    f.attrs['n_events'] = int(n_events)
    try:
        f.attrs['event_ids_json'] = json.dumps(list(event_ids))
    except Exception:
        f.attrs['event_ids_json'] = json.dumps([str(i) for i in range(n_events)])
    return f


def _ensure_root_datasets(f: h5py.File, n_events: int, sample_capacity: int, params: Dict[str, Any]):
    """Ensure resizable 2D datasets exist at root for append mode."""
    chunk_events = min(n_events, int(params.get('io_event_chunk', 2048)))
    chunk_samples = min(max(sample_capacity, 1), int(params.get('hdf5_chunk_samples', 32)))
    compression = params.get('hdf5_compression', None)
    compression_opts = params.get('hdf5_compression_opts', None)
    shuffle = True if params.get('hdf5_shuffle', False) else False

    for name in SAMPLE_FIELDS:
        if name not in f:
            f.create_dataset(
                name,
                shape=(n_events, 0),
                maxshape=(n_events, None),
                chunks=(chunk_events, chunk_samples),
                dtype='float32',
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
            )

def save_map_locations(
    params: Dict[str, Any],
    origins0,
    X_src: torch.Tensor,
    dX_src: torch.Tensor,
    projector,
) -> bool:
    """
    Save per-event MAP locations (lon, lat, depth) to the samples HDF5 file at the root level.
    Overwrites existing datasets if present.
    """
    try:
        store_path = params.get('samples_outfile', 'samples.h5')
        lat0 = float(params['lat_min'])
        lon0 = float(params['lon_min'])
        # Compute MAP in projected coords
        X_src1 = (X_src + dX_src).detach().cpu().numpy().astype('float32')  # (N,4)
        N = X_src1.shape[0]
        # Convert to lon/lat
        lons = np.empty((N,), dtype=np.float32)
        lats = np.empty((N,), dtype=np.float32)
        for i in range(N):
            lo, la = projector(X_src1[i, 0], X_src1[i, 1], inverse=True)
            lons[i] = np.float32(lo)
            lats[i] = np.float32(la)
        deps = X_src1[:, 2].astype('float32')
        # Event IDs
        try:
            event_ids = [str(e) for e in origins0['evid']]
        except Exception:
            event_ids = [str(e) for e in origins0['evid']]
        f = _open_or_init_store(store_path, N, event_ids)
        # Create/overwrite datasets
        for name, arr in (('map_longitude', lons), ('map_latitude', lats), ('map_depth', deps)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=arr, dtype='float32', shape=arr.shape)
        f.attrs['map_present'] = True
        f.flush()
        f.close()
        print(f"Wrote MAP locations to {store_path} (map_longitude/map_latitude/map_depth)")
        return True
    except Exception as e:
        print(f"Warning: could not save MAP locations to HDF5: {e}")
        return False


def clear_samples_file(params) -> bool:
    store_path = params.get('samples_outfile', 'samples.h5')
    if not os.path.exists(store_path):
        return True
    try:
        os.remove(store_path)
        return True
    except Exception:
        return False


def get_next_sample_count(params) -> int:
    store_path = params.get('samples_outfile', 'samples.h5')
    if not os.path.exists(store_path):
        return 0
    try:
        with h5py.File(store_path, 'r') as f:
            nums = []
            for name in f.keys():
                if isinstance(f[name], h5py.Group) and name.startswith('batch_'):
                    try:
                        nums.append(int(name.split('_')[1]))
                    except Exception:
                        pass
            return (max(nums) + 1) if nums else 0
    except Exception:
        return 0


def save_samples_periodic(
    params: Dict[str, Any],
    origins0,
    X_src: torch.Tensor,
    samples: list,
    projector,
    sample_count: int,
    *,
    noise_log_scales: Optional[List[torch.Tensor]] = None,
    global_step_count: Optional[int] = None,
    epoch: Optional[int] = None,
    phase: Optional[str] = None,
) -> int:
    if len(samples) == 0:
        return sample_count

    try:
        n_samples = len(samples)
        n_events_in = int(X_src.shape[0])
        print(f"[Samples] Starting save: {n_samples} samples, {n_events_in} events")
        
        store_path = params.get('samples_outfile', 'samples.h5')
        lat0 = float(params['lat_min'])
        lon0 = float(params['lon_min'])

        # 1. Prepare data on CPU
        print(f"[Samples] Validating and converting samples to numpy...")
        samples_np = []
        for i, s in enumerate(samples):
            # Check for numerical explosion before converting to numpy
            if isinstance(s, torch.Tensor):
                if not torch.isfinite(s).all():
                    print(f"[Samples] Warning: Sample {i} contains non-finite values (inf/nan). Cleaning.")
                    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                samples_np.append(s.detach().cpu().numpy().astype('float32'))
            else:
                s_np = np.asarray(s, dtype='float32')
                if not np.isfinite(s_np).all():
                    print(f"[Samples] Warning: Sample {i} contains non-finite values. Cleaning.")
                    s_np = np.nan_to_num(s_np)
                samples_np.append(s_np)
        
        S_batch = len(samples_np)
        n_events = samples_np[0].shape[0] if S_batch > 0 else 0
        
        if n_events != n_events_in:
             print(f"[Samples] Warning: n_events mismatch! samples={n_events}, X_src={n_events_in}")

        try:
            event_ids = [str(evid) for evid in origins0['evid']]
        except Exception:
            event_ids = [str(e) for e in range(n_events)]

        # 2. Open HDF5
        print(f"[Samples] Opening HDF5: {store_path}")
        # Avoid libver='latest' if it causes issues on some systems
        f = h5py.File(store_path, 'a') 
        f.attrs['n_events'] = int(n_events)
        try:
            f.attrs['event_ids_json'] = json.dumps(list(event_ids))
        except Exception:
            pass

        # Determine next batch index
        batch_idx = get_next_sample_count(params)
        batch_name = f"batch_{batch_idx}"
        while batch_name in f:
            batch_idx += 1
            batch_name = f"batch_{batch_idx}"
        
        print(f"[Samples] Creating group {batch_name}")
        batch = f.create_group(batch_name)
        try:
            batch.attrs["n_events"] = int(n_events)
            batch.attrs["event_ids_json"] = json.dumps(list(event_ids))
            # Provenance (best-effort): helps diagnose "mode switching" artifacts from appended runs.
            # These are purely informational and safe to ignore downstream.
            if global_step_count is not None:
                batch.attrs["global_step_count"] = int(global_step_count)
            if epoch is not None:
                batch.attrs["epoch"] = int(epoch)
            if phase is not None:
                batch.attrs["phase"] = str(phase)
            try:
                import time as _time
                batch.attrs["wall_time_s"] = float(_time.time())
            except Exception:
                pass
        except Exception:
            pass

        # 3. Create datasets
        chunk_events = min(n_events, 1024) # Smaller chunks
        chunk_samples = min(S_batch, 32)
        for name in SAMPLE_FIELDS:
            batch.create_dataset(
                name,
                shape=(n_events, S_batch),
                chunks=(chunk_events, chunk_samples),
                dtype='float32',
                compression=params.get('hdf5_compression', None),
                shuffle=True if params.get('hdf5_shuffle', False) else False,
            )

        write_noise = noise_log_scales is not None and len(noise_log_scales) == S_batch
        if write_noise:
            for name in ('log_sigma_p', 'log_sigma_s'):
                batch.create_dataset(name, shape=(S_batch,), dtype='float32')

        # 4. Lon/lat handling
        compute_lonlat = bool(params.get('samples_store_lonlat', True))
        # Re-create Proj locally for thread-safety and stability
        proj = Proj(proj='laea', lat_0=lat0, lon_0=lon0, datum='WGS84', units='km') if compute_lonlat else None

        # 5. Write in chunks
        print(f"[Samples] Writing datasets in chunks of {chunk_events}...")
        X_src_cpu = X_src.detach().cpu().numpy().astype('float32')
        
        for start in range(0, n_events, chunk_events):
            end = min(start + chunk_events, n_events)
            ev_count = end - start
            
            # (ev_count, S_batch, 4)
            dX_chunk = np.stack([s[start:end, :] for s in samples_np], axis=1)
            # (ev_count, 1, 3)
            X_base = X_src_cpu[start:end, :3][:, np.newaxis, :]
            
            # (ev_count, S_batch, 3)
            XX_abs = X_base + dX_chunk[:, :, :3]
            
            if compute_lonlat and proj is not None:
                out_lon = np.empty((ev_count, S_batch), dtype=np.float32)
                out_lat = np.empty((ev_count, S_batch), dtype=np.float32)
                for s in range(S_batch):
                    try:
                        lo, la = proj(XX_abs[:, s, 0], XX_abs[:, s, 1], inverse=True)
                        out_lon[:, s] = lo
                        out_lat[:, s] = la
                    except Exception:
                        out_lon[:, s] = np.nan
                        out_lat[:, s] = np.nan
                batch['longitude'][start:end, :] = out_lon
                batch['latitude'][start:end, :]  = out_lat
            
            batch['depth'][start:end, :]   = XX_abs[:, :, 2]
            batch['delta_t'][start:end, :] = dX_chunk[:, :, 3]
            batch['X'][start:end, :]       = dX_chunk[:, :, 0]
            batch['Y'][start:end, :]       = dX_chunk[:, :, 1]
            batch['Z'][start:end, :]       = dX_chunk[:, :, 2]

        # 6. Noise scales
        if write_noise:
            n_list = []
            for s in noise_log_scales: # type: ignore
                v = s.detach().cpu().numpy().flatten() if isinstance(s, torch.Tensor) else np.asarray(s).flatten()
                n_list.append(v[:2] if v.size >= 2 else np.pad(v, (0, 2-v.size), constant_values=np.nan))
            n_mat = np.stack(n_list)
            batch['log_sigma_p'][:] = n_mat[:, 0]
            batch['log_sigma_s'][:] = n_mat[:, 1]

        batch.attrs['sample_count'] = int(S_batch)
        f.flush()
        f.close()
        print(f"[Samples] Successfully wrote batch {batch_idx}")
        return sample_count
        
    except Exception as e:
        print(f"[Samples] Fatal error during save: {e}")
        import traceback
        traceback.print_exc()
        return sample_count


def read_all_samples(
    params,
    backend: str = 'numpy',
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    pin_memory: bool = False,
    thin: int = 1,
) -> Dict[str, np.ndarray] | Dict[str, torch.Tensor]:
    """Read all samples from HDF5 file with optional thinning.
    
    Args:
        params: Parameter dictionary containing 'samples_outfile'
        backend: 'numpy' or 'torch' for output format
        device: Target device for torch tensors
        dtype: Data type for torch tensors
        pin_memory: Whether to pin memory for torch tensors
        thin: Thinning factor - keep every nth sample (default: 1 = no thinning)
    """
    # Accept either:
    # - path to params.json (str)
    # - nested config dict (contains 'io' but not yet materialized)
    # - already-materialized dict (may include both nested blocks and legacy flat keys)
    if isinstance(params, str):
        try:
            with open(params, "r") as f:
                params = json.load(f)
        except Exception as e:
            raise ValueError(f"read_all_samples: failed to read params from '{params}': {e}")

    # Resolve samples store path from either:
    # - materialized params dict (contains 'samples_outfile'), OR
    # - nested config dict (contains 'io': {'samples_outfile': ...})
    #
    # IMPORTANT: This is a post-processing utility; it should not require the full SPIDER
    # config schema to be present (e.g., model.domain / inference blocks) just to read an HDF5.
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
    
    with h5py.File(store_path, 'r') as f:
        # Optional MAP datasets at root (may exist even if no batch_* groups; e.g. after locate-map / Phase 1 only)
        map_lon_root = f['map_longitude'][:] if 'map_longitude' in f else None
        map_lat_root = f['map_latitude'][:] if 'map_latitude' in f else None
        map_dep_root = f['map_depth'][:] if 'map_depth' in f else None

        # Collect batch groups (stable sort by parsed index).
        # We keep the group name so we can provide boundary metadata to downstream analysis/plotting.
        batches: list[tuple[int, str, Any]] = []
        for name in f.keys():
            if isinstance(f[name], h5py.Group) and name.startswith('batch_'):
                try:
                    num = int(name.split('_')[1])
                    batches.append((num, str(name), f[name]))
                except Exception:
                    pass
        if not batches:
            # Map-only mode: if the file has MAP datasets, fabricate a single-sample "chain" so analysis code works.
            if map_lon_root is not None and map_lat_root is not None and map_dep_root is not None:
                try:
                    ids = json.loads(f.attrs.get('event_ids_json', '[]'))
                    n_events_meta = int(f.attrs.get('n_events', 0))
                    if not ids or (n_events_meta > 0 and len(ids) != n_events_meta):
                        ids = [str(i) for i in range(n_events_meta)]
                    event_ids = np.asarray(ids, dtype=str)
                except Exception:
                    n_events_meta = int(map_lon_root.shape[0])
                    event_ids = np.asarray([str(i) for i in range(n_events_meta)], dtype=str)

                n_events = int(n_events_meta) if int(n_events_meta) > 0 else int(map_lon_root.shape[0])
                # Align arrays
                map_lon = np.asarray(map_lon_root, dtype=np.float32)[:n_events]
                map_lat = np.asarray(map_lat_root, dtype=np.float32)[:n_events]
                map_dep = np.asarray(map_dep_root, dtype=np.float32)[:n_events]
                if event_ids.shape[0] != n_events:
                    event_ids = event_ids[:n_events]

                # Create 2D arrays with a single sample so downstream expects (n_events, n_samples)
                out_map: Dict[str, Any] = {
                    'event_ids': event_ids,
                    'longitude': map_lon.reshape(n_events, 1),
                    'latitude':  map_lat.reshape(n_events, 1),
                    'depth':     map_dep.reshape(n_events, 1),
                    # No timing samples in MAP-only mode; provide zeros for shape consistency
                    'delta_t':   np.zeros((n_events, 1), dtype=np.float32),
                    'X':         np.zeros((n_events, 1), dtype=np.float32),
                    'Y':         np.zeros((n_events, 1), dtype=np.float32),
                    'Z':         np.zeros((n_events, 1), dtype=np.float32),
                    # Preserve MAP arrays as 1D for convenience
                    'map_longitude': map_lon,
                    'map_latitude':  map_lat,
                    'map_depth':     map_dep,
                }
                print("No batch_* groups found; returning MAP-only samples (n_samples=1).")

                if str(backend).lower() == 'numpy':
                    return out_map  # type: ignore[return-value]

                target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                out_torch: Dict[str, torch.Tensor] | Dict[str, np.ndarray] = {'event_ids': out_map['event_ids']}
                for name in SAMPLE_FIELDS:
                    arr = out_map[name]
                    t = torch.from_numpy(arr).to(dtype)
                    if pin_memory and target_device != 'cpu':
                        t = t.pin_memory()
                    out_torch[name] = t.to(target_device, non_blocking=True) if target_device != 'cpu' else t
                for name in ('map_longitude', 'map_latitude', 'map_depth'):
                    arr = out_map[name]
                    t = torch.from_numpy(arr).to(dtype)
                    if pin_memory and target_device != 'cpu':
                        t = t.pin_memory()
                    out_torch[name] = t.to(target_device, non_blocking=True) if target_device != 'cpu' else t
                return out_torch  # type: ignore[return-value]

            print("No batches found in samples store")
            return {}
        batches.sort(key=lambda x: x[0])

        try:
            ids = json.loads(f.attrs.get('event_ids_json', '[]'))
            n_events_meta = int(f.attrs.get('n_events', 0))
            if not ids or len(ids) != n_events_meta:
                ids = [str(i) for i in range(n_events_meta)]
            event_ids = np.asarray(ids, dtype=str)
        except Exception:
            n_events_meta = batches[0][2]['longitude'].shape[0]
            event_ids = np.asarray([str(i) for i in range(n_events_meta)], dtype=str)

        # Some sample files can contain batches with inconsistent event counts (e.g. resume after filtering).
        # IMPORTANT: truncating to min(n_events) can produce *nonsense chains* by mixing incompatible event sets.
        # Default behavior is to keep only batches matching the file-level n_events (typically the latest run).
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
            if len(set(n_events_per_batch)) > 1:
                try:
                    uniq = sorted(set(n_events_per_batch))
                    print(
                        f"Warning: inconsistent event counts across sample batches ({uniq}); "
                        f"reading n_events=min(...)={n_events}. This may mix incompatible event sets."
                    )
                except Exception:
                    pass
        else:
            # skip/error modes use the file-level n_events as the canonical shape.
            n_events = int(canonical_n_events)
            keep: list[tuple[int, Any]] = []
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
                else:
                    print(f"Warning: {msg}")
            batches = keep if keep else batches

        # --- Provenance filter (default: drop legacy batches when modern batches exist) ---
        # Older SPIDER versions wrote batch_* groups without provenance attrs (epoch/global_step_count/wall_time_s).
        # Mixing legacy + modern batches in one concatenated chain often looks like "mode switching" in trace plots.
        try:
            legacy_mode = str(params.get("read_samples_legacy_mode", "drop")).strip().lower()
        except Exception:
            legacy_mode = "drop"
        if legacy_mode not in {"drop", "keep", "error"}:
            legacy_mode = "drop"

        try:
            has_epoch = [("epoch" in grp.attrs) for _, _, grp in batches]
        except Exception:
            has_epoch = []

        if has_epoch and any(has_epoch) and (not all(has_epoch)):
            legacy_batches = [bname for (_, bname, _), ok in zip(batches, has_epoch) if not ok]
            modern_batches = [t for t, ok in zip(batches, has_epoch) if ok]
            msg = (
                f"Detected legacy sample batches missing provenance attrs (no 'epoch'): "
                f"{legacy_batches[:5]}{'...' if len(legacy_batches) > 5 else ''} "
                f"(count={len(legacy_batches)})."
            )
            if legacy_mode == "error":
                raise ValueError(msg + " Set read_samples_legacy_mode='keep' to include them.")
            if legacy_mode == "drop":
                try:
                    print(f"Warning: {msg} Dropping legacy batches (read_samples_legacy_mode='drop').")
                except Exception:
                    pass
                batches = modern_batches
            else:
                try:
                    print(f"Warning: {msg} Keeping legacy batches (read_samples_legacy_mode='keep').")
                except Exception:
                    pass
        # Optional MAP datasets at root
        map_lon = map_lon_root
        map_lat = map_lat_root
        map_dep = map_dep_root
        
        # Calculate total samples after thinning (over the selected batches)
        total_samples = 0
        batch_names: list[str] = []
        batch_sample_counts: list[int] = []
        for _, bname, grp in batches:
            n_samples_in_batch = grp['longitude'].shape[1]
            if thin == 1:
                n_keep = int(n_samples_in_batch)
            else:
                n_keep = int(len(range(0, int(n_samples_in_batch), int(thin))))
            total_samples += int(n_keep)
            batch_names.append(str(bname))
            batch_sample_counts.append(int(n_keep))

        # Helpful warning for a very common footgun:
        # if the samples file contains multiple batch_* groups, the returned arrays are a concatenation.
        # This may represent a single long run with periodic flushes, OR multiple appended runs/resets.
        # Downstream chain plots can look like "mode switching" if batches are not continuous.
        try:
            warn_multi = bool(params.get("warn_on_multi_batch", True))
        except Exception:
            warn_multi = True
        if warn_multi and len(batch_names) > 1:
            try:
                print(
                    f"read_all_samples: concatenating {len(batch_names)} batch_* groups "
                    f"(thin={thin}) from '{store_path}'. "
                    "If you expected a single continuous chain, ensure you did not append multiple runs. "
                    "Consider setting `clear_samples_on_reset=true` or using a fresh `samples_outfile` per run."
                )
            except Exception:
                pass

        # Align event_ids to chosen n_events
        try:
            if int(getattr(event_ids, "shape", [0])[0]) != int(n_events):
                event_ids = event_ids[:n_events]
        except Exception:
            event_ids = np.asarray([str(i) for i in range(n_events)], dtype=str)

        out: Dict[str, Any] = {
            'event_ids': event_ids,
            'longitude': np.empty((n_events, total_samples), dtype=np.float32),
            'latitude':  np.empty((n_events, total_samples), dtype=np.float32),
            'depth':     np.empty((n_events, total_samples), dtype=np.float32),
            'delta_t':   np.empty((n_events, total_samples), dtype=np.float32),
            'X':         np.empty((n_events, total_samples), dtype=np.float32),
            'Y':         np.empty((n_events, total_samples), dtype=np.float32),
            'Z':         np.empty((n_events, total_samples), dtype=np.float32),
        }
        # Provide batch boundary metadata for plotting/debugging (indices refer to the concatenated sample axis).
        # - _batch_names: ordered list of group names (e.g., ["batch_0","batch_1",...])
        # - _batch_sample_counts: number of kept samples per batch after thinning
        # - _batch_boundaries: sample indices where a new batch starts (excluding 0), e.g. [400, 801, ...]
        # - _batch_slices: dict[name -> (start, end)] in the concatenated axis
        try:
            out["_batch_names"] = list(batch_names)
            out["_batch_sample_counts"] = np.asarray(batch_sample_counts, dtype=np.int64)
            csum = np.cumsum(np.asarray(batch_sample_counts, dtype=np.int64))
            out["_batch_boundaries"] = csum[:-1].astype(np.int64, copy=False)
            starts = np.concatenate([np.asarray([0], dtype=np.int64), csum[:-1]]).astype(np.int64, copy=False)
            out["_batch_slices"] = {str(nm): (int(s), int(e)) for nm, s, e in zip(batch_names, starts, csum)}
        except Exception:
            pass
        if map_lon is not None and map_lat is not None and map_dep is not None:
            # MAP arrays are per-event; align to chosen n_events
            out['map_longitude'] = map_lon[:n_events].astype(np.float32, copy=False)
            out['map_latitude']  = map_lat[:n_events].astype(np.float32, copy=False)
            out['map_depth']     = map_dep[:n_events].astype(np.float32, copy=False)
        # Optional noise datasets (per sample)
        have_noise = any(('log_sigma_p' in grp and 'log_sigma_s' in grp) for _, _, grp in batches)
        if have_noise:
            out['log_sigma_p'] = np.empty((total_samples,), dtype=np.float32)
            out['log_sigma_s'] = np.empty((total_samples,), dtype=np.float32)

        offset = 0
        for _, _, grp in batches:
            w = grp['longitude'].shape[1]
            # Apply thinning to this batch
            
            # Use read_direct to avoid intermediate memory allocation
            if thin == 1:
                # No thinning - use all samples
                sl = slice(offset, offset + w)
                n_current = w
                # Explicitly slice rows to match n_events (some batches may have extra events)
                source_sel = np.s_[0:n_events, :]
            else:
                # Apply thinning - keep every nth sample
                n_current = len(range(0, w, thin))
                sl = slice(offset, offset + n_current)
                source_sel = np.s_[0:n_events, 0:w:thin]

            # Define destination selection (all rows, specific columns)
            dest_sel = np.s_[:, sl]

            # Read fields using read_direct
            for name in SAMPLE_FIELDS:
                if name in grp:
                    # h5py read_direct requires the destination to be C-contiguous
                    # out[name] is created as np.empty which is C-contiguous by default
                    grp[name].read_direct(out[name], source_sel=source_sel, dest_sel=dest_sel)
            
            # Handle noise scales separately as they are 1D (per sample)
            if 'log_sigma_p' in grp and 'log_sigma_s' in grp and 'log_sigma_p' in out:
                # 1D arrays
                if thin == 1:
                    src_sel_1d = None
                    dst_sel_1d = sl
                else:
                    src_sel_1d = np.s_[0:w:thin]
                    dst_sel_1d = sl
                
                grp['log_sigma_p'].read_direct(out['log_sigma_p'], source_sel=src_sel_1d, dest_sel=dst_sel_1d)
                grp['log_sigma_s'].read_direct(out['log_sigma_s'], source_sel=src_sel_1d, dest_sel=dst_sel_1d)
            
            offset += n_current

    # Return as-is if numpy backend requested (default)
    if str(backend).lower() == 'numpy':
        return out

    # Torch backend: wrap numeric arrays as tensors; keep 'event_ids' as-is (numpy of str)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_torch: Dict[str, torch.Tensor] | Dict[str, np.ndarray] = {'event_ids': out['event_ids']}
    for name in SAMPLE_FIELDS:
        arr = out[name]
        t = torch.from_numpy(arr).to(dtype)
        if pin_memory and target_device != 'cpu':
            t = t.pin_memory()
        out_torch[name] = t.to(target_device, non_blocking=True) if target_device != 'cpu' else t
    # Optional noise tensors
    if 'log_sigma_p' in out and 'log_sigma_s' in out:
        for name in ('log_sigma_p', 'log_sigma_s'):
            arr = out[name]
            t = torch.from_numpy(arr).to(dtype)
            if pin_memory and target_device != 'cpu':
                t = t.pin_memory()
            out_torch[name] = t.to(target_device, non_blocking=True) if target_device != 'cpu' else t
    # Optional MAP tensors
    for name in ('map_longitude','map_latitude','map_depth'):
        if name in out:
            arr = out[name]
            t = torch.from_numpy(arr).to(dtype)
            if pin_memory and target_device != 'cpu':
                t = t.pin_memory()
            out_torch[name] = t.to(target_device, non_blocking=True) if target_device != 'cpu' else t

    # Preserve non-tensor batch metadata (useful for plotting/debugging).
    for k in ("_batch_names", "_batch_sample_counts", "_batch_boundaries", "_batch_slices"):
        if k in out:
            out_torch[k] = out[k]  # type: ignore[index]

    return out_torch  # type: ignore[return-value]
