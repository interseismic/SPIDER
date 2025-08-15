import os
from typing import Dict, Any

import json
import numpy as np
import torch
import h5py
from pyproj import Proj


SAMPLE_FIELDS = ['longitude', 'latitude', 'depth', 'delta_t', 'X', 'Y', 'Z']


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
) -> int:
    if len(samples) == 0:
        return sample_count

    store_path = params.get('samples_outfile', 'samples.h5')
    lat0 = float(params['lat_min'])
    lon0 = float(params['lon_min'])

    samples_cpu = [s.detach().cpu() if isinstance(s, torch.Tensor) else s for s in samples]
    S_batch = len(samples_cpu)
    samples_array = torch.stack(samples_cpu, dim=0).numpy().astype('float32')  # (S, N, 4)
    X_src_cpu = X_src.detach().cpu().numpy().astype('float32')                 # (N, 4)

    try:
        event_ids = [str(evid) for evid in origins0['evid']]
    except Exception:
        event_ids = [str(e) for e in origins0['evid']]

    n_events = X_src_cpu.shape[0]
    f = _open_or_init_store(store_path, n_events, event_ids)

    # Determine next batch index and ensure unique group
    batch_idx = get_next_sample_count(params)
    batch_name = f"batch_{batch_idx}"
    while batch_name in f:
        batch_idx += 1
        batch_name = f"batch_{batch_idx}"
    batch = f.create_group(batch_name)

    # Create per-field arrays for this batch: (n_events, S_batch)
    chunk_events = min(n_events, int(params.get('io_event_chunk', 2048)))
    chunk_samples = min(S_batch, 32)
    for name in SAMPLE_FIELDS:
        batch.create_dataset(
            name,
            shape=(n_events, S_batch),
            chunks=(chunk_events, chunk_samples),
            dtype='float32',
            compression=params.get('hdf5_compression', None),
            compression_opts=params.get('hdf5_compression_opts', None),
            shuffle=True if params.get('hdf5_shuffle', False) else False,
        )

    # Lon/lat handling
    compute_lonlat = bool(params.get('samples_store_lonlat', True))
    proj = Proj(proj='laea', lat_0=lat0, lon_0=lon0, datum='WGS84', units='km') if compute_lonlat else None

    # Compute and write per event chunk
    for start in range(0, n_events, chunk_events):
        end = min(start + chunk_events, n_events)
        X_chunk = X_src_cpu[start:end, :]
        S_chunk = samples_array[:, start:end, :]

        ev_count = end - start
        out_lon = np.empty((ev_count, S_batch), dtype=np.float32)
        out_lat = np.empty((ev_count, S_batch), dtype=np.float32)
        out_dep = np.empty((ev_count, S_batch), dtype=np.float32)
        out_dt  = np.empty((ev_count, S_batch), dtype=np.float32)
        out_X   = np.empty((ev_count, S_batch), dtype=np.float32)
        out_Y   = np.empty((ev_count, S_batch), dtype=np.float32)
        out_Z   = np.empty((ev_count, S_batch), dtype=np.float32)

        for idx, ev in enumerate(range(start, end)):
            base_x = X_chunk[idx, 0]
            base_y = X_chunk[idx, 1]
            base_z = X_chunk[idx, 2]

            dx = S_chunk[:, idx, 0]
            dy = S_chunk[:, idx, 1]
            dz = S_chunk[:, idx, 2]
            dt = S_chunk[:, idx, 3]

            XX = base_x + dx
            YY = base_y + dy
            if compute_lonlat and proj is not None:
                lons, lats = proj(XX, YY, inverse=True)
                out_lon[idx, :] = lons.astype(np.float32, copy=False)
                out_lat[idx, :] = lats.astype(np.float32, copy=False)
            else:
                out_lon[idx, :].fill(np.nan)
                out_lat[idx, :].fill(np.nan)
            out_dep[idx, :] = (base_z + dz).astype(np.float32, copy=False)
            out_dt[idx, :]  = dt.astype(np.float32, copy=False)
            out_X[idx, :]   = dx.astype(np.float32, copy=False)
            out_Y[idx, :]   = dy.astype(np.float32, copy=False)
            out_Z[idx, :]   = dz.astype(np.float32, copy=False)

        write_slice = slice(start, end)
        batch['longitude'][write_slice, :] = out_lon
        batch['latitude'][write_slice, :]  = out_lat
        batch['depth'][write_slice, :]     = out_dep
        batch['delta_t'][write_slice, :]   = out_dt
        batch['X'][write_slice, :]         = out_X
        batch['Y'][write_slice, :]         = out_Y
        batch['Z'][write_slice, :]         = out_Z

    batch.attrs['sample_count'] = int(S_batch)
    f.flush()
    f.close()
    print(f"Wrote batch {batch_idx} with {S_batch} samples to {store_path}")
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
    store_path = params.get('samples_outfile', 'samples.h5')
    if not os.path.exists(store_path):
        print(f"Samples store not found: {store_path}")
        return {}
    
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    
    with h5py.File(store_path, 'r') as f:
        batches = []
        for name in f.keys():
            if isinstance(f[name], h5py.Group) and name.startswith('batch_'):
                try:
                    num = int(name.split('_')[1])
                    batches.append((num, f[name]))
                except Exception:
                    pass
        if not batches:
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
            n_events_meta = batches[0][1]['longitude'].shape[0]
            event_ids = np.asarray([str(i) for i in range(n_events_meta)], dtype=str)

        n_events = batches[0][1]['longitude'].shape[0]
        
        # Calculate total samples after thinning
        total_samples = 0
        for _, grp in batches:
            n_samples_in_batch = grp['longitude'].shape[1]
            if thin == 1:
                total_samples += n_samples_in_batch
            else:
                total_samples += len(range(0, n_samples_in_batch, thin))

        out = {
            'event_ids': event_ids,
            'longitude': np.empty((n_events, total_samples), dtype=np.float32),
            'latitude':  np.empty((n_events, total_samples), dtype=np.float32),
            'depth':     np.empty((n_events, total_samples), dtype=np.float32),
            'delta_t':   np.empty((n_events, total_samples), dtype=np.float32),
            'X':         np.empty((n_events, total_samples), dtype=np.float32),
            'Y':         np.empty((n_events, total_samples), dtype=np.float32),
            'Z':         np.empty((n_events, total_samples), dtype=np.float32),
        }

        offset = 0
        for _, grp in batches:
            w = grp['longitude'].shape[1]
            # Apply thinning to this batch
            if thin == 1:
                # No thinning - use all samples
                sl = slice(offset, offset + w)
                out['longitude'][:, sl] = grp['longitude'][:]
                out['latitude'][:, sl]  = grp['latitude'][:]
                out['depth'][:, sl]     = grp['depth'][:]
                out['delta_t'][:, sl]   = grp['delta_t'][:]
                out['X'][:, sl]         = grp['X'][:]
                out['Y'][:, sl]         = grp['Y'][:]
                out['Z'][:, sl]         = grp['Z'][:]
                offset += w
            else:
                # Apply thinning - keep every nth sample
                thinned_indices = slice(0, w, thin)
                n_thinned = len(range(0, w, thin))
                sl = slice(offset, offset + n_thinned)
                out['longitude'][:, sl] = grp['longitude'][:, thinned_indices]
                out['latitude'][:, sl]  = grp['latitude'][:, thinned_indices]
                out['depth'][:, sl]     = grp['depth'][:, thinned_indices]
                out['delta_t'][:, sl]   = grp['delta_t'][:, thinned_indices]
                out['X'][:, sl]         = grp['X'][:, thinned_indices]
                out['Y'][:, sl]         = grp['Y'][:, thinned_indices]
                out['Z'][:, sl]         = grp['Z'][:, thinned_indices]
                offset += n_thinned

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

    return out_torch  # type: ignore[return-value]
