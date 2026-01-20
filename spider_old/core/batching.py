from __future__ import annotations

from typing import Generator, List

import numpy as np
import torch

from spider.utils.console import warn

from .state import LocateState


@torch.no_grad()
def _build_event_to_row_map(state: LocateState) -> None:
    """
    Build a CPU-side mapping from each event index to the list of row indices in II
    where that event appears as either e1 or e2.
    """
    try:
        Ne = int(state.X_src.shape[0])
        if Ne <= 0 or int(state.N) <= 0:
            state._event_to_rows = [np.empty((0,), dtype=np.int64) for _ in range(max(Ne, 1))]
            return
        II_cpu = state.II.detach().cpu().numpy()
        lists: List[list[int]] = [list() for _ in range(Ne)]
        for r in range(int(state.N)):
            a = int(II_cpu[r, 0]); b = int(II_cpu[r, 1])
            if 0 <= a < Ne:
                lists[a].append(r)
            if 0 <= b < Ne:
                lists[b].append(r)
        state._event_to_rows = [
            np.asarray(rows, dtype=np.int64) if len(rows) > 0 else np.empty((0,), dtype=np.int64)
            for rows in lists
        ]
    except Exception as e:
        warn(f"Failed to build event->row map: {e}", section="BATCH")
        state._event_to_rows = None


@torch.no_grad()
def _iter_event_batches(
    state: LocateState,
    *,
    epoch_seed: int,
    max_edges_per_batch: int,
    events_per_batch: int,
) -> Generator[torch.Tensor, None, None]:
    """
    Yield index tensors of rows (edges) for event-centric batches.
    - Randomly permutes events using epoch_seed for reproducibility.
    - For each chunk of events, unions all incident edges (rows).
    - If max_edges_per_batch > 0, randomly subsamples without replacement to that cap.
    """
    if state._event_to_rows is None:
        _build_event_to_row_map(state)
    if state._event_to_rows is None:
        # Fallback: no mapping; yield nothing
        return
    Ne = len(state._event_to_rows)
    if Ne == 0:
        return
    events_per_batch = max(1, int(events_per_batch))
    # Create deterministic permutation per epoch on CPU
    rng = np.random.default_rng(int(epoch_seed) & 0x7fffffff)
    perm = np.arange(Ne, dtype=np.int64)
    rng.shuffle(perm)
    # Iterate in chunks of events
    for i0 in range(0, Ne, events_per_batch):
        chunk = perm[i0:i0 + events_per_batch]
        # Collect rows for these events
        if len(chunk) == 1:
            rows = state._event_to_rows[int(chunk[0])]
            idx_np = rows
        else:
            # concatenate and unique to avoid duplicates for shared edges
            parts = [state._event_to_rows[int(e)] for e in chunk]
            if len(parts) == 0:
                continue
            if len(parts) == 1:
                idx_np = parts[0]
            else:
                idx_np = np.unique(np.concatenate(parts, axis=0))  # sorted ascending
        if idx_np.size == 0:
            continue
        # Optional cap
        if int(max_edges_per_batch) > 0 and idx_np.size > int(max_edges_per_batch):
            sel = rng.choice(idx_np.size, size=int(max_edges_per_batch), replace=False)
            idx_np = idx_np[sel]
        # Convert to device tensor (no guaranteed order required)
        batch_idx = torch.tensor(idx_np, dtype=torch.int64, device=state.II.device)
        yield batch_idx


@torch.no_grad()
def _prepare_owner_buckets(
    state: LocateState,
    *,
    epoch_seed: int,
    events_per_batch: int,
    max_edges_per_batch: int,
    reorder_all: bool,
) -> None:
    """
    Build owner-buckets once per epoch with O(N_edges) preprocessing, producing
    contiguous per-bucket slices for II/YY to avoid per-batch unions/uniques and scattered gathers.
    """
    Ne = int(state.X_src.shape[0])
    N = int(state.N)
    if Ne <= 0 or N <= 0:
        state._bucket_rows_order = None
        state._bucket_offsets = None
        state._bucket_II = None
        state._bucket_YY = None
        state._bucket_p_counts = None
        state._bucket_nodes_p = None
        state._bucket_u_p = None
        state._bucket_v_p = None
        state._bucket_nodes_s = None
        state._bucket_u_s = None
        state._bucket_v_s = None
        state._bucket_chunks_p = None
        state._bucket_chunks_s = None
        return
    events_per_batch = max(1, int(events_per_batch))
    # Permute events and compute owner chunk for each edge
    rng = np.random.default_rng(int(epoch_seed) & 0x7fffffff)
    perm = np.arange(Ne, dtype=np.int64)
    rng.shuffle(perm)
    pos = np.empty(Ne, dtype=np.int64)
    pos[perm] = np.arange(Ne, dtype=np.int64)
    chunk_id = pos // events_per_batch
    num_chunks = int((Ne + events_per_batch - 1) // events_per_batch)
    # Vectorized owner computation
    if state._II_cpu is not None and state._II_cpu.shape[0] == N:
        II_cpu = state._II_cpu
    else:
        II_cpu = state.II.detach().cpu().numpy()
    a_idx = II_cpu[:, 0]
    b_idx = II_cpu[:, 1]
    # Guard bounds
    a_idx = np.clip(a_idx, 0, Ne - 1)
    b_idx = np.clip(b_idx, 0, Ne - 1)
    owner = np.minimum(chunk_id[a_idx], chunk_id[b_idx]).astype(np.int64, copy=False)
    # Stable argsort by owner gives grouped rows
    order = np.argsort(owner, kind="mergesort")
    owner_sorted = owner[order]
    # Counts per chunk
    counts = np.bincount(owner_sorted, minlength=num_chunks)
    # Base offsets per chunk
    offsets = np.zeros((num_chunks + 1,), dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    # Apply maximum edges per bucket by slicing offsets into sub-buckets.
    # Additionally, when reorder_all=True, we "organize" each bucket by sorting rows by phase (P then S),
    # and we store per-bucket p_counts so downstream likelihood can avoid per-batch grouping logic.
    cap = int(max_edges_per_batch) if int(max_edges_per_batch) > 0 else 0
    new_offsets: List[int] = [0]
    final_order_slices: List[np.ndarray] = []
    p_counts: List[int] = []
    phase_cpu = None
    sta_cpu = None
    comp_cpu = None
    if reorder_all:
        try:
            phase_cpu = state.YY[:, 4].detach().cpu().numpy()
        except Exception:
            phase_cpu = None
        try:
            if getattr(state, "row_station_index", None) is not None:
                sta_cpu = state.row_station_index.detach().cpu().numpy()
        except Exception:
            sta_cpu = None
        # Optional: also sort by connected-component id (event cluster id) within each station/phase.
        # This is mainly for correlated likelihoods like slowness_re where per-(station,phase,component)
        # grouping is performance critical. It is opt-in to avoid extra CPU work on very large N.
        sort_by_component = False
        try:
            inf = state.params.get("inference", None)
            bat = inf.get("batching", None) if isinstance(inf, dict) else None
            eb = bat.get("event_batches", None) if isinstance(bat, dict) else None
            if isinstance(eb, dict):
                sort_by_component = bool(eb.get("sort_by_component", False))
        except Exception:
            sort_by_component = False
        if sort_by_component:
            try:
                if getattr(state, "cluster_ids", None) is not None:
                    # Map each row -> component id via e1 index (edges do not cross components).
                    II_cpu = state.II.detach().cpu().numpy()
                    e1 = II_cpu[:, 0].astype(np.int64, copy=False)
                    c_ev = state.cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False)
                    comp_cpu = c_ev[e1]
            except Exception:
                comp_cpu = None

    # If station indices are available, it can be beneficial to order each bucket by (phase, station)
    # for better memory locality in station-dependent latent models (e.g. shared_event_latent).
    sort_station_within_phase = bool(reorder_all and (sta_cpu is not None))
    sort_comp_within_phase = bool(reorder_all and (comp_cpu is not None))

    for cid in range(num_chunks):
        s = int(offsets[cid]); e = int(offsets[cid + 1]); size = e - s
        if size <= 0:
            continue
        rows = order[s:e]
        step = cap if cap > 0 else size
        for i0 in range(0, size, step):
            sl = rows[i0:i0 + step]
            if phase_cpu is not None:
                # Stable partition: P (phase<0.5) then S.
                ph = phase_cpu[sl]
                p_mask = (ph < 0.5)
                p_rows = sl[p_mask]
                s_rows = sl[~p_mask]
                if sort_station_within_phase:
                    # Stable sort within each phase by station id.
                    p_rows = p_rows[np.argsort(sta_cpu[p_rows], kind="mergesort")]
                    s_rows = s_rows[np.argsort(sta_cpu[s_rows], kind="mergesort")]
                if sort_comp_within_phase:
                    # Sort within each phase by (station, component). This maximizes contiguity of
                    # (station,phase,component) groups for correlated likelihoods.
                    try:
                        if sta_cpu is not None:
                            p_rows = p_rows[np.lexsort((comp_cpu[p_rows], sta_cpu[p_rows]))]
                            s_rows = s_rows[np.lexsort((comp_cpu[s_rows], sta_cpu[s_rows]))]
                        else:
                            # Phase-only: group by component if no station ids.
                            p_rows = p_rows[np.argsort(comp_cpu[p_rows], kind="mergesort")]
                            s_rows = s_rows[np.argsort(comp_cpu[s_rows], kind="mergesort")]
                    except Exception:
                        pass
                sl = np.concatenate([p_rows, s_rows], axis=0)
                p_counts.append(int(p_rows.size))
            else:
                p_counts.append(-1)
            final_order_slices.append(sl)
            new_offsets.append(new_offsets[-1] + int(sl.size))

    if not final_order_slices:
        state._bucket_rows_order = None
        state._bucket_offsets = None
        state._bucket_II = None
        state._bucket_YY = None
        state._bucket_p_counts = None
        state._bucket_nodes_p = None
        state._bucket_u_p = None
        state._bucket_v_p = None
        state._bucket_nodes_s = None
        state._bucket_u_s = None
        state._bucket_v_s = None
        state._bucket_chunks_p = None
        state._bucket_chunks_s = None
        return

    rows_order = np.concatenate(final_order_slices, axis=0)
    offsets = np.asarray(new_offsets, dtype=np.int64)
    if rows_order.size == 0:
        state._bucket_rows_order = None
        state._bucket_offsets = None
        state._bucket_II = None
        state._bucket_YY = None
        state._bucket_p_counts = None
        state._bucket_nodes_p = None
        state._bucket_u_p = None
        state._bucket_v_p = None
        state._bucket_nodes_s = None
        state._bucket_u_s = None
        state._bucket_v_s = None
        state._bucket_chunks_p = None
        state._bucket_chunks_s = None
        return
    device = state.II.device
    rows_t = torch.tensor(rows_order, dtype=torch.int64, device=device)
    state._bucket_rows_order = rows_t
    state._bucket_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
    if reorder_all:
        state._bucket_II = state.II.index_select(0, rows_t).contiguous()
        state._bucket_YY = state.YY.index_select(0, rows_t).contiguous()
        try:
            if state.row_station_index is not None:
                state._bucket_station_index = state.row_station_index.index_select(0, rows_t).contiguous()
            else:
                state._bucket_station_index = None
        except Exception:
            state._bucket_station_index = None
        # Also stash per-row component id (connected component label), aligned with bucket order.
        try:
            if getattr(state, "cluster_ids", None) is not None and state._bucket_II is not None:
                e1_b = state._bucket_II[:, 0].to(dtype=torch.int64)
                state._bucket_comp_index = state.cluster_ids.index_select(0, e1_b).contiguous()
            else:
                state._bucket_comp_index = None
        except Exception:
            state._bucket_comp_index = None
    else:
        state._bucket_II = None
        state._bucket_YY = None
        state._bucket_station_index = None
        state._bucket_comp_index = None
    # Per-bucket number of P rows (only meaningful when reorder_all=True and phase_cpu was available).
    try:
        if reorder_all and p_counts and (p_counts[0] >= 0):
            state._bucket_p_counts = torch.tensor(p_counts, dtype=torch.int64, device=device)
        else:
            state._bucket_p_counts = None
    except Exception:
        state._bucket_p_counts = None

    # Precompute per-bucket phase graphs (nodes + (u,v) in local node indexing) once.
    # This makes the correlated likelihood deterministic and avoids per-batch torch.unique.
    try:
        if reorder_all and (state._bucket_II is not None) and (state._bucket_offsets is not None) and (state._bucket_p_counts is not None):
            nb = int(state._bucket_offsets.numel() - 1)
            nodes_p: list = []
            u_p: list = []
            v_p: list = []
            nodes_s: list = []
            u_s: list = []
            v_s: list = []
            for bi in range(nb):
                i0 = int(state._bucket_offsets[bi].item())
                i1 = int(state._bucket_offsets[bi + 1].item())
                p = int(state._bucket_p_counts[bi].item())
                p = max(0, min(p, max(i1 - i0, 0)))
                # P block
                mP = int(p)
                if mP > 0:
                    IIp = state._bucket_II[i0:i0 + mP, :]
                    ev_flat = IIp.reshape(-1)
                    nodes, inv_nodes = torch.unique(ev_flat, return_inverse=True)
                    u = inv_nodes[:mP]
                    v = inv_nodes[mP:]
                else:
                    nodes = torch.empty((0,), dtype=torch.int64, device=device)
                    u = torch.empty((0,), dtype=torch.int64, device=device)
                    v = torch.empty((0,), dtype=torch.int64, device=device)
                nodes_p.append(nodes)
                u_p.append(u)
                v_p.append(v)
                # S block
                mS = int(i1 - (i0 + mP))
                if mS > 0:
                    IIs = state._bucket_II[i0 + mP:i1, :]
                    ev_flat = IIs.reshape(-1)
                    nodes, inv_nodes = torch.unique(ev_flat, return_inverse=True)
                    u = inv_nodes[:mS]
                    v = inv_nodes[mS:]
                else:
                    nodes = torch.empty((0,), dtype=torch.int64, device=device)
                    u = torch.empty((0,), dtype=torch.int64, device=device)
                    v = torch.empty((0,), dtype=torch.int64, device=device)
                nodes_s.append(nodes)
                u_s.append(u)
                v_s.append(v)
            state._bucket_nodes_p = nodes_p
            state._bucket_u_p = u_p
            state._bucket_v_p = v_p
            state._bucket_nodes_s = nodes_s
            state._bucket_u_s = u_s
            state._bucket_v_s = v_s
        else:
            state._bucket_nodes_p = None
            state._bucket_u_p = None
            state._bucket_v_p = None
            state._bucket_nodes_s = None
            state._bucket_u_s = None
            state._bucket_v_s = None
    except Exception:
        state._bucket_nodes_p = None
        state._bucket_u_p = None
        state._bucket_v_p = None
        state._bucket_nodes_s = None
        state._bucket_u_s = None
        state._bucket_v_s = None

    # Residual-correlation chunking removed (no backward compatibility).
    state._bucket_chunks_p = None
    state._bucket_chunks_s = None
    state._bucket_last_epoch = int(epoch_seed)


@torch.no_grad()
def _ensure_owner_buckets(
    state: LocateState,
    *,
    epoch_index: int,
    events_per_batch: int,
    max_edges_per_batch: int,
    reorder_all: bool,
    reuse_epochs: int,
) -> None:
    """
    Rebuild owner buckets only when needed:
      - if never built, or
      - if reuse_epochs <= 1, rebuild every epoch, or
      - if (epoch_index - _bucket_last_epoch) >= reuse_epochs
    """
    reuse_epochs = max(1, int(reuse_epochs))
    need_rebuild = False
    if state._bucket_last_epoch is None:
        need_rebuild = True  # never built → build once
    else:
        delta = int(epoch_index) - int(state._bucket_last_epoch)
        # Negative deltas shouldn't happen; rebuild defensively.
        if delta < 0:
            need_rebuild = True
        elif reuse_epochs <= 1:
            need_rebuild = True
        elif delta >= int(reuse_epochs):
            need_rebuild = True
    if need_rebuild:
        _prepare_owner_buckets(
            state,
            epoch_seed=epoch_index,
            events_per_batch=events_per_batch,
            max_edges_per_batch=max_edges_per_batch,
            reorder_all=reorder_all,
        )
        # Optional debug print (off by default to keep console output clean).
        # Enable by setting `runtime.log_owner_bucket_rebuild=true` in the JSON (extra keys are ignored by the schema).
        if bool(state.params.get("log_owner_bucket_rebuild", False)) or bool(state.params.get("_log_owner_bucket_rebuild", False)):
            try:
                nb = int(state._bucket_offsets.numel() - 1) if state._bucket_offsets is not None else 0
                Nrows = int(state.N)
                print(
                    f"Owner buckets rebuilt (epoch={epoch_index}): num_buckets={nb}, rows={Nrows}, "
                    f"events_per_batch={events_per_batch}, max_edges_per_batch={max_edges_per_batch}"
                )
            except Exception:
                pass


