import torch
from ..core.state import LocateState, _current_noise_scales
from spider.utils.console import info, warn


# Standardized stdout helper
def _log(*parts, section: str = "DIAG", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

@torch.no_grad()
def compute_block_fim(state: LocateState, batch_size: int = 4096, return_sparse: bool = False):
    """
    Compute the Approximate Fisher Information Matrix (FIM) for the full dataset.
    
    The FIM is defined as J^T W J, where:
      J is the Jacobian of the residuals (d_res / d_params)
      W is the weight matrix (inverse covariance of data noise)
    
    Since we optimize for parameters [dX, dY, dZ, dT] for each event, 
    the full Jacobian J has size (N_obs, 4 * N_events).
    
    The resulting FIM J^T W J has size (4*N_events, 4*N_events).
    
    This function computes the FIM by accumulating J_batch^T W_batch J_batch
    batch-by-batch to save memory.
    
    For double-difference relocation, each observation involves exactly two events (i, j).
    So each row of J has non-zero entries only at columns [4i:4i+4] and [4j:4j+4].
    
    Thus, J^T J is block-sparse. The diagonal blocks (4x4) correspond to single-event constraints,
    and off-diagonal blocks (4x4) at (i,j) correspond to the link between event i and j.
    
    Parameters:
        state: LocateState containing model, data, and current parameters.
        batch_size: Batch size for Jacobian computation.
        return_sparse: If True, return (indices, values) for sparse tensor construction
                       or a sparse CSR/COO representation.
                       If False, returns just the diagonal blocks (N_events, 4, 4) 
                       which are useful for "single event stability" checks.
    """
    N_events = state.dX_src.shape[0]
    N_obs = state.N
    
    # We will accumulate diagonal blocks (N_events, 4, 4)
    # FIM_diag[k] = sum_obs ( J_obs[k]^T * w * J_obs[k] )
    fim_diag = torch.zeros(N_events, 4, 4, device=state.device)
    
    # If sparse FIM requested, we need to accumulate off-diagonals too.
    # We can store them in a dictionary mapping (i,j) -> 4x4 block, then convert.
    
    # We need to enable grad for dX_src momentarily to compute Jacobian
    original_requires_grad = state.dX_src.requires_grad
    state.dX_src.requires_grad_(True)
    
    # Current noise scales
    σp, σs = _current_noise_scales(state)
    σp_sq_inv = 1.0 / (σp**2)
    σs_sq_inv = 1.0 / (σs**2)
    
    total_batches = (N_obs + batch_size - 1) // batch_size
    _log(f"Computing FIM (Full Dataset, {N_obs} obs, {N_events} events)...")
    
    # List to collect sparse entries: (row_block_idx, col_block_idx, 4x4 block)
    # Row/Col block indices are event indices.
    # We can convert to actual indices later: [4*r:4*r+4, 4*c:4*c+4]
    sparse_blocks = {} # Key: (ev_i, ev_j), Value: Tensor(4,4)
    
    for i in range(0, N_obs, batch_size):
        end = min(i + batch_size, N_obs)
        bs = end - i
        
        # Get batch indices
        idx_b = state.II[i:end] # (B, 2)
        y_b = state.YY[i:end]   # (B, 5)
        
        # Weights (inverse variance)
        phases = y_b[:, 4]
        weights = torch.where(phases < 0.5, σp_sq_inv, σs_sq_inv) # (B,)
        
        # Compute Jacobian via autograd
        # We need gradients of the network output w.r.t input source coords.
        e1_idx = idx_b[:, 0]
        e2_idx = idx_b[:, 1]
        
        # Forward pass:
        X_rec = y_b[:, 1:4]
        phs = y_b[:, 4:5]
        
        # Prepare inputs with requires_grad=True
        # pos1: (B, 4) -- (x,y,z,t) from X_src + dX_src
        pos1 = state.X_src[e1_idx] + state.dX_src[e1_idx]
        pos2 = state.X_src[e2_idx] + state.dX_src[e2_idx]
        
        # Detach param-based positions and create new leaf variables for batch computation
        # to avoid backpropping through the entire history or full parameter tensor.
        pos1_var = pos1.detach().requires_grad_(True)
        pos2_var = pos2.detach().requires_grad_(True)
        
        # EikoNet forward for e1
        # Explicitly make sure gradients flow from pos1_var
        in1 = torch.cat([pos1_var[:, :3], X_rec, phs], dim=1)
        # Model is single-device in SPIDER (multi-GPU uses multiple processes).
        T1 = state.model(in1).squeeze()
        
        # Compute gradients of sum(T1) w.r.t pos1_var to get Jacobian rows.
        # If autograd says "does not require grad", it usually means either:
        # - output doesn't depend on the input in the graph, OR
        # - gradient tracking was disabled (this function is decorated with @torch.no_grad()).
        if T1.grad_fn is None:
             # Should not happen unless model is in eval mode with no_grad context or similar?
             # We are in @torch.no_grad() decorator for this function!
             # Ah! @torch.no_grad() disables gradient tracking globally!
             # We need to enable grad context for this specific forward pass part.
             pass

        # We must re-enable grad for the forward pass and gradient computation, 
        # even though the parent function is @torch.no_grad().
        with torch.enable_grad():
             # Forward again inside enable_grad
             in1 = torch.cat([pos1_var[:, :3], X_rec, phs], dim=1)
             T1 = model_to_use(in1).squeeze()
             grad1 = torch.autograd.grad(T1.sum(), pos1_var, create_graph=False)[0]
             
             in2 = torch.cat([pos2_var[:, :3], X_rec, phs], dim=1)
             T2 = model_to_use(in2).squeeze()
             grad2 = torch.autograd.grad(T2.sum(), pos2_var, create_graph=False)[0]
        
        # Construct full gradient vectors (Jacobian rows) for e1 and e2
        # J1 = [-dT1/dx, -dT1/dy, -dT1/dz, -1]
        # J2 = [ dT2/dx,  dT2/dy,  dT2/dz, +1]
        
        # grad1, grad2 are (B, 4)
        J1 = torch.cat([-grad1[:, :3], torch.full((bs, 1), -1.0, device=state.device)], dim=1) # (B, 4)
        J2 = torch.cat([grad2[:, :3], torch.full((bs, 1), 1.0, device=state.device)], dim=1) # (B, 4)
        
        # Contribution to FIM: J^T W J.
        
        # Compute weighted outer products
        w_b = weights.view(bs, 1, 1)
        
        # (B, 4, 1) @ (B, 1, 4) -> (B, 4, 4)
        term_ii = torch.bmm(J1.unsqueeze(2), J1.unsqueeze(1)) * w_b
        term_jj = torch.bmm(J2.unsqueeze(2), J2.unsqueeze(1)) * w_b
        term_ij = torch.bmm(J1.unsqueeze(2), J2.unsqueeze(1)) * w_b
        
        # Accumulate diagonals
        fim_diag.index_add_(0, e1_idx, term_ii)
        fim_diag.index_add_(0, e2_idx, term_jj)
        
        if return_sparse:
            # Convert indices to numpy for dictionary operations
            e1_np = e1_idx.cpu().numpy()
            e2_np = e2_idx.cpu().numpy()
            t_ij_cpu = term_ij.cpu() # (B, 4, 4)
            
            for k in range(bs):
                u, v = int(e1_np[k]), int(e2_np[k])
                block = t_ij_cpu[k] # (4,4)
                
                if u == v:
                    continue
                    
                pair_key = (u, v)
                if pair_key not in sparse_blocks:
                    sparse_blocks[pair_key] = block.clone() # Keep on CPU
                else:
                    sparse_blocks[pair_key] += block
                
    # Restore grad state
    state.dX_src.requires_grad_(original_requires_grad)
    
    _log("FIM computation complete.")
    
    return fim_diag, sparse_blocks

def filter_unstable_events(state: LocateState, fim_diag: torch.Tensor, threshold: float = 1e-6) -> int:
    """
    Identify and remove events with FIM eigenvalues smaller than threshold.
    
    Args:
        state: LocateState object (modified in-place)
        fim_diag: (N_events, 4, 4) diagonal blocks of FIM
        threshold: Minimum allowed eigenvalue
        
    Returns:
        Number of events removed.
    """
    _log(f"\n--- Filtering Unstable Events (Min Eigenvalue < {threshold:.1e}) ---")
    
    # 1. Eigendecomposition
    try:
        L, _ = torch.linalg.eigh(fim_diag) # L is (N, 4) ascending
    except Exception as e:
        _log(f"Eigendecomposition failed: {e}")
        return 0
        
    # 2. Identify bad events
    min_eig = L[:, 0] # Smallest eigenvalue
    keep_mask = min_eig >= threshold # (N_events,) bool
    
    n_total = fim_diag.shape[0]
    n_keep = int(keep_mask.sum().item())
    n_drop = n_total - n_keep
    
    if n_drop == 0:
        _log("No unstable events found.")
        return 0
        
    _log(f"Dropping {n_drop} / {n_total} events due to instability.")
    
    # 3. Apply filter to State
    # We need to filter:
    # - state.X_src, state.dX_src
    # - state.origins0 (DataFrame)
    # - state.II (re-index!)
    # - state.dtimes (DataFrame)
    
    # Get indices of events to keep
    keep_indices = torch.nonzero(keep_mask).squeeze(-1) # (n_keep,)
    
    # Create mapping: old_idx -> new_idx
    # Initialize with -1
    old_to_new = torch.full((n_total,), -1, dtype=torch.long, device=state.device)
    # new_indices range from 0 to n_keep-1
    new_indices = torch.arange(n_keep, device=state.device)
    old_to_new[keep_indices] = new_indices
    
    # 4. Filter Observations (II)
    # Both evid1 and evid2 must be in keep set
    idx_old = state.II
    e1_new = old_to_new[idx_old[:, 0]]
    e2_new = old_to_new[idx_old[:, 1]]
    
    # Valid rows: both mapped to >= 0
    valid_rows_mask = (e1_new >= 0) & (e2_new >= 0)
    n_obs_old = state.N
    
    # Update II
    state.II = torch.stack([e1_new[valid_rows_mask], e2_new[valid_rows_mask]], dim=1)
    state.YY = state.YY[valid_rows_mask]
    state.N = int(state.II.shape[0])
    
    _log(f"Observations reduced: {n_obs_old} -> {state.N}")
    
    # 5. Filter Parameters
    state.X_src = state.X_src[keep_indices]
    state.dX_src = torch.nn.Parameter(state.dX_src[keep_indices].detach())
    # Re-attach optimizer? 
    # NOTE: modifying dX_src in-place breaks the optimizer reference.
    # We must re-initialize the optimizer later or patch it.
    # Since this happens at Phase transition, we can expect the pipeline to rebuild/resume.
    # BUT current pipeline reuses optimizer. We must patch param groups.
    
    # 6. Filter DataFrames (for bookkeeping)
    # origins0
    try:
        # Convert keep_mask to cpu numpy boolean
        keep_np = keep_mask.cpu().numpy()
        import polars as pl
        # Filter origins
        state.origins0 = state.origins0.filter(pl.Series(keep_np))
        
        # dtimes: use valid_rows_mask (cpu)
        valid_rows_np = valid_rows_mask.cpu().numpy()
        state.dtimes = state.dtimes.filter(pl.Series(valid_rows_np))
    except Exception as e:
        _log(f"Warning: DataFrame filtering failed: {e}")

    # 6b. Filter Cluster IDs and Counts (for hierarchical prior)
    if state.cluster_ids is not None:
        # Original K (assume state.cluster_counts has correct size K)
        K = state.cluster_counts.shape[0] if state.cluster_counts is not None else int(state.cluster_ids.max().item()) + 1
        
        # Filter IDs
        state.cluster_ids = state.cluster_ids[keep_indices]
        
        # Recompute counts for ALL K clusters (including empty ones)
        # bincount is perfect for this: counts occurrences of 0..max_id
        # We need minlength=K to ensure size is K
        state.cluster_counts = torch.bincount(state.cluster_ids, minlength=K)

    # 6c. Handle Event Precision Matrix (P0)
    # If P0 is (N, 4, 4), we must filter it.
    # If P0 is (K, 4, 4), we leave it alone (it corresponds to clusters).
    if state.event_precision_matrix is not None:
        if state.event_precision_matrix.shape[0] == n_total:
            # It's per-event (e.g. non-hierarchical or flat)
            state.event_precision_matrix = state.event_precision_matrix[keep_indices]
        # Else if shape[0] == K, do nothing.

    # 7. Handle Optimizer State
    # This is the tricky part. We replaced state.dX_src.
    # The optimizer still holds reference to the OLD parameter tensor.
    # We need to update the optimizer's param_groups AND clear its internal state for the old param.
    # Otherwise, checkpoint saving will crash on KeyError or using old keys.
    
    found = False
    old_param = None
    
    # 1. Update Param Groups and find old param
    for group in state.optimizer.param_groups:
        new_params = []
        for p in group['params']:
            if p.shape == (n_total, 4): # Identify the old dX_src by shape/ref
                 old_param = p
                 new_params.append(state.dX_src)
                 found = True
            else:
                 new_params.append(p)
        group['params'] = new_params
    
    # 2. Clean up Optimizer State Dict
    if old_param is not None and old_param in state.optimizer.state:
        # Delete state for the old parameter to prevent checkpoint crash
        del state.optimizer.state[old_param]
        # Initialize state for new parameter? 
        # Ideally yes, but the optimizer will init it on next step() anyway.
        # However, to be safe for checkpointing, we should leave it clean or empty.
        
    if found:
        _log("Optimizer parameters updated and old state cleared.")
    else:
        _log("Warning: Could not link new dX_src to optimizer.")

    return n_drop

@torch.no_grad()
def analyze_fim_stability(fim_diag: torch.Tensor, threshold: float = 1e-6) -> None:
    """
    Analyze the eigenvalues and eigenvectors of the diagonal blocks of the FIM.
    
    Identifies "weak" directions in parameter space (x, y, z, t) where the 
    curvature (eigenvalue) is small, indicating high uncertainty/instability.
    
    Parameters:
        fim_diag: Tensor of shape (N_events, 4, 4)
        threshold: Eigenvalue threshold below which a direction is considered "unstable".
    """
    _log("\n--- FIM Stability Analysis (Diagonal Blocks) ---")
    
    # 1. Eigendecomposition
    # eigh returns eigenvalues in ascending order
    # L: (N, 4), V: (N, 4, 4)
    try:
        L, V = torch.linalg.eigh(fim_diag)
    except Exception as e:
        _log(f"Eigendecomposition failed: {e}")
        return

    # 2. Check for weak directions
    # Smallest eigenvalue is L[:, 0]
    min_eig = L[:, 0]
    
    # Filter events with min_eig < threshold
    unstable_mask = min_eig < threshold
    n_unstable = int(unstable_mask.sum().item())
    
    _log(f"Events with min_eigenvalue < {threshold:.1e}: {n_unstable} / {fim_diag.shape[0]}")
    
    if n_unstable > 0:
        # Get indices of worst offenders
        sorted_indices = torch.argsort(min_eig)
        worst_k = min(10, n_unstable)
        worst_indices = sorted_indices[:worst_k]
        
        _log(f"\nTop {worst_k} most unstable events (weakest direction vector [dx, dy, dz, dt]):")
        
        for idx in worst_indices:
            eig_val = min_eig[idx].item()
            # Corresponding eigenvector (column 0)
            vec = V[idx, :, 0] # (4,)
            
            # Format vector
            v_str = f"[{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}, {vec[3]:.2f}]"
            
            # Interpretation: dominant component
            abs_vec = torch.abs(vec)
            dom_idx = torch.argmax(abs_vec).item()
            labels = ["E-W", "N-S", "Depth", "Time"]
            dom_label = labels[dom_idx]
            
            _log(f"  Event {idx.item()}: λ={eig_val:.3e} | Vec={v_str} (Unstable in {dom_label})")
            
    _log("------------------------------------------------\n")
