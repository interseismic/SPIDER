from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DisjointPartition:
    """
    Disjoint partition of nodes into blocks.
    - block_ids: per-node block id in [0, n_blocks)
    - blocks: list of np.ndarray node indices for each block
    """

    block_ids: np.ndarray
    blocks: List[np.ndarray]


def partition_graph_greedy_unionfind(
    *,
    n_nodes: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    max_cluster_size: int,
) -> DisjointPartition:
    """
    Fast disjoint clustering with a size cap using a greedy, weight-descending union-find.

    This approximately keeps heavy edges inside clusters:
      - sort edges by weight descending
      - union endpoints if their clusters can be merged without exceeding max_cluster_size
      - stop early once we hit the theoretical minimum number of clusters ceil(n_nodes / max_cluster_size)
    """
    if n_nodes <= 0:
        return DisjointPartition(block_ids=np.zeros((0,), dtype=np.int64), blocks=[])
    if max_cluster_size < 1:
        raise ValueError(f"max_cluster_size must be >= 1, got {max_cluster_size}")

    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("u, v, w must have the same shape")

    parent = np.arange(int(n_nodes), dtype=np.int64)
    size = np.ones((int(n_nodes),), dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        sa = int(size[ra])
        sb = int(size[rb])
        if sa + sb > max_cluster_size:
            return False
        if sa < sb:
            ra, rb = rb, ra
            sa, sb = sb, sa
        parent[rb] = ra
        size[ra] = np.int32(sa + sb)
        return True

    if u.size > 0:
        order = np.argsort(w)[::-1]  # descending
        target_clusters = int((int(n_nodes) + int(max_cluster_size) - 1) // int(max_cluster_size))
        n_clusters = int(n_nodes)
        for ei in order.tolist():
            if n_clusters <= target_clusters:
                break
            if union(int(u[ei]), int(v[ei])):
                n_clusters -= 1

    roots = np.fromiter((find(i) for i in range(int(n_nodes))), dtype=np.int64, count=int(n_nodes))
    uniq, inv = np.unique(roots, return_inverse=True)
    blocks: List[np.ndarray] = []
    for k in range(int(uniq.size)):
        blocks.append(np.nonzero(inv == k)[0].astype(np.int64, copy=False))
    block_ids = inv.astype(np.int64, copy=False)
    return DisjointPartition(block_ids=block_ids, blocks=blocks)


def _balanced_cut_index(n: int, *, min_balance: float) -> int:
    """
    Pick a cut index in [1, n-1] respecting a minimum balance ratio.
    """
    if n <= 1:
        return 0
    lo = int(np.ceil(float(min_balance) * n))
    hi = int(np.floor((1.0 - float(min_balance)) * n))
    lo = max(1, lo)
    hi = min(n - 1, hi)
    if lo > hi:
        # fall back: closest to half
        return max(1, min(n - 1, n // 2))
    return max(lo, min(hi, n // 2))


def partition_graph_recursive_bisection(
    *,
    n_nodes: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    max_cluster_size: int,
    min_balance: float = 0.30,
    eig_tol: float = 1e-6,
    eig_maxiter: int = 20_000,
    seed: int = 0,
) -> DisjointPartition:
    """
    Partition an undirected weighted graph into disjoint blocks of size <= max_cluster_size
    using spectral recursive bisection (Fiedler vector of the unnormalized Laplacian).

    The objective is to approximately minimize weighted edge cut while keeping splits balanced.

    Args:
        n_nodes: number of nodes.
        u, v: edge endpoints (int arrays) of the UNIQUE undirected edges.
        w: edge weights (float/int array), typically pair_count.
        max_cluster_size: maximum allowed block size (>=1).
        min_balance: minimum fraction on each side of a bisection (0<min_balance<0.5).
        eig_tol/eig_maxiter: eigensolver controls for scipy.sparse.linalg.eigsh.
        seed: used only for deterministic fallbacks when the eigensolver fails.
    """
    if n_nodes <= 0:
        return DisjointPartition(block_ids=np.zeros((0,), dtype=np.int64), blocks=[])
    if max_cluster_size < 1:
        raise ValueError(f"max_cluster_size must be >= 1, got {max_cluster_size}")

    # Local import so SPIDER can still import without scipy in limited contexts.
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh

    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("u, v, w must have the same shape")

    # Build symmetric weighted adjacency
    if u.size == 0:
        # all isolates
        blocks = [np.array([i], dtype=np.int64) for i in range(int(n_nodes))]
        block_ids = np.arange(int(n_nodes), dtype=np.int64)
        return DisjointPartition(block_ids=block_ids, blocks=blocks)

    uu = np.concatenate([u, v])
    vv = np.concatenate([v, u])
    ww = np.concatenate([w, w])
    W = coo_matrix((ww, (uu, vv)), shape=(int(n_nodes), int(n_nodes))).tocsr()

    # Component detection (unweighted structure implied by non-zeros)
    n_comp, comp_labels = connected_components(W, directed=False, return_labels=True)
    comp_labels = np.asarray(comp_labels, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    blocks: List[np.ndarray] = []

    def recurse(nodes: np.ndarray) -> None:
        nodes = np.asarray(nodes, dtype=np.int64)
        n = int(nodes.size)
        if n <= max_cluster_size:
            blocks.append(nodes)
            return

        # Induced subgraph
        subW = W[nodes][:, nodes].tocsr()
        if subW.nnz == 0:
            # No internal edges: arbitrary split
            order = nodes.copy()
            rng.shuffle(order)
            cut = _balanced_cut_index(n, min_balance=min_balance)
            recurse(order[:cut])
            recurse(order[cut:])
            return

        # Laplacian L = D - W
        deg = np.asarray(subW.sum(axis=1)).reshape(-1)
        L = diags(deg, 0, format="csr") - subW

        # Fiedler vector (2nd smallest eigenvector)
        try:
            vals, vecs = eigsh(L, k=2, which="SM", tol=float(eig_tol), maxiter=int(eig_maxiter))
            order = np.argsort(vals)
            # vecs columns correspond to vals order returned; reorder explicitly
            fiedler = np.asarray(vecs[:, order[1]], dtype=np.float64)
            if not np.all(np.isfinite(fiedler)):
                raise FloatingPointError("non-finite fiedler vector")
        except Exception:
            # Fallback: degree-based cut (still tries to keep heavy hubs together loosely)
            fiedler = np.asarray(deg, dtype=np.float64)

        sort_idx = np.argsort(fiedler)
        cut = _balanced_cut_index(n, min_balance=min_balance)
        A = nodes[sort_idx[:cut]]
        B = nodes[sort_idx[cut:]]

        if A.size == 0 or B.size == 0:
            # Ultimate fallback: split by order
            cut = max(1, min(n - 1, n // 2))
            A = nodes[:cut]
            B = nodes[cut:]

        recurse(A)
        recurse(B)

    # Partition each connected component independently
    for c in range(int(n_comp)):
        comp_nodes = np.nonzero(comp_labels == c)[0].astype(np.int64, copy=False)
        if comp_nodes.size == 0:
            continue
        recurse(comp_nodes)

    # Build per-node ids
    block_ids = np.full((int(n_nodes),), -1, dtype=np.int64)
    for bid, nd in enumerate(blocks):
        block_ids[nd] = int(bid)
    # Any missing (shouldn't happen): assign singletons
    missing = np.nonzero(block_ids < 0)[0]
    for i in missing.tolist():
        block_ids[i] = int(len(blocks))
        blocks.append(np.array([i], dtype=np.int64))

    return DisjointPartition(block_ids=block_ids, blocks=blocks)


def partition_graph_disjoint_blocks(
    *,
    n_nodes: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    max_cluster_size: int,
    method: str = "auto",
    min_balance: float = 0.30,
    seed: int = 0,
) -> DisjointPartition:
    """
    Convenience wrapper selecting a partition method.
    - auto: greedy for larger graphs, spectral recursive bisection for smaller.
    - greedy: greedy union-find (fast)
    - spectral: recursive bisection (slow on large components)
    """
    m = str(method).strip().lower()
    if m not in {"auto", "greedy", "spectral"}:
        raise ValueError(f"Unknown method={method!r}; expected 'auto','greedy','spectral'")

    if m == "greedy":
        return partition_graph_greedy_unionfind(
            n_nodes=n_nodes, u=u, v=v, w=w, max_cluster_size=max_cluster_size
        )
    if m == "spectral":
        return partition_graph_recursive_bisection(
            n_nodes=n_nodes,
            u=u,
            v=v,
            w=w,
            max_cluster_size=max_cluster_size,
            min_balance=min_balance,
            seed=seed,
        )

    # auto
    n_edges = int(np.asarray(u).size)
    if (int(n_nodes) > 5000) or (n_edges > 200_000):
        return partition_graph_greedy_unionfind(
            n_nodes=n_nodes, u=u, v=v, w=w, max_cluster_size=max_cluster_size
        )
    return partition_graph_recursive_bisection(
        n_nodes=n_nodes,
        u=u,
        v=v,
        w=w,
        max_cluster_size=max_cluster_size,
        min_balance=min_balance,
        seed=seed,
    )


