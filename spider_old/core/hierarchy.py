import torch
import torch.distributions as dist
import warnings
from typing import Optional

def sample_wishart_posterior(
    scatter_matrix: torch.Tensor,
    n_events: int,
    prior_dof: float,
    prior_scale_inv: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Sample from the conditional posterior of the precision matrix P0 given the data (events).
    
    Model:
      P0 ~ Wishart(nu, V)  [Hyperprior on precision]
      dX_i ~ N(0, P0^-1)   [Likelihood of events]
      
    Posterior:
      P0 | dX ~ Wishart(nu + N, (V^-1 + S)^-1)
      where S = sum(dX_i dX_i^T) is the scatter matrix.
      
    Args:
        scatter_matrix: (p, p) tensor S (scatter / sum of outer products)
        n_events: N (number of observations contributing to the scatter)
        prior_dof: nu (degrees of freedom of hyperprior, must satisfy nu > p-1)
        prior_scale_inv: V^-1 (inverse of prior Wishart scale matrix)
        
    Returns:
        Sampled precision matrix (p, p), or None on failure.
    """
    p = int(scatter_matrix.shape[0])
    # Posterior parameters
    post_dof = prior_dof + n_events
    
    # The scale matrix for the posterior Wishart is (V^-1 + S)^-1
    # We construct the inverse scale matrix explicitly: V_inv + S
    # Use float64 for stability during inversion and sampling
    prior_scale_inv_64 = prior_scale_inv.to(torch.float64)
    scatter_matrix_64 = scatter_matrix.to(torch.float64)
    post_scale_inv_64 = prior_scale_inv_64 + scatter_matrix_64
    
    try:
        # We invert (V^-1 + S) to get the Wishart scale matrix V'
        post_scale_64 = torch.linalg.inv(post_scale_inv_64)
        
        # Ensure symmetry/PSD numerically
        post_scale_64 = 0.5 * (post_scale_64 + post_scale_64.T)
        
        # Add jitter for stability (small enough not to bias the mean significantly for large N)
        # With float64, 1e-10 is usually sufficient to ensure PD without affecting physics
        jitter = 1e-10 * torch.eye(p, device=post_scale_64.device, dtype=torch.float64)
        post_scale_64 = post_scale_64 + jitter
        
        # Sample
        # PyTorch Wishart takes 'covariance_matrix' as the scale matrix (aka V).
        # Suppress "Singular sample detected" warnings which can occur harmlessly with large dof/small scales.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Singular sample detected")
                m = dist.Wishart(df=post_dof, covariance_matrix=post_scale_64)
                P0_sample_64 = m.sample()
        except Exception as e:
            # Fallback: Bartlett decomposition if torch.distributions.Wishart is unavailable/unstable.
            # A ~ Wishart(df, V) can be sampled by:
            #   A = L @ T @ T^T @ L^T, where L=chol(V), and T lower-tri with
            #   T_ii = sqrt(Chi2(df - i)), T_ij ~ N(0,1) for j<i.
            try:
                L = torch.linalg.cholesky(post_scale_64)
                T = torch.zeros((p, p), device=post_scale_64.device, dtype=torch.float64)
                for i in range(p):
                    dof_i = max(float(post_dof - i), 1.0)
                    T[i, i] = torch.sqrt(dist.Chi2(dof_i).sample().to(torch.float64))
                    if i > 0:
                        T[i, :i] = dist.Normal(0.0, 1.0).sample((i,)).to(torch.float64)
                A = L @ T @ T.T @ L.T
                P0_sample_64 = 0.5 * (A + A.T)
            except Exception:
                raise e
        
        return P0_sample_64.to(prior_scale_inv.dtype)
    except Exception as e:
        print(f"Warning: Wishart sampling failed ({e}), keeping current precision.")
        return None

def update_precision_hyperparameter(
    dX_src: torch.Tensor,
    prior_dof: float,
    prior_scale_inv: torch.Tensor,
    mode: str = "sample"
) -> torch.Tensor:
    """
    Update the global event precision matrix P0 based on current event locations.
    
    Args:
        dX_src: (N, 4) event perturbations
        prior_dof: nu
        prior_scale_inv: V^-1 (approx prior_std^-2)
        mode: "sample" (Gibbs) or "map" (Optimization)
    """
    # 1. Compute Scatter Matrix S = sum(x x^T)
    S = dX_src.T @ dX_src
    N = dX_src.shape[0]
    
    if mode == "map":
        # MAP estimate of Wishart posterior
        # Mode of Wishart(n, V) is (n - p - 1) * V for n >= p + 1
        # Here n = nu + N, V = (V_inv + S)^-1
        # P_map = (nu + N - 4 - 1) * (V_inv + S)^-1
        # Actually, for the precision matrix in this conjugate setup:
        # The posterior is W(nu', V').
        # If we want the MAP of P0:
        df = prior_dof + N
        p = int(dX_src.shape[1])
        scalar = max(0.0, df - p - 1)
        
        inv_term = prior_scale_inv + S
        term_inv = torch.linalg.inv(inv_term)
        
        P0_map = scalar * term_inv
        return P0_map
        
    else:
        # Gibbs Sample
        sample = sample_wishart_posterior(S, N, prior_dof, prior_scale_inv)
        if sample is not None:
            return sample
        else:
            # Fallback
            return torch.eye(4, device=dX_src.device)

def update_corr_error_tau_hyperparameter(
    b: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    q_diag: float,
    prior_dof: float,
    prior_scale_inv: torch.Tensor,
    mode: str = "sample"
) -> torch.Tensor:
    """
    Update the P/S covariance matrix for corr_error latents using a Gibbs update.
    
    Args:
        b: (Ne, R, 2) latent tensor
        u, v, w: event graph edge tensors
        q_diag: diagonal jitter for precision Q
        prior_dof: nu_0 for Inverse-Wishart hyperprior
        prior_scale_inv: V_0 for Inverse-Wishart hyperprior (2x2 matrix)
        mode: "sample" or "map"
    """
    Ne = int(b.shape[0])
    R = int(b.shape[1])
    dev = b.device
    
    # 1. Compute components of the 2x2 scatter matrix Sb
    # Sb[i,j] = sum_r (b[:,r,i]^T Q b[:,r,j])
    # To prevent DC runaway, we center b before computing the scatter matrix.
    # This ensures tau represents spatial variation, not global offsets.
    bP = b[:, :, 0]
    bS = b[:, :, 1]
    
    bP = bP - bP.mean(dim=0, keepdim=True)
    bS = bS - bS.mean(dim=0, keepdim=True)
    
    u_i = u.to(torch.int64)
    v_i = v.to(torch.int64)
    w_f = w.to(dtype=b.dtype, device=dev)

    def _apply_Q(xNR: torch.Tensor) -> torch.Tensor:
        y = xNR * float(max(0.0, q_diag))
        if int(u_i.numel()) > 0:
            xu = xNR.index_select(0, u_i)
            xv = xNR.index_select(0, v_i)
            diff = xu - xv
            dw = diff * w_f.unsqueeze(1)
            y.index_add_(0, u_i, dw)
            y.index_add_(0, v_i, -dw)
        return y

    qP = _apply_Q(bP)
    qS = _apply_Q(bS)
    
    e00 = (bP * qP).sum()
    e11 = (bS * qS).sum()
    e01 = (bP * qS).sum()
    
    Sb = torch.tensor([[e00, e01], [e01, e11]], device=dev, dtype=torch.float32)
    
    # Total degrees of freedom contributed by the latent field
    # Since Q is full rank (due to q_diag), each dimension r adds Ne degrees of freedom.
    N_eff = R * Ne
    
    if mode == "map":
        df = prior_dof + N_eff
        inv_term = prior_scale_inv + Sb
        # Mode of Wishart distribution for the precision matrix
        p = 2
        scalar = max(0.0, df - p - 1)
        # If df is too small, the mode is ill-defined; fall back to a prior-scale covariance.
        if not (scalar > 0.0):
            # For our chosen parameterization where prior_scale_inv = nu * diag(scale_std^2),
            # this equals diag(scale_std^2) (a reasonable prior covariance baseline).
            return (prior_scale_inv / max(prior_dof, 1e-12)).to(torch.float32)
        try:
            P_tau_map = scalar * torch.linalg.inv(inv_term)
            cov = torch.linalg.inv(P_tau_map.to(torch.float64))
            cov = 0.5 * (cov + cov.T)
            return cov.to(torch.float32)
        except Exception:
            return (prior_scale_inv / max(prior_dof, 1e-12)).to(torch.float32)
    else:
        # Sample precision matrix from Wishart posterior
        P_tau_sample = sample_wishart_posterior(Sb, N_eff, prior_dof, prior_scale_inv)
        if P_tau_sample is not None:
            # Return covariance matrix (tau^2 and rho*tau1*tau2)
            try:
                cov = torch.linalg.inv(P_tau_sample.to(torch.float64))
                cov = 0.5 * (cov + cov.T)
                return cov.to(torch.float32)
            except Exception:
                return (prior_scale_inv / max(prior_dof, 1e-12)).to(torch.float32)
        else:
            # Prior fallback (covariance baseline). See note above.
            return (prior_scale_inv / max(prior_dof, 1e-12)).to(torch.float32)
