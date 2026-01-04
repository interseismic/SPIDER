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
