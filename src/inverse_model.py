import numpy as np

def inverse_ez(R_obs, M_obs, V_obs, s=0.1):
    """
    Recover estimated parameters from observed summary statistics using inverse EZ equations.
    
    Parameters:
      R_obs: observed accuracy
      M_obs: observed mean RT
      V_obs: observed RT variance
      s: scaling parameter (default 0.1)
      
    Returns:
      v_est: estimated drift rate
      a_est: estimated boundary separation
      t_est: estimated nondecision time
    """
    R_obs = np.clip(R_obs, 1e-6, 1 - 1e-6)
    L = np.log(R_obs / (1 - R_obs))
    
    term = (L * (R_obs**2) * ((1 - R_obs)**2)) / V_obs
    v_est = np.sign(R_obs - 0.5) * s * (term ** 0.25)
    
    a_est = (s**2 * L) / v_est
    
    exp_term = np.exp(-v_est * a_est)
    t_est = M_obs - (a_est**2 / v_est) * ((1 - exp_term) / (1 + exp_term))
    
    return v_est, a_est, t_est