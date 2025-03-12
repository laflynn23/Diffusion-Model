import numpy as np

def forward_ez(v, a, t):
    """
    Compute predicted summary statistics using the forward EZ equations.
    
    Parameters:
      v : drift rate
      a : boundary separation
      t : nondecision time
      
    Returns:
      R_pred: predicted accuracy
      M_pred: predicted mean RT
      V_pred: predicted RT variance
    """
    y = np.exp(-a * v)
    R_pred = 1 / (1 + y)
    M_pred = t + (a**2 / v) * ((1 - y) / (1 + y))
    V_pred = (a**2 / v**3) * ((1 - 2 * a * v * y - y**2) / (1 + y)**2)
    return R_pred, M_pred, V_pred

def simulate_observed_stats(R_pred, M_pred, V_pred, N):
    """
    Simulate observed summary statistics:
      - R_obs: observed accuracy from a binomial distribution.
      - M_obs: observed mean RT from a normal distribution.
      - V_obs: observed RT variance from a gamma distribution.
      
    Parameters:
      R_pred, M_pred, V_pred: predicted statistics from forward_ez
      N: sample size
      
    Returns:
      R_obs, M_obs, V_obs
    """
    T_obs = np.random.binomial(N, R_pred)
    R_obs = T_obs / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    shape = (N - 1) / 2
    scale = (2 * V_pred) / (N - 1)
    V_obs = np.random.gamma(shape, scale)
    return R_obs, M_obs, V_obs