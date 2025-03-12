import numpy as np
from forward_model import forward_ez, simulate_observed_stats
from inverse_model import inverse_ez

def simulate_and_recover(true_v, true_a, true_t, N):
    """
    Simulate one iteration:
      - Compute predicted stats using forward equations.
      - Simulate observed stats based on sample size N.
      - Recover parameters using inverse equations.
      - Return bias (true - estimated) and squared error.
    """
    R_pred, M_pred, V_pred = forward_ez(true_v, true_a, true_t)
    R_obs, M_obs, V_obs = simulate_observed_stats(R_pred, M_pred, V_pred, N)
    v_est, a_est, t_est = inverse_ez(R_obs, M_obs, V_obs)
    bias = np.array([true_v - v_est, true_a - a_est, true_t - t_est])
    sq_error = bias**2
    return bias, sq_error

def run_simulation(N, iterations=1000):
    """
    Run simulate-and-recover for a given sample size N over a specified number of iterations.
    
    Returns:
      mean_bias: average bias for [v, a, t] over iterations
      mean_sq_error: average squared error for [v, a, t]
    """
    biases = []
    sq_errors = []
    for _ in range(iterations):

        true_a = np.random.uniform(0.5, 2)
        true_v = np.random.uniform(0.5, 2)
        true_t = np.random.uniform(0.1, 0.5)
        bias, sq_error = simulate_and_recover(true_v, true_a, true_t, N)
        biases.append(bias)
        sq_errors.append(sq_error)
    
    mean_bias = np.mean(biases, axis=0)
    mean_sq_error = np.mean(sq_errors, axis=0)
    return mean_bias, mean_sq_error

def main():
    sample_sizes = [10, 40, 4000]
    iterations = 1000
    for N in sample_sizes:
        mean_bias, mean_sq_error = run_simulation(N, iterations)
        print(f"Sample Size: {N}")
        print("Average Bias [v, a, t]:", mean_bias)
        print("Average Squared Error [v, a, t]:", mean_sq_error)
        print("---------------------------------------------------")

if __name__ == '__main__':
    main()