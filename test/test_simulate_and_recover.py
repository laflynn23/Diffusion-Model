import numpy as np
from forward_model import forward_ez, simulate_observed_stats
from inverse_model import inverse_ez
from simulate_and_recover import simulate_and_recover, run_simulation

def test_forward_inverse_consistency():
    true_v, true_a, true_t = 1.0, 1.0, 0.3
    R_pred, M_pred, V_pred = forward_ez(true_v, true_a, true_t)
    R_obs, M_obs, V_obs = R_pred, M_pred, V_pred
    v_est, a_est, t_est = inverse_ez(R_obs, M_obs, V_obs, s=1)
    assert np.allclose([true_v, true_a, true_t], [v_est, a_est, t_est], atol=1e-3), \
        f"Expected {[true_v, true_a, true_t]} but got {[v_est, a_est, t_est]}"

def test_forward_returns_valid_accuracy():
    v, a, t = 1.0, 1.0, 0.3
    R_pred, _, _ = forward_ez(v, a, t)
    assert 0 <= R_pred <= 1

def test_fixed_seed_reproducibility():
    np.random.seed(42)
    bias1, sq_error1 = simulate_and_recover(1.0, 1.0, 0.3, N=100)
    np.random.seed(42)
    bias2, sq_error2 = simulate_and_recover(1.0, 1.0, 0.3, N=100)
    assert np.allclose(bias1, bias2, atol=1e-6) and np.allclose(sq_error1, sq_error2, atol=1e-6), \

def test_sample_size_effect():
    iterations = 500
    _, sq_error_small = run_simulation(N=10, iterations=iterations)
    _, sq_error_large = run_simulation(N=4000, iterations=iterations)
    assert np.all(sq_error_large < sq_error_small), \

if __name__ == '__main__':
    test_forward_inverse_consistency()
    test_forward_returns_valid_accuracy()
    test_fixed_seed_reproducibility()
    test_sample_size_effect()
    print("All tests passed!")