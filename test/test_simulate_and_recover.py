import numpy as np
from forward_model import forward_ez, simulate_observed_stats
from inverse_model import inverse_ez
from simulate_and_recover import simulate_and_recover, run_simulation

def test_forward_inverse_consistency():
    true_v, true_a, true_t = 1.0, 1.0, 0.3
    R_pred, M_pred, V_pred = forward_ez(true_v, true_a, true_t)
    R_obs, M_obs, V_obs = R_pred, M_pred, V_pred
    v_est, a_est, t_est = inverse_ez(R_obs, M_obs, V_obs, s=1)
    expected = np.array([true_v, true_a, true_t])
    recovered = np.array([v_est, a_est, t_est])
    assert np.allclose(expected, recovered, atol=1e-3), \
        f"test_forward_inverse_consistency FAILED: Expected {expected} but got {recovered}"

def test_forward_returns_valid_accuracy():
    v, a, t = 1.0, 1.0, 0.3
    R_pred, _, _ = forward_ez(v, a, t)
    assert 0 <= R_pred <= 1, f"test_forward_returns_valid_accuracy FAILED: Predicted accuracy {R_pred} is not between 0 and 1."

def test_fixed_seed_reproducibility():
    np.random.seed(42)
    bias1, sq_error1 = simulate_and_recover(1.0, 1.0, 0.3, N=100)
    np.random.seed(42)
    bias2, sq_error2 = simulate_and_recover(1.0, 1.0, 0.3, N=100)
    assert np.allclose(bias1, bias2, atol=1e-6) and np.allclose(sq_error1, sq_error2, atol=1e-6), \
        "test_fixed_seed_reproducibility FAILED: Simulation results are not reproducible with a fixed seed."

def test_sample_size_effect():
    iterations = 500
    _, sq_error_small = run_simulation(N=10, iterations=iterations)
    _, sq_error_large = run_simulation(N=4000, iterations=iterations)
    assert np.all(sq_error_large < sq_error_small), \
        f"test_sample_size_effect FAILED: For N=10, sq_error={sq_error_small} but for N=4000, sq_error={sq_error_large}"

def integration_test_fixed_params(true_v, true_a, true_t, N, iterations=1000):
    biases = []
    for _ in range(iterations):
        bias, _ = simulate_and_recover(true_v, true_a, true_t, N)
        biases.append(bias)
    mean_bias = np.mean(biases, axis=0)
    return mean_bias

def test_integration_parameter_set_1():
    true_v, true_a, true_t = 1.0, 1.0, 0.3
    mean_bias = integration_test_fixed_params(true_v, true_a, true_t, N=4000, iterations=1000)
    assert np.all(np.abs(mean_bias) < 0.01), \
        f"test_integration_parameter_set_1 FAILED: Average bias {mean_bias} exceeds threshold 0.01."


if __name__ == '__main__':
    tests = [
        test_forward_inverse_consistency,
        test_forward_returns_valid_accuracy,
        test_fixed_seed_reproducibility,
        test_sample_size_effect,
        test_integration_parameter_set_1,
    ]
    
    failed_tests = []
    for test in tests:
        try:
            test()
            print(f"{test.__name__} passed.")
        except AssertionError as e:
            print(e)
            failed_tests.append(test.__name__)
    
    if failed_tests:
        print("The following tests failed:")
        for name in failed_tests:
            print(f" - {name}")
    else:
        print("All tests passed!")
