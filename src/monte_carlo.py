import numpy as np
import torch
import matplotlib.pyplot as plt

from lqr_solver import LQRSolver

def quadratic_form(x, A):
    """
    Compute x^T A x for batched vectors.

    Parameters
    ----------
    x : np.ndarray, shape (..., 2)
    A : np.ndarray, shape (2, 2)

    Returns
    -------
    np.ndarray, shape (...)
    """
    return np.einsum("...i,ij,...j->...", x, A, x)


def simulate_cost_explicit(solver, t0, x0, N, n_paths, seed=None):
    """
    Monte Carlo estimate of J(t0, x0) under the optimal control.

    Uses the explicit Euler scheme:
        X_{n+1} = X_n + (H X_n + M a_n) dt + sigma dW_n

    Parameters
    ----------
    solver : LQRSolver
    t0 : float
        Initial time
    x0 : np.ndarray, shape (2,)
        Initial state
    N : int
        Number of time steps
    n_paths : int
        Number of Monte Carlo sample paths
    seed : int or None
        Random seed

    Returns
    -------
    float
        Monte Carlo estimate of the cost
    """
    if seed is not None:
        np.random.seed(seed)

    x0 = np.array(x0, dtype=float)
    T = solver.T

    dt = (T - t0) / N
    sqrt_dt = np.sqrt(dt)

    X = np.tile(x0, (n_paths, 1))   # shape (n_paths, 2)
    running_cost = np.zeros(n_paths, dtype=float)

    for n in range(N):
        tn = t0 + n * dt

        t_batch = torch.full((n_paths,), tn, dtype=torch.float32)
        x_batch = torch.tensor(X[:, None, :], dtype=torch.float32)  # (n_paths, 1, 2)

        a = solver.optimal_control(t_batch, x_batch).detach().cpu().numpy()  # (n_paths, 2)

        state_cost = quadratic_form(X, solver.C)
        control_cost = quadratic_form(a, solver.D)
        running_cost += (state_cost + control_cost) * dt

        drift = X @ solver.H.T + a @ solver.M.T
        dW = np.random.randn(n_paths, 2) * sqrt_dt
        diffusion = dW @ solver.sigma.T

        X = X + drift * dt + diffusion

    terminal_cost = quadratic_form(X, solver.R)
    total_cost = running_cost + terminal_cost

    return float(np.mean(total_cost))


def compute_abs_error(solver, t0, x0, N, n_paths, seed=None):
    """
    Error is defined as:
        |J_hat(t0, x0) - v(t0, x0)|

    where J_hat is the Monte Carlo estimate under the optimal control,
    and v is the benchmark value function from Exercise 1.1.
    """
    mc_value = simulate_cost_explicit(
        solver=solver,
        t0=t0,
        x0=x0,
        N=N,
        n_paths=n_paths,
        seed=seed
    )

    true_value = solver.value_function(
        torch.tensor([t0], dtype=torch.float32),
        torch.tensor(x0[None, None, :], dtype=torch.float32)
    ).item()

    return abs(mc_value - true_value)


def build_test_solver():
    """
    Build a simple test solver.
    Replace these matrices later with the actual coursework setting if needed.
    """
    H = np.array([[0.1, 0.0],
                  [0.0, 0.2]])
    M = np.eye(2)
    C = np.eye(2)
    D = np.eye(2)
    R = np.eye(2)
    sigma = 0.3 * np.eye(2)
    T = 1.0

    solver = LQRSolver(H, M, C, D, R, sigma, T)
    time_grid = np.linspace(0.0, T, 2001)
    solver.solve_riccati(time_grid)

    return solver


def basic_mc_test():
    """
    Minimal sanity check for Exercise 1.2.
    """
    solver = build_test_solver()

    t0 = 0.0
    x0 = np.array([1.0, 0.5])

    mc_value = simulate_cost_explicit(
        solver=solver,
        t0=t0,
        x0=x0,
        N=1000,
        n_paths=5000,
        seed=123
    )

    true_value = solver.value_function(
        torch.tensor([t0], dtype=torch.float32),
        torch.tensor(x0[None, None, :], dtype=torch.float32)
    ).item()

    error = abs(mc_value - true_value)

    print("MC estimate =", mc_value)
    print("True value  =", true_value)
    print("Abs error   =", error)


def time_step_convergence_test(save_path="figures/mc_time_convergence.png", show_plot=True):
    """
    Fix a large sample size and vary the number of time steps N.
    Save the resulting log-log convergence plot.
    """
    solver = build_test_solver()

    t0 = 0.0
    x0 = np.array([1.0, 0.5])

    n_paths = 20000
    N_list = [1, 10, 50, 100, 500, 1000, 5000]
    errors = []

    print("\nTime-step convergence test")
    for N in N_list:
        err = compute_abs_error(
            solver=solver,
            t0=t0,
            x0=x0,
            N=N,
            n_paths=n_paths,
            seed=123
        )
        errors.append(err)
        print(f"N = {N:5d}, abs error = {err:.8f}")

    plt.figure(figsize=(6, 4))
    plt.loglog(N_list, errors, marker="o")
    plt.xlabel("Number of time steps N")
    plt.ylabel("Absolute error")
    plt.title("Monte Carlo convergence with respect to time steps")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Saved figure to: {save_path}")


def sample_convergence_test(save_path="figures/mc_sample_convergence.png", show_plot=True):
    """
    Fix a large time discretization and vary the Monte Carlo sample size.
    Save the resulting log-log convergence plot.
    """
    solver = build_test_solver()

    t0 = 0.0
    x0 = np.array([1.0, 0.5])

    N = 5000
    n_paths_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    errors = []

    print("\nSample-size convergence test")
    for n_paths in n_paths_list:
        err = compute_abs_error(
            solver=solver,
            t0=t0,
            x0=x0,
            N=N,
            n_paths=n_paths,
            seed=123
        )
        errors.append(err)
        print(f"n_paths = {n_paths:6d}, abs error = {err:.8f}")

    plt.figure(figsize=(6, 4))
    plt.loglog(n_paths_list, errors, marker="o")
    plt.xlabel("Number of Monte Carlo samples")
    plt.ylabel("Absolute error")
    plt.title("Monte Carlo convergence with respect to sample size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Saved figure to: {save_path}")


def run_all_mc_experiments(show_plots=True):
    """
    Run all Monte Carlo experiments for Exercise 1.2.
    """
    basic_mc_test()
    time_step_convergence_test(
        save_path="figures/mc_time_convergence.png",
        show_plot=show_plots
    )
    sample_convergence_test(
        save_path="figures/mc_sample_convergence.png",
        show_plot=show_plots
    )


if __name__ == "__main__":
    run_all_mc_experiments(show_plots=True)