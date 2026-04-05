import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lqr_solver import LQRSolver


class NetDGM(nn.Module):
    """
    DGM-style network for PDE approximation.
    Input: (t, x1, x2)
    Output: scalar u(t, x)
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.Tanh()

    def forward(self, t, x):
        z = torch.cat([t, x], dim=1)
        h = self.activation(self.input_layer(z))
        h = self.activation(self.hidden1(h))
        h = self.activation(self.hidden2(h))
        h = self.activation(self.hidden3(h))
        out = self.output_layer(h)
        return out


def build_test_solver():
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


def quadratic_form_torch(x, A):
    """
    x: (batch, 2)
    A: (2, 2)
    return: (batch, 1)
    """
    Ax = torch.matmul(x, A.T)
    return torch.sum(x * Ax, dim=1, keepdim=True)


def sample_interior(batch_size, T):
    """
    Sample interior points (t, x) for PDE residual.
    t in [0, T), x in [-3,3]^2
    """
    t = torch.rand(batch_size, 1, dtype=torch.float32) * T
    x = -3.0 + 6.0 * torch.rand(batch_size, 2, dtype=torch.float32)

    t.requires_grad_(True)
    x.requires_grad_(True)

    return t, x


def sample_terminal(batch_size, T):
    """
    Sample terminal points (T, x) for boundary condition.
    """
    t = torch.full((batch_size, 1), T, dtype=torch.float32)
    x = -3.0 + 6.0 * torch.rand(batch_size, 2, dtype=torch.float32)

    return t, x


def pde_residual(model, solver, t, x):
    """
    Compute PDE residual for:
    u_t + 1/2 tr(sigma sigma^T u_xx)
        + (u_x)^T Hx + (u_x)^T M alpha
        + x^T C x + alpha^T D alpha = 0
    with alpha = (1,1)^T
    """
    alpha = torch.tensor([[1.0, 1.0]], dtype=torch.float32).repeat(x.shape[0], 1)

    H = torch.tensor(solver.H, dtype=torch.float32)
    M = torch.tensor(solver.M, dtype=torch.float32)
    C = torch.tensor(solver.C, dtype=torch.float32)
    D = torch.tensor(solver.D, dtype=torch.float32)
    sigma = torch.tensor(solver.sigma, dtype=torch.float32)

    u = model(t, x)  # (batch, 1)

    # u_t
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # grad_x u
    grad_u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # second derivatives
    grad_u_x1 = grad_u_x[:, 0:1]
    grad_u_x2 = grad_u_x[:, 1:2]

    second_x1 = torch.autograd.grad(
        grad_u_x1,
        x,
        grad_outputs=torch.ones_like(grad_u_x1),
        create_graph=True
    )[0]

    second_x2 = torch.autograd.grad(
        grad_u_x2,
        x,
        grad_outputs=torch.ones_like(grad_u_x2),
        create_graph=True
    )[0]

    h11 = second_x1[:, 0:1]
    h12 = second_x1[:, 1:2]
    h21 = second_x2[:, 0:1]
    h22 = second_x2[:, 1:2]

    sigma_sigma_T = sigma @ sigma.T
    trace_term = (
        sigma_sigma_T[0, 0] * h11
        + sigma_sigma_T[0, 1] * h12
        + sigma_sigma_T[1, 0] * h21
        + sigma_sigma_T[1, 1] * h22
    )

    Hx = torch.matmul(x, H.T)
    Malpha = torch.matmul(alpha, M.T)

    drift_term = torch.sum(grad_u_x * Hx, dim=1, keepdim=True)
    control_term = torch.sum(grad_u_x * Malpha, dim=1, keepdim=True)
    state_cost = quadratic_form_torch(x, C)
    control_cost = quadratic_form_torch(alpha, D)

    residual = (
        u_t
        + 0.5 * trace_term
        + drift_term
        + control_term
        + state_cost
        + control_cost
    )

    return residual


def boundary_residual(model, solver, t_terminal, x_terminal):
    """
    Terminal condition:
    u(T, x) = x^T R x
    """
    R = torch.tensor(solver.R, dtype=torch.float32)
    u_terminal = model(t_terminal, x_terminal)
    target = quadratic_form_torch(x_terminal, R)
    return u_terminal - target


def simulate_cost_constant_control(solver, t0, x0, N, n_paths, seed=None):
    """
    Monte Carlo estimate for the fixed control alpha = (1,1)^T.
    """
    if seed is not None:
        np.random.seed(seed)

    x0 = np.array(x0, dtype=float)
    T = solver.T
    dt = (T - t0) / N
    sqrt_dt = np.sqrt(dt)

    alpha = np.array([1.0, 1.0], dtype=float)

    X = np.tile(x0, (n_paths, 1))
    running_cost = np.zeros(n_paths, dtype=float)

    for _ in range(N):
        state_cost = np.einsum("...i,ij,...j->...", X, solver.C, X)
        control_cost = alpha @ solver.D @ alpha
        running_cost += (state_cost + control_cost) * dt

        drift = X @ solver.H.T + alpha @ solver.M.T
        dW = np.random.randn(n_paths, 2) * sqrt_dt
        diffusion = dW @ solver.sigma.T

        X = X + drift * dt + diffusion

    terminal_cost = np.einsum("...i,ij,...j->...", X, solver.R, X)
    total_cost = running_cost + terminal_cost

    return float(np.mean(total_cost))


def compute_single_mc_relative_error(model, solver, t0=0.0, x0=None, N=2000, n_paths=20000, seed=123):
    """
    Relative error between the DGM prediction and the Monte Carlo benchmark
    at one fixed test point.

    relative error = |u_DGM - u_MC| / |u_MC|
    """
    if x0 is None:
        x0 = np.array([1.0, 0.5], dtype=float)

    mc_val = simulate_cost_constant_control(
        solver=solver,
        t0=float(t0),
        x0=x0,
        N=N,
        n_paths=n_paths,
        seed=seed
    )

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor([[t0]], dtype=torch.float32)
        x_tensor = torch.tensor([[x0[0], x0[1]]], dtype=torch.float32)
        dgm_val = model(t_tensor, x_tensor).item()

    rel_error = abs(dgm_val - mc_val) / max(abs(mc_val), 1e-12)
    return rel_error


def print_metrics_table(metrics_rows):
    """
    Print the Exercise 3.1 metrics table in terminal only.
    """
    print("\nExercise 3.1 metrics table")
    print(
        f"{'Epoch':>8} "
        f"{'Total Loss':>15} "
        f"{'Eqn Residual':>15} "
        f"{'Boundary':>15} "
        f"{'Relative Error':>15}"
    )

    for row in metrics_rows:
        epoch = int(row[0])
        total_loss = row[1]
        eqn_loss = row[2]
        boundary_loss = row[3]
        rel_error = row[4]

        print(
            f"{epoch:8d} "
            f"{total_loss:15.6e} "
            f"{eqn_loss:15.6e} "
            f"{boundary_loss:15.6e} "
            f"{rel_error:15.6e}"
        )


def plot_relative_error_curve(metrics_rows, save_path="figures/ex3_1_dgm_mc_error.png", show_plot=True):
    """
    Plot relative error against training epochs, similar to the reference style.
    """
    epochs = [int(row[0]) for row in metrics_rows]
    rel_errors = [row[4] for row in metrics_rows]

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        epochs,
        rel_errors,
        marker="o",
        linewidth=1.2,
        markersize=4,
        label="Relative Error vs MC"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Relative Error (Log Scale)")
    plt.title("DGM Relative Error Evaluated against Monte Carlo")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Saved figure to: {save_path}")


def train_dgm_linear_pde(
    save_path="figures/ex3_1_dgm_loss.png",
    n_steps=4000,
    batch_size=512,
    lr=1e-3,
    hidden_dim=100,
    show_plot=True
):
    solver = build_test_solver()
    model = NetDGM(input_dim=3, hidden_dim=hidden_dim, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    # rows: [epoch, total_loss, eqn_loss, boundary_loss, relative_mc_error]
    metrics_rows = []

    for step in range(1, n_steps + 1):
        t_in, x_in = sample_interior(batch_size, solver.T)
        t_T, x_T = sample_terminal(batch_size, solver.T)

        reqn = pde_residual(model, solver, t_in, x_in)
        rbd = boundary_residual(model, solver, t_T, x_T)

        loss_eqn = torch.mean(reqn ** 2)
        loss_bd = torch.mean(rbd ** 2)
        loss = loss_eqn + loss_bd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 200 == 0:
            print(
                f"step = {step:5d}, total loss = {loss.item():.8f}, "
                f"eqn = {loss_eqn.item():.8f}, boundary = {loss_bd.item():.8f}"
            )

            rel_error = compute_single_mc_relative_error(
                model=model,
                solver=solver,
                t0=0.0,
                x0=np.array([1.0, 0.5], dtype=float),
                N=2000,
                n_paths=20000,
                seed=123
            )

            metrics_rows.append([
                step,
                loss.item(),
                loss_eqn.item(),
                loss_bd.item(),
                rel_error
            ])

            print(
                f"[eval] epoch = {step:5d}, total = {loss.item():.8f}, "
                f"eqn = {loss_eqn.item():.8f}, boundary = {loss_bd.item():.8f}, "
                f"relative_mc_error = {rel_error:.8e}"
            )

    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history, linewidth=1.5, color="purple", label="Total DGM Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.title("DGM Training Loss")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Saved figure to: {save_path}")

    print_metrics_table(metrics_rows)

    return model, loss_history, solver, metrics_rows


if __name__ == "__main__":
    model, loss_history, solver, metrics_rows = train_dgm_linear_pde(
        save_path="figures/ex3_1_dgm_loss.png",
        n_steps=4000,
        batch_size=512,
        lr=1e-3,
        hidden_dim=100,
        show_plot=True
    )

    plot_relative_error_curve(
        metrics_rows,
        save_path="figures/ex3_1_dgm_mc_error.png",
        show_plot=True
    )