import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lqr_solver import LQRSolver


class ValueNet(nn.Module):
    """
    Value network v(t, x; theta_val)
    Input: (t, x1, x2)
    Output: scalar value
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
        return self.output_layer(h)


class ControlNet(nn.Module):
    """
    Control network a(t, x; theta_act)
    Input: (t, x1, x2)
    Output: 2D control
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, t, x):
        z = torch.cat([t, x], dim=1)
        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        return self.fc_out(h)


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
    t = torch.rand(batch_size, 1, dtype=torch.float32) * T
    x = -3.0 + 6.0 * torch.rand(batch_size, 2, dtype=torch.float32)
    t.requires_grad_(True)
    x.requires_grad_(True)
    return t, x


def sample_terminal(batch_size, T):
    t = torch.full((batch_size, 1), T, dtype=torch.float32)
    x = -3.0 + 6.0 * torch.rand(batch_size, 2, dtype=torch.float32)
    return t, x


def pde_residual_with_policy(value_net, control_net, solver, t, x):
    """
    PDE for policy evaluation step:
    u_t + 1/2 tr(sigma sigma^T u_xx)
        + (u_x)^T Hx + (u_x)^T M a(t,x;theta_act)
        + x^T C x + a^T D a = 0
    """
    H = torch.tensor(solver.H, dtype=torch.float32)
    M = torch.tensor(solver.M, dtype=torch.float32)
    C = torch.tensor(solver.C, dtype=torch.float32)
    D = torch.tensor(solver.D, dtype=torch.float32)
    sigma = torch.tensor(solver.sigma, dtype=torch.float32)

    u = value_net(t, x)
    a = control_net(t, x)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    grad_u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    grad_u_x1 = grad_u_x[:, 0:1]
    grad_u_x2 = grad_u_x[:, 1:2]

    second_x1 = torch.autograd.grad(
        grad_u_x1, x, grad_outputs=torch.ones_like(grad_u_x1), create_graph=True
    )[0]
    second_x2 = torch.autograd.grad(
        grad_u_x2, x, grad_outputs=torch.ones_like(grad_u_x2), create_graph=True
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
    Ma = torch.matmul(a, M.T)

    drift_term = torch.sum(grad_u_x * Hx, dim=1, keepdim=True)
    control_term = torch.sum(grad_u_x * Ma, dim=1, keepdim=True)
    state_cost = quadratic_form_torch(x, C)
    control_cost = quadratic_form_torch(a, D)

    residual = (
        u_t
        + 0.5 * trace_term
        + drift_term
        + control_term
        + state_cost
        + control_cost
    )
    return residual


def boundary_residual(value_net, solver, t_terminal, x_terminal):
    R = torch.tensor(solver.R, dtype=torch.float32)
    u_terminal = value_net(t_terminal, x_terminal)
    target = quadratic_form_torch(x_terminal, R)
    return u_terminal - target


def policy_evaluation_step(
    value_net,
    control_net,
    solver,
    n_steps=1000,
    batch_size=512,
    lr=1e-3
):
    optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    loss_history = []

    for step in range(1, n_steps + 1):
        t_in, x_in = sample_interior(batch_size, solver.T)
        t_T, x_T = sample_terminal(batch_size, solver.T)

        reqn = pde_residual_with_policy(value_net, control_net, solver, t_in, x_in)
        rbd = boundary_residual(value_net, solver, t_T, x_T)

        loss_eqn = torch.mean(reqn ** 2)
        loss_bd = torch.mean(rbd ** 2)
        loss = loss_eqn + loss_bd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 200 == 0:
            print(
                f"[value] step = {step:5d}, total = {loss.item():.8f}, "
                f"eqn = {loss_eqn.item():.8f}, boundary = {loss_bd.item():.8f}"
            )

    return loss_history


def actor_hamiltonian(value_net, control_net, solver, t, x):
    """
    Hamiltonian to minimize in the policy improvement step:
    (grad_x v)^T Hx + (grad_x v)^T M a + x^T C x + a^T D a
    """
    H = torch.tensor(solver.H, dtype=torch.float32)
    M = torch.tensor(solver.M, dtype=torch.float32)
    C = torch.tensor(solver.C, dtype=torch.float32)
    D = torch.tensor(solver.D, dtype=torch.float32)

    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    v = value_net(t, x)
    grad_v_x = torch.autograd.grad(
        v, x, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]

    a = control_net(t, x)

    Hx = torch.matmul(x, H.T)
    Ma = torch.matmul(a, M.T)

    term1 = torch.sum(grad_v_x * Hx, dim=1, keepdim=True)
    term2 = torch.sum(grad_v_x * Ma, dim=1, keepdim=True)
    term3 = quadratic_form_torch(x, C)
    term4 = quadratic_form_torch(a, D)

    ham = term1 + term2 + term3 + term4
    return torch.mean(ham)


def policy_improvement_step(
    value_net,
    control_net,
    solver,
    n_steps=1500,
    batch_size=512,
    lr=5e-4
):
    optimizer = torch.optim.Adam(control_net.parameters(), lr=lr)
    objective_history = []

    for step in range(1, n_steps + 1):
        t, x = sample_interior(batch_size, solver.T)

        ham = actor_hamiltonian(value_net, control_net, solver, t, x)

        optimizer.zero_grad()
        ham.backward()
        optimizer.step()

        objective_history.append(ham.item())

        if step % 200 == 0:
            print(f"[actor] step = {step:5d}, Hamiltonian = {ham.item():.8f}")

    return objective_history


def evaluate_against_lqr(value_net, control_net, solver, n_test=2000):
    """
    Compare current policy iteration networks with the Exercise 1.1 benchmark.
    """
    t = torch.rand(n_test, 1, dtype=torch.float32) * solver.T
    x = -3.0 + 6.0 * torch.rand(n_test, 2, dtype=torch.float32)

    with torch.no_grad():
        v_pred = value_net(t, x)
        a_pred = control_net(t, x)

        v_true = solver.value_function(
            t.squeeze(1),
            x.unsqueeze(1)
        )
        a_true = solver.optimal_control(
            t.squeeze(1),
            x.unsqueeze(1)
        )

        value_mse = torch.mean((v_pred - v_true) ** 2).item()
        control_mse = torch.mean((a_pred - a_true) ** 2).item()

    return value_mse, control_mse


def print_pi_table(rows):
    print("\nExercise 4.1 policy iteration table")
    print(
        f"{'Iter':>6} "
        f"{'Value PDE Loss':>18} "
        f"{'Actor Ham':>15} "
        f"{'Value MSE':>15} "
        f"{'Control MSE':>15}"
    )

    for row in rows:
        k = int(row[0])
        v_loss = row[1]
        a_loss = row[2]
        v_mse = row[3]
        a_mse = row[4]

        print(
            f"{k:6d} "
            f"{v_loss:18.6e} "
            f"{a_loss:15.6e} "
            f"{v_mse:15.6e} "
            f"{a_mse:15.6e}"
        )


def run_policy_iteration(
    n_iterations=8,
    value_steps=1000,
    actor_steps=1500,
    batch_size=512,
    lr_value=1e-3,
    lr_actor=5e-4,
    show_plots=True
):
    solver = build_test_solver()

    value_net = ValueNet(hidden_dim=100)
    control_net = ControlNet(hidden_dim=100)

    metrics_rows = []

    for k in range(1, n_iterations + 1):
        print(f"\n=== Policy iteration {k} / {n_iterations} ===")

        value_losses = policy_evaluation_step(
            value_net, control_net, solver,
            n_steps=value_steps, batch_size=batch_size, lr=lr_value
        )

        actor_objectives = policy_improvement_step(
            value_net, control_net, solver,
            n_steps=actor_steps, batch_size=batch_size, lr=lr_actor
        )

        value_mse, control_mse = evaluate_against_lqr(
            value_net, control_net, solver, n_test=2000
        )

        metrics_rows.append([
            k,
            value_losses[-1],
            actor_objectives[-1],
            value_mse,
            control_mse
        ])

        print(
            f"[iter {k}] value_mse = {value_mse:.6e}, "
            f"control_mse = {control_mse:.6e}"
        )

    print_pi_table(metrics_rows)

    iters = [int(row[0]) for row in metrics_rows]
    value_mses = [row[3] for row in metrics_rows]
    control_mses = [row[4] for row in metrics_rows]

    # Figure 1: value convergence only
    plt.figure(figsize=(8, 5))
    plt.semilogy(iters, value_mses, marker="o", linewidth=1.5, label="Value MSE vs LQR")
    plt.xlabel("Policy iteration")
    plt.ylabel("MSE (Log Scale)")
    plt.title("Exercise 4.1: Value Function Convergence to Exercise 1.1 benchmark")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/ex4_1_value_convergence.png", dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Figure 2: control convergence only
    plt.figure(figsize=(8, 5))
    plt.semilogy(iters, control_mses, marker="s", linewidth=1.5, label="Control MSE vs LQR")
    plt.xlabel("Policy iteration")
    plt.ylabel("MSE (Log Scale)")
    plt.title("Exercise 4.1: Markov Control Convergence to Exercise 1.1 benchmark")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/ex4_1_control_convergence.png", dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    print("Saved figure to: figures/ex4_1_value_convergence.png")
    print("Saved figure to: figures/ex4_1_control_convergence.png")

    return value_net, control_net, metrics_rows


if __name__ == "__main__":
    run_policy_iteration(
        n_iterations=8,
        value_steps=1000,
        actor_steps=1500,
        batch_size=512,
        lr_value=1e-3,
        lr_actor=5e-4,
        show_plots=True
    )