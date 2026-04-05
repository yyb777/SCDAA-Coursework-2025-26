# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lqr_solver import LQRSolver


class NetDGM(nn.Module):
    """
    A simple DGM-style feedforward network for supervised learning.
    Input:  (t, x1, x2) -> dimension 3
    Output: scalar value function -> dimension 1
    Hidden size: 100
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
        """
        t: tensor of shape (batch, 1)
        x: tensor of shape (batch, 2)
        """
        z = torch.cat([t, x], dim=1)   # (batch, 3)

        h = self.activation(self.input_layer(z))
        h = self.activation(self.hidden1(h))
        h = self.activation(self.hidden2(h))
        h = self.activation(self.hidden3(h))
        out = self.output_layer(h)

        return out


def build_test_solver():
    """
    Same test matrices as used in Exercises 1.1 and 1.2.
    Replace later if your coursework specifies different matrices.
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


def sample_value_data(solver, batch_size):
    """
    Sample training data for Exercise 2.1:
      t ~ Uniform([0,T])
      x ~ Uniform([-3,3]^2)
      y = v(t, x)
    """
    T = solver.T

    t = torch.rand(batch_size, 1, dtype=torch.float32) * T
    x = -3.0 + 6.0 * torch.rand(batch_size, 2, dtype=torch.float32)

    y = solver.value_function(
        t.squeeze(1),
        x.unsqueeze(1)
    ).detach()

    return t, x, y


def train_value_network(
    save_path="figures/ex2_1_value_loss.png",
    n_steps=3000,
    batch_size=512,
    lr=1e-3,
    hidden_dim=100,
    show_plot=True
):
    """
    Train a NetDGM network to approximate the value function v(t,x).
    """
    solver = build_test_solver()
    model = NetDGM(input_dim=3, hidden_dim=hidden_dim, output_dim=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_history = []
    value_metrics = []   # [epoch, value_loss]

    for step in range(1, n_steps + 1):
        t_batch, x_batch, y_batch = sample_value_data(solver, batch_size)

        pred = model(t_batch, x_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 200 == 0:
            print(f"step = {step:5d}, loss = {loss.item():.8f}")

        if step % 500 == 0:
            value_metrics.append([step, loss.item()])

    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history, linewidth=1.5, label="Value Function Loss (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.title("Exercise 2.1: Training Loss for Value Function (NetDGM)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Saved figure to: {save_path}")

    return model, loss_history, value_metrics

class FFNControl(nn.Module):
    """
    Feedforward neural network for Exercise 2.2.

    Input:  (t, x1, x2) -> dimension 3
    Output: control a(t, x) -> dimension 2

    Required by the coursework:
    - 2 hidden layers
    - hidden size 100
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.Tanh()

    def forward(self, t, x):
        """
        t: tensor of shape (batch, 1)
        x: tensor of shape (batch, 2)
        """
        z = torch.cat([t, x], dim=1)   # (batch, 3)

        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        out = self.fc_out(h)           # (batch, 2)

        return out


def sample_control_data(solver, batch_size):
    """
    Sample training data for Exercise 2.2:
      t ~ Uniform([0,T])
      x ~ Uniform([-3,3]^2)
      y = a(t, x)
    """
    T = solver.T

    t = torch.rand(batch_size, 1, dtype=torch.float32) * T
    x = -3.0 + 6.0 * torch.rand(batch_size, 2, dtype=torch.float32)

    y = solver.optimal_control(
        t.squeeze(1),
        x.unsqueeze(1)
    ).detach()

    return t, x, y


def train_control_network(
    save_path="figures/ex2_2_control_loss.png",
    n_steps=3000,
    batch_size=512,
    lr=1e-3,
    hidden_dim=100,
    show_plot=True
):
    """
    Train a 2-hidden-layer FFN to approximate the optimal control a(t, x).
    """
    solver = build_test_solver()
    model = FFNControl(input_dim=3, hidden_dim=hidden_dim, output_dim=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_history = []
    control_metrics = []   # [epoch, control_loss]

    for step in range(1, n_steps + 1):
        t_batch, x_batch, y_batch = sample_control_data(solver, batch_size)

        pred = model(t_batch, x_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 200 == 0:
            print(f"[control] step = {step:5d}, loss = {loss.item():.8f}")

        if step % 500 == 0:
            control_metrics.append([step, loss.item()])

    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history, linewidth=1.5, label="Control Loss (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.title("Exercise 2.2: Training Loss for Markov Control (FFN)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Saved figure to: {save_path}")

    return model, loss_history, control_metrics

def print_ex2_metrics_table(metrics_rows):
    """
    Print Exercise 2 metrics table in terminal only.

    Each row should be:
    [epoch, value_loss, control_loss]
    """
    print("\nExercise 2 metrics table")
    print(f"{'Epoch':>8} {'Value Loss':>15} {'Control Loss':>15}")

    for row in metrics_rows:
        epoch = int(row[0])
        value_loss = row[1]
        control_loss = row[2]

        print(
            f"{epoch:8d} "
            f"{value_loss:15.6e} "
            f"{control_loss:15.6e}"
        )

def evaluate_value_test_mse(model, solver, n_test=2000):
    """
    Evaluate test MSE for the value network on freshly sampled points.
    """
    model.eval()
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        t_test, x_test, y_test = sample_value_data(solver, n_test)
        pred_test = model(t_test, x_test)
        test_mse = loss_fn(pred_test, y_test).item()

    return test_mse


def evaluate_control_test_mse(model, solver, n_test=2000):
    """
    Evaluate test MSE for the control network on freshly sampled points.
    """
    model.eval()
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        t_test, x_test, y_test = sample_control_data(solver, n_test)
        pred_test = model(t_test, x_test)
        test_mse = loss_fn(pred_test, y_test).item()

    return test_mse

if __name__ == "__main__":
    value_model, value_loss_history, value_metrics = train_value_network(
        save_path="figures/ex2_1_value_loss.png",
        n_steps=3000,
        batch_size=512,
        lr=1e-3,
        hidden_dim=100,
        show_plot=True
    )

    control_model, control_loss_history, control_metrics = train_control_network(
        save_path="figures/ex2_2_control_loss.png",
        n_steps=3000,
        batch_size=512,
        lr=1e-3,
        hidden_dim=100,
        show_plot=True
    )

    # merge the two metric lists by epoch
    metrics_rows = []
    n_rows = min(len(value_metrics), len(control_metrics))

    for i in range(n_rows):
        epoch_v, loss_v = value_metrics[i]
        epoch_c, loss_c = control_metrics[i]

        if epoch_v == epoch_c:
            metrics_rows.append([epoch_v, loss_v, loss_c])

    print_ex2_metrics_table(metrics_rows)

    # small test MSE evaluation
    solver = build_test_solver()

    value_test_mse = evaluate_value_test_mse(value_model, solver, n_test=2000)
    control_test_mse = evaluate_control_test_mse(control_model, solver, n_test=2000)

    print("\nExercise 2 test MSE")
    print(f"Value network test MSE   = {value_test_mse:.6e}")
    print(f"Control network test MSE = {control_test_mse:.6e}")