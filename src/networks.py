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

    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history, linewidth=1.5, label="Value Function Loss (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.title("Exercise 2.1: Training Loss for Value Function (NetDGM)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Saved figure to: {save_path}")

    return model, loss_history


if __name__ == "__main__":
    train_value_network(
        save_path="figures/ex2_1_value_loss.png",
        n_steps=3000,
        batch_size=512,
        lr=1e-3,
        hidden_dim=100,
        show_plot=True
    )



# %%
