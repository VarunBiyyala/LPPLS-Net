
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import torch.nn.functional as F

class PLNNet1(nn.Module):
    """
    Feedforward network that takes a 1D time series (length = n_points)
    and outputs (t_c, m, omega). Fixed hidden layer dimensions.
    """
    def __init__(self, n_points=252, hidden_size=128, p_drop = 0.1):
        super(PLNNet1, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_points, hidden_size),
            nn.ReLU(),
            #nn.Dropout(p_drop),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(p_drop),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(p_drop),
            nn.Linear(hidden_size, 3)  # final layer outputs [tc, m, omega]
        )

    def forward(self, x):
        out = self.net(x)
        tc = 252 + 50 * torch.sigmoid(out[:, 0])     # tc ∈ [252, 302]
        m = 0.8 * torch.sigmoid(out[:, 1]) + 0.1      # m ∈ [0.1, 0.9]
        w = 7 * torch.sigmoid(out[:, 2]) + 6          # omega ∈ [6, 13]
        return torch.stack([tc, m, w], dim=1)

class PLNNet2(nn.Module):
    """
    Feedforward network that takes a 1D time series (length = n_points)
    and outputs (t_c, m, omega). Dynamic hidden layer dimensions.
    """
    def __init__(self, n_points=252, hidden_size=128, p_drop = 0.1):
        super(PLNNet2, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_points, hidden_size),
            nn.ReLU(),
            #nn.Dropout(p_drop),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            #nn.Dropout(p_drop),
            nn.Linear(int(hidden_size/2), int(hidden_size/4)),
            nn.ReLU(),
            #nn.Dropout(p_drop),
            nn.Linear(int(hidden_size/4), 3)  # final layer outputs [tc, m, omega]
        )

    def forward(self, x):
        out = self.net(x)
        tc = 252 + 50 * torch.sigmoid(out[:, 0])     # tc ∈ [252, 302]
        m = 0.8 * torch.sigmoid(out[:, 1]) + 0.1      # m ∈ [0.1, 0.9]
        w = 7 * torch.sigmoid(out[:, 2]) + 6          # omega ∈ [6, 13]
        return torch.stack([tc, m, w], dim=1)

alpha_tc = 1.0 / (50**2)
alpha_m  = 1.0 / (0.8**2)
alpha_w  = 1.0 / (7.0**2)

def weighted_mse_loss(preds, targets):
    """
    preds, targets each shape (batch_size, 3):
      preds[:,0] = predicted tc
      preds[:,1] = predicted m
      preds[:,2] = predicted w
    targets: the ground truth for (tc, m, w)

    We'll define:
      loss = alpha_tc*(tc_pred - tc_gt)^2 + alpha_m*(m_pred - m_gt)^2 + alpha_w*(w_pred - w_gt)^2
    """
    # separate each dimension
    tc_pred, m_pred, w_pred = preds[:,0], preds[:,1], preds[:,2]
    tc_gt,   m_gt,   w_gt   = targets[:,0], targets[:,1], targets[:,2]

    loss_tc = alpha_tc * F.mse_loss(tc_pred, tc_gt, reduction='none')
    loss_m  = alpha_m  * F.mse_loss(m_pred,  m_gt,  reduction='none')
    loss_w  = alpha_w  * F.mse_loss(w_pred,  w_gt,  reduction='none')

    # sum or average across batch dimension
    loss = torch.mean(loss_tc + loss_m + loss_w)
    return loss

def train_p_lnn(model, train_dataset, val_dataset, n_epochs=20, batch_size=8, lr=1e-5, weight_decay=1e-4, patience=5):
    """
    Train the P-LNN model with early stopping if train and val loss do not improve by >1%.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # Convert datasets to Tensors
    X_train = torch.stack([torch.from_numpy(sample[0]) for sample in train_dataset])
    Y_train = torch.stack([torch.from_numpy(sample[1]) for sample in train_dataset])
    X_val = torch.stack([torch.from_numpy(sample[0]) for sample in val_dataset])
    Y_val = torch.stack([torch.from_numpy(sample[1]) for sample in val_dataset])

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    n_samples = len(train_dataset)
    indices = np.arange(n_samples)

    model.train()
    train_losses = []
    val_losses = []

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            x_batch = X_train[batch_indices].to(device)
            y_batch = Y_train[batch_indices].to(device)

            preds = model(x_batch)
            loss = weighted_mse_loss(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_indices)
        epoch_loss /= n_samples
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        n_val = len(val_dataset)
        with torch.no_grad():
            for start_idx in range(0, n_val, batch_size):
                end_idx = start_idx + batch_size
                x_val_batch = X_val[start_idx:end_idx].to(device)
                y_val_batch = Y_val[start_idx:end_idx].to(device)

                preds_val = model(x_val_batch)
                loss_val = weighted_mse_loss(preds_val, y_val_batch)
                val_loss += loss_val.item() * (end_idx - start_idx)
        val_loss /= n_val
        val_losses.append(val_loss)

        # Check for >1% improvement
        train_improved = epoch_loss < 0.99 * best_train_loss
        val_improved = val_loss < 0.99 * best_val_loss

        if train_improved:
            best_train_loss = epoch_loss
        if val_improved:
            best_val_loss = val_loss

        if not (train_improved or val_improved):
            no_improvement_epochs += 1
        else:
            no_improvement_epochs = 0

        model.train()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss = {epoch_loss:.6f}, Val Loss = {val_loss:.6f}")

        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no 1% improvement in train/val loss for {patience} epochs.")
            break

    return train_losses, val_losses
