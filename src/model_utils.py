
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import torch.nn.functional as F
import copy

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

def train_p_lnn(
    model,
    train_dataset,
    val_dataset,
    n_epochs=100,
    batch_size=8,
    lr=1e-5,
    weight_decay=1e-4,
    patience=10,
    min_delta=1e-5
):
    """
    Train the P-LNN model using weighted MSE loss.

    Early stopping is based only on validation loss.

    Parameters
    ----------
    model : torch.nn.Module
        LPPLS neural network.

    train_dataset : list
        Training samples of the form (X, Y).

    val_dataset : list
        Validation samples of the form (X, Y).

    n_epochs : int
        Maximum number of training epochs.

    batch_size : int
        Number of samples per mini-batch.

    lr : float
        Learning rate.

    weight_decay : float
        L2 regularization strength.

    patience : int
        Number of consecutive epochs without sufficient
        validation improvement before stopping.

    min_delta : float
        Minimum absolute decrease in validation loss that
        counts as an improvement.

    Returns
    -------
    train_losses : list
        Mean training loss for each completed epoch.

    val_losses : list
        Mean validation loss for each completed epoch.
    """

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = model.to(device)

    # --------------------------------------------------
    # Convert datasets to tensors
    # --------------------------------------------------
    X_train = torch.stack([
        torch.from_numpy(sample[0]).float()
        for sample in train_dataset
    ])

    Y_train = torch.stack([
        torch.from_numpy(sample[1]).float()
        for sample in train_dataset
    ])

    X_val = torch.stack([
        torch.from_numpy(sample[0]).float()
        for sample in val_dataset
    ])

    Y_val = torch.stack([
        torch.from_numpy(sample[1]).float()
        for sample in val_dataset
    ])

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    indices = np.arange(n_train)

    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    no_improvement_epochs = 0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(n_epochs):

        # ==================================================
        # TRAINING
        # ==================================================
        model.train()

        np.random.shuffle(indices)

        total_train_loss = 0.0

        for start_idx in range(0, n_train, batch_size):

            end_idx = min(
                start_idx + batch_size,
                n_train
            )

            batch_indices = indices[start_idx:end_idx]

            x_batch = X_train[batch_indices].to(device)
            y_batch = Y_train[batch_indices].to(device)

            # Forward pass
            preds = model(x_batch)

            loss = weighted_mse_loss(
                preds,
                y_batch
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Actual number of samples in this batch
            actual_batch_size = x_batch.size(0)

            total_train_loss += (
                loss.item()
                * actual_batch_size
            )

        epoch_train_loss = (
            total_train_loss / n_train
        )

        train_losses.append(
            epoch_train_loss
        )

        # ==================================================
        # VALIDATION
        # ==================================================
        model.eval()

        total_val_loss = 0.0

        with torch.no_grad():

            for start_idx in range(
                0,
                n_val,
                batch_size
            ):

                end_idx = min(
                    start_idx + batch_size,
                    n_val
                )

                x_val_batch = (
                    X_val[start_idx:end_idx]
                    .to(device)
                )

                y_val_batch = (
                    Y_val[start_idx:end_idx]
                    .to(device)
                )

                preds_val = model(
                    x_val_batch
                )

                loss_val = weighted_mse_loss(
                    preds_val,
                    y_val_batch
                )

                # IMPORTANT:
                # Use actual batch size,
                # especially for the final partial batch.
                actual_batch_size = (
                    x_val_batch.size(0)
                )

                total_val_loss += (
                    loss_val.item()
                    * actual_batch_size
                )

        epoch_val_loss = (
            total_val_loss / n_val
        )

        val_losses.append(
            epoch_val_loss
        )

        # ==================================================
        # CHECK FOR VALIDATION IMPROVEMENT
        # ==================================================
        improvement = (
            best_val_loss
            - epoch_val_loss
        )

        if improvement > min_delta:

            best_val_loss = epoch_val_loss

            best_epoch = epoch + 1

            # Save weights corresponding to
            # best validation performance
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            no_improvement_epochs = 0

        else:

            no_improvement_epochs += 1

        # ==================================================
        # LOGGING
        # ==================================================
        if (
            (epoch + 1) == 1
            or (epoch + 1) % 5 == 0
        ):

            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"Train Loss: "
                f"{epoch_train_loss:.6f} | "
                f"Val Loss: "
                f"{epoch_val_loss:.6f}"
            )

        # ==================================================
        # EARLY STOPPING
        # ==================================================
        if (
            no_improvement_epochs
            >= patience
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch + 1}."
            )

            print(
                f"Validation loss did not improve "
                f"by more than {min_delta} for "
                f"{patience} consecutive epochs."
            )

            break

    # --------------------------------------------------
    # Restore best model
    # --------------------------------------------------
    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

        print(
            f"\nRestored model from "
            f"epoch {best_epoch} "
            f"with best validation loss "
            f"{best_val_loss:.6f}."
        )

    print("Training complete!")

    return train_losses, val_losses