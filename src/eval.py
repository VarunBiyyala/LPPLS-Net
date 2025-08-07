
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def evaluate_and_plot_lppls(
    model,
    test_dataset,
    predict_linear_params,
    reconstruct_lppls,
    n_samples=32,
    n_rows=8,
    n_cols=4,
    n_points=252,
    device=None,
    fig_width=16,
    fig_height=32,
    show=True
):
    """
    Evaluate the model on test_dataset, plot predictions, and compute parameter-space MSE.

    Args:
        model: The trained PyTorch model.
        test_dataset: List of (series, params, noise_info) samples.
        predict_linear_params: Function to predict LPPLS linear parameters.
        reconstruct_lppls: Function to reconstruct LPPLS series.
        n_samples: Number of samples to plot/evaluate.
        n_rows, n_cols: Grid size for subplots.
        n_points: Length of the time series.
        device: torch.device, optional.
        fig_width, fig_height: Figure size.
        show: If True, show the plot.
    Returns:
        test_loss: MSE loss (float) over parameter predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    selected_indices = np.linspace(0, len(test_dataset) - 1, n_samples, dtype=int)
    plt.figure(figsize=(fig_width, fig_height))

    all_preds = []
    all_targets = []

    model.eval()
    for i, idx in enumerate(selected_indices, start=1):
        test_series, test_params, noise_info = test_dataset[idx]
        scaled_test_tensor = torch.from_numpy(test_series.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_params = model(scaled_test_tensor)
        pred_tc, pred_m, pred_omega = pred_params[0].cpu().numpy()

        # Reconstruct predicted LPPLS
        a, b, c1, c2 = predict_linear_params(pred_tc, pred_m, pred_omega, test_series)
        pred_lppls = reconstruct_lppls(pred_tc, pred_m, pred_omega, a, b, c1, c2)
        lppls_min, lppls_max = pred_lppls.min(), pred_lppls.max()
        pred_lppls_scaled = (pred_lppls - lppls_min) / (lppls_max - lppls_min + 1e-12)

        ax = plt.subplot(n_rows, n_cols, i)
        ax.plot(test_series, label="Noisy test data", color='C0')
        ax.plot(pred_lppls_scaled, label="Predicted LPPLS fit", color='C1')

        if 0 <= pred_tc <= n_points:
            ax.axvline(x=pred_tc, color='r', linestyle='--', label="predicted tc")

        ax.set_title(
            f"Sample idx={idx}\n"
            f"Noise type: {noise_info[0]} Noise level: {float(noise_info[1])*100:.2f}%\n"
            f"True (tc={test_params[0]:.2f}, m={test_params[1]:.2f}, w={test_params[2]:.2f})\n"
            f"Pred (tc={pred_tc:.2f}, m={pred_m:.2f}, w={pred_omega:.2f})",
            fontsize=9
        )
        ax.grid(True)
        ax.legend(fontsize=7)

        all_targets.append([test_params[0], test_params[1], test_params[2]])
        all_preds.append([pred_tc, pred_m, pred_omega])

    plt.tight_layout()
    if show:
        plt.show()

    # Compute parameter-space MSE
    criterion = nn.MSELoss()
    all_preds_t = torch.tensor(all_preds, dtype=torch.float32)
    all_targets_t = torch.tensor(all_targets, dtype=torch.float32)
    test_loss = criterion(all_preds_t, all_targets_t)
    print(f"\nParameter-space MSE on these {n_samples} test samples: {test_loss.item():.6f}")
    return test_loss.item()

def evaluate_model_param_mse(model, test_dataset, device=None, verbose=True):
    """
    Evaluate a model on test_dataset and return parameter-space MSE.

    Args:
        model: PyTorch model.
        test_dataset: list of (series, params, noise_info) samples.
        device: torch.device or None. If None, will auto-select.
        verbose: Print the loss if True.

    Returns:
        test_loss (float)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for i in range(len(test_dataset)):
        test_series, test_params, _ = test_dataset[i]
        input_tensor = torch.from_numpy(test_series.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_params = model(input_tensor)[0].cpu().numpy()

        all_preds.append(pred_params.tolist())
        all_targets.append(test_params)

    criterion = nn.MSELoss()
    all_preds_t = torch.tensor(all_preds, dtype=torch.float32)
    all_targets_t = torch.tensor(all_targets, dtype=torch.float32)
    test_loss = criterion(all_preds_t, all_targets_t)

    if verbose:
        print(f"\nParameter-space MSE on all {len(test_dataset)} test samples: {test_loss.item():.6f}")

    return test_loss.item()
