
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import argparse

def add_white_noise(series, noise_level=0.05):
    """
    Add Gaussian white noise to a series.
    noise_level is the std dev as fraction of the time series range.
    """
    # Determine amplitude based on the range
    data_range = np.max(series) - np.min(series)
    if data_range < 1e-12:
        data_range = 1.0
    sigma = noise_level * data_range
    noise = np.random.normal(0, sigma, size=len(series))
    return series + noise

def generate_ar1_noise(length, phi=0.9, sigma=0.01):
    """
    Generate AR(1) noise with parameter phi:
      eta_t = phi * eta_{t-1} + epsilon_t,
    where epsilon_t ~ Normal(0, sigma^2).
    """
    eta = np.zeros(length)
    for t in range(1, length):
        eps = np.random.normal(0, sigma)
        eta[t] = phi * eta[t-1] + eps
    return eta

def add_ar1_noise(series, phi=0.9, noise_level=0.05):
    """
    Add AR(1) noise to a series. noise_level is again fraction of range.
    """
    data_range = np.max(series) - np.min(series)
    if data_range < 1e-12:
        data_range = 1.0
    sigma = noise_level * data_range
    ar1 = generate_ar1_noise(len(series), phi=phi, sigma=sigma)
    return series + ar1

def generate_lppls_series(tc, m, omega, n_points=252, A=1.0, B=-1.0):
    """
    Generate a noise-free LPPLS time series of length `n_points`,
    with user-specified (tc, m, omega).
    """
    t_array = np.arange(n_points)
    epsilon = 1e-7  # to avoid log(0)

    # Evaluate (tc - t)
    denom = (tc - t_array)
    # Basic LPPLS
    lppls = A + B*np.power(np.abs(denom), m)
    c1 = 0.1
    c2 = 0.1
    lppls = lppls*(1 + c1*np.cos(omega*np.log(np.abs(denom)+epsilon)) + c2*np.sin(omega*np.log(np.abs(denom)+epsilon)))

    return lppls

def add_arch_garch_noise(series, model_type='ARCH', noise_level=0.1, arch_params=None):
    """
    Add ARCH or GARCH noise to the LPPLS series with amplitude control.

    Parameters:
    - model_type: 'ARCH' or 'GARCH'
    - noise_level: fraction of series range to control amplitude
    - arch_params: [omega, alpha1, (beta1)] depending on model

    Returns:
    - noisy_series as a numpy array
    """
    n = len(series)
    if arch_params is None:
        arch_params = [0.01, 0.05, 0.8] if model_type == 'GARCH' else [0.05, 0.8]

    mod = arch_model(None, mean='Zero', vol=model_type, p=1, q=(1 if model_type == 'GARCH' else 0))
    sim = mod.simulate(arch_params, nobs=n)
    noise = sim['data'].values  # Convert Series to numpy array

    series_range = np.max(series) - np.min(series)
    if series_range < 1e-12:
        series_range = 1.0

    # Desired standard deviation for noise (RMS amplitude)
    desired_noise_std = noise_level * series_range

    # Actually scale by standard deviation
    actual_noise_std = np.std(noise)
    if actual_noise_std < 1e-12:
        actual_noise_std = 1.0  # Avoid division by zero

    scaled_noise = noise / actual_noise_std * desired_noise_std

    return (series + scaled_noise).astype(np.float32)

def generate_synthetic_dataset(n_samples=10000,
                               n_points=252,
                               noise_type='white', phi = 0.9):
    """
    Generate a dataset of (time_series, parameters),
    where each time_series is 1D with length n_points,
    and parameters = (tc, m, omega) used to generate it.

    noise_type: 'white', 'ar1', 'ARCH', 'GARCH', 'white+ar1', 'ARCH+GARCH' or 'ar1+GARCH'
    """

    dataset = []
    for _ in range(n_samples):
        # Sample LPPLS parameters
        tc = np.random.uniform(n_points, n_points+50)
        m = np.random.uniform(0.1, 0.9)
        omega = np.random.uniform(6, 13)

        # Generate LPPLS signal
        series = generate_lppls_series(tc, m, omega, n_points=n_points,
                                       A=1.0, B=-1.0)

        if noise_type == 'white':
            # Sample noise amplitude
            chosen_noise = np.random.uniform(0.01, 0.15)
            series_noisy = add_white_noise(series, noise_level=chosen_noise)
            selected_noise_type = 'white'
        elif noise_type == 'ar1':
            chosen_noise = np.random.uniform(0.01, 0.05)
            series_noisy = add_ar1_noise(series, phi=phi, noise_level=chosen_noise)
            selected_noise_type = 'ar1'
        elif noise_type == 'ARCH':
            chosen_noise = np.random.uniform(0.01, 0.05)
            series_noisy = add_arch_garch_noise(series, noise_level=chosen_noise)
            selected_noise_type = 'ARCH'
        elif noise_type == 'GARCH':
            chosen_noise = np.random.uniform(0.01, 0.05)
            series_noisy = add_arch_garch_noise(series, noise_level=chosen_noise)
            selected_noise_type = 'GARCH'
        elif noise_type == 'white+ar1': # 'mixed' or 'both'
            # 50% chance white, 50% chance AR1
            if np.random.rand() < 0.5:
                selected_noise_type  = 'white'
                chosen_noise = np.random.uniform(0.01, 0.15)
                series_noisy = add_white_noise(series, noise_level=chosen_noise)
            else:
                selected_noise_type  = 'ar1'
                chosen_noise = np.random.uniform(0.01, 0.05)
                series_noisy = add_ar1_noise(series, phi=phi, noise_level=chosen_noise)
        elif noise_type == 'ARCH+GARCH': # 'mixed' or 'both'
            # 50% chance white, 50% chance AR1
            if np.random.rand() < 0.5:
                selected_noise_type  = 'ARCH'
                chosen_noise = np.random.uniform(0.01, 0.05)
                series_noisy = add_arch_garch_noise(series, noise_level=chosen_noise)
            else:
                selected_noise_type  = 'GARCH'
                chosen_noise = np.random.uniform(0.01, 0.05)
                series_noisy = add_arch_garch_noise(series, noise_level=chosen_noise)

        elif noise_type == 'ar1+GARCH': # 'mixed' or 'both'
            # 50% chance white, 50% chance AR1
            if np.random.rand() < 0.5:
                selected_noise_type  = 'ar1'
                chosen_noise = np.random.uniform(0.01, 0.05)
                series_noisy = add_ar1_noise(series, phi=phi, noise_level=chosen_noise)
            else:
                selected_noise_type  = 'GARCH'
                chosen_noise = np.random.uniform(0.01, 0.05)
                series_noisy = add_arch_garch_noise(series, noise_level=chosen_noise)


        elif noise_type == 'none':
            chosen_noise = 0
            series_noisy = series.astype(np.float32) # Ensure it's a numpy array
            selected_noise_type = 'none'
        else:
            raise ValueError("'white', 'ar1', 'ARCH', 'GARCH', 'white+ar1', 'ARCH+GARCH' or 'ar1+GARCH'")

        # Optional scaling in [0,1] (per sample)
        min_val, max_val = np.min(series_noisy), np.max(series_noisy)
        denom = max_val - min_val
        if denom < 1e-12:
            scaled_series = series_noisy
        else:
            scaled_series = (series_noisy - min_val) / denom

        dataset.append((scaled_series.astype(np.float32),
                        np.array([tc, m, omega], dtype=np.float32),
                        np.array([selected_noise_type, chosen_noise])))

    return dataset

def call_generate_synthetic_dataset(n_samples=100, n_points=252, noise_type="ARCH+GARCH", phi=0.8):
    N_SAMPLES = n_samples
    N_POINTS = n_points
    NOISE_TYPE = noise_type
    phi = phi

    print("Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(
        n_samples=N_SAMPLES,
        n_points=N_POINTS,
        noise_type=NOISE_TYPE,
        phi=phi
    )
    print("Dataset generation complete! Size:", len(dataset))

    # Split dataset into train/val
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_size = int(0.8 * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Plotting
    plt.figure(figsize=(18, 12))
    sample_indices = np.random.choice(len(dataset), size=16, replace=False)
    for i, idx in enumerate(sample_indices, start=1):
        series, params, noise_info = dataset[idx]
        true_tc = params[0]

        ax = plt.subplot(4, 4, i)
        ax.plot(series, label="Noisy data", lw=1)
        if 0 <= true_tc <= N_POINTS:
            ax.axvline(x=true_tc, color='r', linestyle='--', label="true tc")
        ax.set_title(f"tc={true_tc:.1f}, m={params[1]:.2f}, w={params[2]:.2f}\nNoise type: {noise_info[0]}\n Noise level: {float(noise_info[1])*100:.2f}% ID: {idx}")
        ax.grid(True)
        ax.legend(fontsize=4)
    plt.tight_layout()
    plt.savefig("dataset_samples.png")
    print("Plot saved to dataset_samples.png")
    return train_dataset, val_dataset
