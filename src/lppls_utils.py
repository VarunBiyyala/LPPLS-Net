
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

def reconstruct_lppls(tc, m, omega, a, b, c1, c2):
    """
    Rebuild an LPPLS (noise-free) time series from given parameters.
    """
    n_points=252
    t_array = np.arange(n_points).astype(float)
    epsilon = 1e-7
    denom = (tc - t_array)
    lppls = a + np.abs(denom)**m

    lppls = lppls*(b + c1*np.cos(omega*np.log(np.abs(denom)+epsilon)) + c2*np.sin(omega*np.log(np.abs(denom)+epsilon)))

    return lppls

def predict_linear_params(tc, m, w, data):
  # Matrix helpers
  local_DataSeries = [np.arange(len(data)).astype(float), data]
  def _tc():
      return torch.tensor(local_DataSeries[0])
  def _yi():
      return torch.tensor(local_DataSeries[1])

  def _fi(tc, m):
      data = torch.tensor(local_DataSeries[0])
      return torch.abs((tc - data)) ** m

  def _gi(tc, m, w):
      data = torch.tensor(local_DataSeries[0])
      return (torch.abs(tc - data)** m) * torch.cos(w * torch.log(torch.abs(tc - data)))

  def _hi(tc, m, w):
      data = torch.tensor(local_DataSeries[0])
      return (torch.abs(tc -data)** m) * torch.sin(w * torch.log(torch.abs(tc - data)))

  def _fi_pow_2(tc, m):
      return _fi(tc, m) ** 2


  def _gi_pow_2(tc, m, w):
      return _gi(tc, m, w) ** 2


  def _hi_pow_2(tc, m, w):
      return _hi(tc, m, w) ** 2


  def _figi(tc, m, w):
      return _fi(tc, m) * _gi(tc, m, w)


  def _fihi(tc, m, w):
      return _fi(tc, m) * _hi(tc, m, w)


  def _gihi(tc, m, w):
      return _gi(tc, m, w) * _hi(tc, m, w)


  def _yifi(tc, m):
      return _yi() * _fi(tc, m)


  def _yigi(tc, m, w):
      return _yi() * _gi(tc, m, w)


  def _yihi(tc, m, w):
      return _yi() * _hi(tc, m, w)

  def matrix_equation(tc, m, w):
    N = len(local_DataSeries[1])
    fi = torch.sum(_fi(tc, m))
    gi = torch.sum(_gi(tc, m, w))
    hi = torch.sum(_hi(tc, m, w))
    fi_pow_2 = torch.sum(_fi_pow_2(tc, m))
    gi_pow_2 = torch.sum(_gi_pow_2(tc, m, w))
    hi_pow_2 = torch.sum(_hi_pow_2(tc, m, w))
    figi = torch.sum(_figi(tc, m, w))
    fihi = torch.sum(_fihi(tc, m, w))
    gihi = torch.sum(_gihi(tc, m, w))

    yi = torch.sum(_yi())
    yifi = torch.sum(_yifi(tc, m))
    yigi = torch.sum(_yigi(tc, m, w))
    yihi = torch.sum(_yihi(tc, m, w))

    # Create the matrices as 2D tensors
    matrix_1 = torch.tensor([
        [N, fi, gi, hi],
        [fi, fi_pow_2, figi, fihi],
        [gi, figi, gi_pow_2, gihi],
        [hi, fihi, gihi, hi_pow_2]
    ], dtype = torch.float64)
    # Check if matrix_1 is singular
    if torch.linalg.det(matrix_1) == 0:
        # print("Matrix 1 is singular!")
        # Add regularization to avoid singularity
        regularization_strength = 1e-5
        identity_matrix = torch.eye(matrix_1.size(0), dtype=matrix_1.dtype)
        matrix_1 += regularization_strength * identity_matrix

    matrix_2 = torch.tensor([
    [yi],
    [yifi],
    [yigi],
    [yihi]
    ], dtype = torch.float64)
    # print('matrix2', matrix_2)
    product = torch.linalg.solve(matrix_1, matrix_2)

    return product

  # Calculate the linear values from the matrix equation
  lin_vals = matrix_equation(tc, m, w)

  a = lin_vals[0].item()
  b = lin_vals[1].item()
  c1 = lin_vals[2].item()
  c2 = lin_vals[3].item()

  return a, b, c1, c2
