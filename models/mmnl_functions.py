import torch
import numpy as np
from functools import partial
import copy
from typing import Union

def get_revenue_function_mmnl(device, data):
    """
    Prepare revenue computation functions for the MMNL model in both CPU (NumPy) and GPU (PyTorch) versions.

    Args:
        device (torch.device): The target device for the PyTorch tensors (e.g., torch.device('cpu') or torch.device('cuda')).
        data (object): An object containing MMNL model parameters:
            - data.u (np.ndarray): Utility matrix, shape (m, n) 
            - data.p (np.ndarray): Price vector, shape (1, n) 
            - data.v0 (np.ndarray): Baseline utility, shape (m,) 
            - data.omega (np.ndarray): Choice probability vector, shape (m,) 

    Returns:
        tuple:
            - revenue_fn (function): A partial function for computing expected revenue using PyTorch tensors on the specified device.
            - revenue_fn_cpu (function): A partial function for computing expected revenue using NumPy arrays on CPU.
    """
    u = copy.copy(data.u)
    price = copy.copy(data.price)
    v0 = copy.copy(data.v0)
    omega = copy.copy(data.omega)
    revenue_fn_cpu = partial(expected_revenue_mmnl_cpu, utility=u, price=price, v0=v0, omega=omega)
    u = torch.tensor(u).to(device)
    price = torch.tensor(price).to(device)
    v0 = torch.tensor(v0).to(device)
    omega = torch.tensor(omega).to(device)
    revenue_fn = partial(expected_revenue_mmnl, utility=u, price=price, v0=v0, omega=omega)

    return revenue_fn, revenue_fn_cpu

def expected_revenue_mmnl(hard_solution, utility, price, v0, omega):
    """
    Compute the expected revenue for the MMNL model (supports both batch and non-batch torch tensors).

    Args:
        hard_solution (torch.Tensor): Assortment solution, shape (B, n) or (n,)
        utility (torch.Tensor): Utility matrix, shape (B, m, n) or (m, n)
        price (torch.Tensor): Price vector, shape (B, 1, n) or (1, n)
        v0 (torch.Tensor): Baseline utility, shape (B, m) or (m,)
        omega (torch.Tensor): Choice probability vector, shape (B, m) or (m,)

    Returns:
        total_profit (torch.Tensor): Expected revenue, shape (B,)
    """
    if utility.dim() == 2:
        utility = utility.unsqueeze(0)
    if v0.dim() == 1:
        v0 = v0.unsqueeze(0)
    if omega.dim() == 1:
        omega = omega.unsqueeze(0)
    if price.dim() == 2:
        price = price.unsqueeze(0)
    if hard_solution.dim() == 1:
        hard_solution = hard_solution.unsqueeze(0)

    sol_exp = hard_solution.unsqueeze(1)  # (B, 1, n)
    v0_exp = v0.unsqueeze(2)              # (B, m, 1)
    denom = v0_exp + torch.sum(utility * sol_exp, dim=2, keepdim=True)  # (B, m, 1)
    prob = utility * sol_exp / denom  # (B, m, n)
    revenue = torch.sum(prob * price, dim=2)  # (B, m)
    total_profit = torch.sum(revenue * omega, dim=1)  # (B,)
    return total_profit

def expected_revenue_mmnl_cpu(hard_solution, utility, price, v0, omega):
    """
    Compute the expected revenue for the MMNL model (supports numpy arrays).

    Args:
        hard_solution (np.ndarray): Assortment solution, shape (B, n) or (n,)
        utility (np.ndarray): Utility matrix, shape (B, m, n) or (m, n)
        price (np.ndarray): Price vector, shape (B, 1, n) or (1, n)
        v0 (np.ndarray): Baseline utility, shape (B, m) or (m,)
        omega (np.ndarray): Choice probability vector, shape (B, m) or (m,)

    Returns:
        total_profit (np.ndarray ): Expected revenue, shape (B,) 
    """
    hard_solution = np.asarray(hard_solution)
    if utility.ndim == 2:
        utility = np.expand_dims(utility, axis=0)
    if v0.ndim == 1:
        v0 = np.expand_dims(v0, axis=0)
    if omega.ndim == 1:
        omega = np.expand_dims(omega, axis=0)
    if price.ndim == 2:
        price = np.expand_dims(price, axis=0)
    if hard_solution.ndim == 1:
        hard_solution = np.expand_dims(hard_solution, axis=0)

    sol_exp = np.expand_dims(hard_solution, axis=1)  # (B, 1, n)
    v0_exp = np.expand_dims(v0, axis=2)              # (B, m, 1)
    denom = v0_exp + np.sum(utility * sol_exp, axis=2, keepdims=True)  # (B, m, 1)
    prob = (utility * sol_exp) / denom  # (B, m, n)
    revenue = np.sum(prob * price, axis=2)  # (B, m)
    total_profit = np.sum(revenue * omega, axis=1)  # (B,)
    return total_profit

