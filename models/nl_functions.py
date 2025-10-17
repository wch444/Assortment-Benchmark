import torch
import numpy as np
from functools import partial
import copy

def get_revenue_function_nl(device, data):
    """
    Returns revenue functions for NL model.
    Provides both torch (GPU) and numpy (CPU) versions.

    Args:
        args: Arguments containing device information.
        data: Data object with price, v, gamma, v0, vi0

    Returns:
        revenue_fn: Callable for expected revenue using torch tensors.
        revenue_fn_cpu: Callable for expected revenue using numpy arrays.
    """
    price = copy.copy(data.price)
    v = copy.copy(data.v)
    gamma = copy.copy(data.gamma)
    v0 = copy.copy(data.v0)
    vi0 = copy.copy(data.vi0)

    # Partial function for CPU (numpy) version
    revenue_fn_cpu = partial(expected_revenue_nl_cpu, utility=v, price=price, v0=v0, vi0=vi0, gamma=gamma)
    # Move data to the specified device for torch version
    price = torch.tensor(price).to(device)
    v = torch.tensor(v).to(device)
    gamma = torch.tensor(gamma).to(device)
    if isinstance(vi0, np.ndarray):
        vi0 = torch.tensor(vi0).to(device)
    
    # Partial function for torch version
    revenue_fn = partial(expected_revenue_nl, utility=v, price=price, v0=v0, vi0=vi0, gamma=gamma)
    return revenue_fn, revenue_fn_cpu

def expected_revenue_nl(hard_solution, utility, price, v0, vi0, gamma):
    """
    Computes the expected revenue for the NL model (supports batch tensors).

    Args:
        hard_solution (torch.Tensor): Assortment solution, shape (B, m*n) .
        utility (torch.Tensor): utility matrix, shape (m, n).
        price (torch.Tensor): price matrix, shape (m, n).
        v0 (float): the baseline utility scalar.
        vi0 (float or torch.Tensor): the baseline utility scalar or vector (m,) in each nest.
        gamma (torch.Tensor): dissimilarity parameter vector, shape  (m,).

    Returns:
        expected_rev (torch.Tensor): Expected revenue, shape (B,)
    """
    if hard_solution.ndim == 1:
        hard_solution = hard_solution.unsqueeze(0) 
    B, _ = hard_solution.shape
    m, n = utility.shape
    hard_solution = hard_solution.reshape(B, m, n)
    utility = utility.unsqueeze(0)
    price = price.unsqueeze(0)
    gamma = gamma.unsqueeze(0)

    # Compute nest utility sums
    V_nest = torch.sum(hard_solution * utility, dim=2) + vi0  # (B, m)
    V_gamma = V_nest ** gamma
    # Compute nest choice probabilities
    prob_nest = V_gamma / (v0 + torch.sum(V_gamma, dim=1, keepdim=True))  # (B, m)
    # Compute revenue per nest
    numerator = torch.sum(price * utility * hard_solution, dim=2)
    rev_nest = numerator / V_nest
    # Compute expected revenue
    expected_rev = torch.sum(prob_nest * rev_nest, dim=1)
    return expected_rev

def expected_revenue_nl_cpu(hard_solution, utility, price, v0, vi0, gamma):
    """
    Computes the expected revenue for the NL model (supports numpy arrays).

    Args:
        hard_solution (np.ndarray): Assortment solution, shape (B, m*n) .
        utility (np.ndarray): utility matrix, shape (m, n).
        price (np.ndarray): price matrix, shape (m, n).
        v0 (float): the baseline utility scalar.
        vi0 (float or np.ndarray): the baseline utility scalar or vector (m,) in each nest.
        gamma (np.ndarray): dissimilarity parameter vector, shape  (m,).

    Returns:
        expected_rev (np.ndarray): Expected revenue, shape (B,)
    """
    if hard_solution.ndim == 1:
        hard_solution = hard_solution.reshape(1, -1)
    B, _ = hard_solution.shape
    m, n = utility.shape
    hard_solution = hard_solution.reshape(B, m, n)

    # Convert torch tensors to numpy arrays if necessary
    if isinstance(utility, torch.Tensor):
        utility = utility.cpu().numpy()
    if isinstance(price, torch.Tensor):
        price = price.cpu().numpy()
    if isinstance(v0, torch.Tensor):
        v0 = v0.cpu().numpy()
    if isinstance(vi0, torch.Tensor):
        vi0 = vi0.cpu().numpy()
    if isinstance(gamma, torch.Tensor):
        gamma = gamma.cpu().numpy()
    if isinstance(hard_solution, torch.Tensor):
        hard_solution = hard_solution.cpu().numpy()

    utility = np.expand_dims(utility, axis=0)
    price = np.expand_dims(price, axis=0)
    gamma = np.expand_dims(gamma, axis=0)

    V_nest = np.sum(hard_solution * utility, axis=2) + vi0  # (B, m)
    V_gamma = V_nest ** gamma
    prob_nest = V_gamma / (v0 + np.sum(V_gamma, axis=1, keepdims=True))  # (B, m)
    numerator = np.sum(price * utility * hard_solution, axis=2)
    rev_nest = numerator / V_nest
    expected_rev = np.sum(prob_nest * rev_nest, axis=1)
    return expected_rev