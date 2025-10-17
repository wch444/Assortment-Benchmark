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

# def get_revenue_function_mmnl_batch(args, data, batch_size=1024):
#     device = args.device
#     u = copy.copy(data.u)
#     price = copy.copy(data.p)
#     v0 = copy.copy(data.v0)
#     omega = copy.copy(data.omega)
#     revenue_fn = partial(
#         expected_revenue_mmnl_cpu_batch,
#         utility=u,
#         price=price,
#         v0=v0,
#         omega=omega,
#         batch_size=batch_size)  

#     return revenue_fn

# def expected_revenue_mmnl_cpu_batch(
#     hard_solution: np.ndarray,
#     utility: np.ndarray,
#     price: np.ndarray,
#     v0: Union[float, np.ndarray],
#     omega: Union[float, np.ndarray],
#     batch_size: int = 1024
# ) -> np.ndarray:
#     """
#     Compute expected revenue for MMNL model with batched processing to save memory.
    
#     Parameters:
#         hard_solution (np.ndarray): Assortment solutions, shape (B, n)
#         utility (np.ndarray): Utility matrix, shape (m, n)
#         price (np.ndarray or scalar): Price vector, shape (n,) or scalar
#         v0 (float or np.ndarray): Baseline utility, shape (m,) or scalar
#         omega (float or np.ndarray): Choice weights, shape (m,) or scalar
#         batch_size (int): Number of solutions per batch

#     Returns:
#         np.ndarray: Negative expected revenue, shape (B,)
#     """
#     # Input validation
#     assert hard_solution.ndim == 2, f"hard_solution must be 2D (B, n), got {hard_solution.shape}"
#     B, n = hard_solution.shape

#     assert utility.ndim == 2, f"utility must be 2D (m, n), got {utility.shape}"
#     m, n_util = utility.shape
#     assert n_util == n, f"n mismatch: utility {n_util} vs solution {n}"

#     # Normalize price
#     if isinstance(price, (int, float)) or np.isscalar(price):
#         price = np.full(n, float(price))
#     else:
#         price = np.asarray(price).flatten()
#         assert price.size == n, f"price size {price.size} != n={n}"
#         if price.ndim == 0:  # 0-dim scalar array
#             price = np.full(n, price.item())

#     # Normalize v0
#     v0 = np.asarray(v0)
#     if v0.size == 1:
#         v0_val = v0.item() if v0.ndim == 0 else v0[0]
#         v0 = np.full(m, v0_val)
#     else:
#         v0 = v0.flatten()
#         assert v0.size == m, f"v0 size {v0.size} != m={m}"

#     # Normalize omega
#     omega = np.asarray(omega)
#     if omega.size == 1:
#         omega_val = omega.item() if omega.ndim == 0 else omega[0]
#         omega = np.full(m, omega_val)
#     else:
#         omega = omega.flatten()
#         assert omega.size == m, f"omega size {omega.size} != m={m}"

#     # Pre-compute expanded shapes
#     v0_exp = v0.reshape(m, 1, 1)          # (m, 1, 1)
#     price_exp = price.reshape(1, 1, n)    # (1, 1, n)
#     omega = omega.reshape(1, m)           # (1, m)

#     # Output array: one revenue per solution
#     revenues = np.zeros(B, dtype=np.float64)  # (B,)

#     # Process in batches
#     for start in range(0, B, batch_size):
#         end = min(start + batch_size, B)
#         batch_sol = hard_solution[start:end]  # (b, n)
#         b = batch_sol.shape[0]

#         # Expand solution: (b, n) -> (b, 1, n)
#         sol_exp = np.expand_dims(batch_sol, axis=1)  # (b, 1, n)

#         # Utility in assortment: (m, n) * (b, 1, n) -> (b, m, n)
#         util_in = utility[np.newaxis, :, :] * sol_exp  # broadcasting

#         # Denominator: v0 + sum_n(util_in) -> (b, m, 1)
#         denom = v0_exp + np.sum(util_in, axis=2, keepdims=True)

#         # Choice probability: (b, m, n)
#         prob = util_in / denom

#         # Revenue per segment: sum_n(prob * price) -> (b, m)
#         rev_per_segment = np.sum(prob * price_exp, axis=2)  # (b, m)

#         # Total revenue: sum_m(rev_per_segment * omega) -> (b,)
#         batch_revenue = np.sum(rev_per_segment * omega, axis=1)  # (b,)

#         # Assign to output (b,)
#         revenues[start:end] = batch_revenue  # ✅ shape matches: (b,) -> (B,) slice

#     return -revenues  # (B,)