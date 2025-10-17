import numpy as np
from functools import partial
import copy

def get_revenue_function_nl(data):
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
   
    return revenue_fn_cpu


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