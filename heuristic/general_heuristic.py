import numpy as np
import copy
from functools import partial
from models.mmnl_functions import expected_revenue_mmnl_cpu


def revenue_order(choice_model, data, A=None, B=None, revenue_fn_order=None):
    """
    Finds the best revenue-ordered assortment under optional linear constraints.

    Args:
        num_prod (int): Number of products.
        choice_model (str): Choice model type.
        data (object): Data object with model parameters.
        A (np.ndarray, optional): Constraint matrix.
        B (np.ndarray, optional): Constraint bounds.
        revenue_fn_order (function, optional): Revenue function with sorted price.

    Returns:
        tuple: (best_rev, k, best_ass)
            best_rev (float): Best revenue found.
            k (int): Number of products in the best assortment.
            best_ass (np.ndarray): Assortment in original order as a binary array.
    """
    if revenue_fn_order is None:
        revenue_fn_order, sorted_idx = get_function_revenue_order(choice_model, data)
    best_rev = 0.0
    k = 0
    best_ass = None
    num_prod = data.u.shape[1]
    for i in range(num_prod):
        current_ass = np.zeros((1, num_prod))
        current_ass[:, :i+1] = 1

        if A is not None and B is not None:
            lhs = current_ass @ A.T
            if np.any(lhs > B):
                continue

        cur_rev = revenue_fn_order(current_ass)[0]
        if cur_rev > best_rev:
            best_rev = cur_rev
            k = i + 1
            best_ass = current_ass[0]
    if revenue_fn_order is None:
        inverse_idx = np.argsort(sorted_idx)
        best_ass = best_ass[inverse_idx]
    return best_rev, k, best_ass


def get_function_revenue_order(choice_model, data):
    """
    Returns a revenue function with sorted inputs according to the choice model.

    Args:
        choice_model (str): The choice model type.
        data (object): Data object containing model parameters.

    Returns:
        tuple: (revenue_fn_order, sorted_idx)
            revenue_fn_order: Revenue function with sorted parameters.
            sorted_idx: Indices for sorting products by descending price.

    Raises:
        ValueError: If the choice model is unsupported.
    """
    if choice_model == 'mmnl':
        u = copy.copy(data.u)
        price = copy.copy(data.price)
        v0 = copy.copy(data.v0)
        omega = copy.copy(data.omega)

        if price.ndim == 1:
            price = price.reshape(1, -1)
        elif price.ndim == 2:
            pass
        else:
            raise ValueError(f"Invalid price shape: {price.shape}")
        
        sorted_idx = np.argsort(-price[0])  
        price = price[:, sorted_idx]       
        u = u[:, sorted_idx] 
        revenue_fn_order = partial(expected_revenue_mmnl_cpu, utility=u, price=price, v0=v0, omega=omega)
    else:
        raise ValueError("Unsupported choice model")
    return revenue_fn_order, sorted_idx
