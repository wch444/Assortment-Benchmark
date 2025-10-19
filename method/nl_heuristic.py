import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Union, Tuple


def revenue_order_nl(
    m: int,
    n: int,
    utility: np.ndarray,
    price: np.ndarray,
    v0: float,
    vi0: Union[np.ndarray, float],
    gamma: np.ndarray
) -> np.ndarray:
    """
    Solves the optimal nested logit assortment problem by revenue order using Gurobi.

    This function implements a revenue-ordered approach to find the optimal product 
    assortment under a nested logit choice model. Products within each nest are 
    considered in descending order of price.

    Args:
        m: Number of nests.
        n: Number of products per nest.
        utility: Utility matrix of shape (m, n).
        price: Price matrix of shape (m, n).
        v0: Outside option utility (no-purchase option).
        vi0: Outside option utility for each nest. Can be a scalar or array of shape (m,).
        gamma: Dissimilarity parameters for each nest, array of shape (m,).

    Returns:
        Optimal assortment as a binary array of shape (1, m*n).
    """
    # Sort products by descending price and track original indices
    sorted_price, sorted_utility, inverse_indices = _sort_products_by_price(price, utility)
    
    # Build and solve optimization model
    model = _build_nl_model(m, n, sorted_utility, sorted_price, v0, vi0, gamma)
    model.optimize()
    
    # Extract optimal assortment
    x_opt = model.getVarByName("x").X
    y_values = {i: model.getVarByName(f"y_{i}").X for i in range(m)}
    
    # Reconstruct assortment and restore original order
    assortment = _extract_assortment(
        m, n, sorted_utility, sorted_price, vi0, gamma, x_opt, y_values
    )
    assortment = np.take_along_axis(assortment, inverse_indices, axis=1)
    
    return assortment.reshape(1, -1)


def _sort_products_by_price(
    price: np.ndarray, 
    utility: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort products in each nest by descending price.
    
    Args:
        price: Price matrix of shape (m, n).
        utility: Utility matrix of shape (m, n).
    
    Returns:
        Tuple of (sorted_price, sorted_utility, inverse_indices).
    """
    sorted_indices = np.argsort(-price, axis=1)
    sorted_price = np.take_along_axis(price, sorted_indices, axis=1)
    sorted_utility = np.take_along_axis(utility, sorted_indices, axis=1)
    inverse_indices = np.argsort(sorted_indices, axis=1)
    
    return sorted_price, sorted_utility, inverse_indices


def _build_nl_model(
    m: int,
    n: int,
    utility: np.ndarray,
    price: np.ndarray,
    v0: float,
    vi0: Union[np.ndarray, float],
    gamma: np.ndarray
) -> gp.Model:
    """
    Build the Gurobi optimization model for nested logit assortment.
    
    Args:
        m: Number of nests.
        n: Number of products per nest.
        utility: Sorted utility matrix of shape (m, n).
        price: Sorted price matrix of shape (m, n).
        v0: Outside option utility.
        vi0: Outside option utility for each nest.
        gamma: Dissimilarity parameters.
    
    Returns:
        Configured Gurobi model.
    """
    model = gp.Model("NestedLogitAssortment")
    model.setParam("OutputFlag", 0)
    
    # Add variable x: the revenue threshold to minimize
    x = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")
    # Add variables y[i]: auxiliary variables for each nest
    y = {}
    for i in range(m):
        y[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{i}")
    
    # Main constraint: v0 * x >= sum_i y_i
    model.addConstr(
        v0 * x >= gp.quicksum(y[i] for i in range(m)),
        name="c0"
    )
    
    # Add constraints for each nest and subset
    _add_nest_constraints(model, m, n, utility, price, vi0, gamma, x, y)
    
    # Objective: minimize revenue threshold
    model.setObjective(x, GRB.MINIMIZE)
    
    return model


def _add_nest_constraints(
    model: gp.Model,
    m: int,
    n: int,
    utility: np.ndarray,
    price: np.ndarray,
    vi0: Union[np.ndarray, float],
    gamma: np.ndarray,
    x: gp.Var,
    y: gp.tupledict
):
    """
    Add constraints for all possible revenue-ordered subsets in each nest.
    
    Args:
        model: Gurobi model.
        m: Number of nests.
        n: Number of products per nest.
        utility: Utility matrix.
        price: Price matrix.
        vi0: Outside option utility for each nest.
        gamma: Dissimilarity parameters.
        x: Revenue threshold variable.
        y: Auxiliary variables for each nest.
    """
    for i in range(m):
        vi0_val = vi0[i] if isinstance(vi0, np.ndarray) else vi0
        
        for k in range(n + 1):
            vi, ri = _calculate_nest_metrics(i, k, n, utility, price, vi0_val)

            # Constraint: y_i >= V_i^gamma_i * (R_i - x)
            model.addConstr( y[i] >= (vi ** gamma[i]) * (ri - x),name=f"nest_{i}_subset_{k}")


def _extract_assortment(
    m: int,
    n: int,
    utility: np.ndarray,
    price: np.ndarray,
    vi0: Union[np.ndarray, float],
    gamma: np.ndarray,
    x_opt: float,
    y_values: dict
) -> np.ndarray:
    """
    Extract the optimal assortment from the solved model.
    
    Args:
        m: Number of nests.
        n: Number of products per nest.
        utility: Sorted utility matrix.
        price: Sorted price matrix.
        vi0: Outside option utility for each nest.
        gamma: Dissimilarity parameters.
        x_opt: Optimal revenue threshold.
        y_values: Optimal y variable values.
    
    Returns:
        Binary assortment matrix of shape (m, n).
    """
    assortment = np.zeros((m, n), dtype=int)
    tolerance = 1e-4
    
    for i in range(m):
        vi0_val = vi0[i] if isinstance(vi0, np.ndarray) else vi0
       
        min_violation = float('inf')
        best_k = 0
        
        for k in range(n + 1):
            vi, ri = _calculate_nest_metrics(i, k, n, utility, price, vi0_val)
            
            # Calculate constraint violation
            lhs = (vi ** gamma[i]) * (ri - x_opt)
            violation = abs(lhs - y_values[i])
            
            # Find the tightest constraint (smallest violation)
            if violation < tolerance and violation < min_violation:
                min_violation = violation
                best_k = k

        assortment[i, :best_k] = 1
    return assortment


def _calculate_nest_metrics(
    nest_idx: int,
    k: int,
    n: int,
    utility: np.ndarray,
    price: np.ndarray,
    vi0_val: float
) -> Tuple[float, float]:
    """
    Calculate the inclusive value (vi) and revenue (ri) for a subset of products.
    
    Args:
        nest_idx: Index of the nest.
        k: Number of products to include (0 to n).
        n: Total number of products per nest.
        utility: Utility matrix.
        price: Price matrix.
        vi0_val: Outside option utility for this nest.
    
    Returns:
        Tuple of (inclusive_value, revenue_per_unit).
    """
    if k == 0:
        # Empty assortment
        return vi0_val, 0.0
    
    # Select first k products (highest revenue in sorted order)
    subset_mask = np.zeros(n)
    subset_mask[:k] = 1
    
    # Calculate inclusive value
    vi = np.sum(utility[nest_idx] * subset_mask) + vi0_val
    
    # Calculate expected revenue
    ri = np.sum(price[nest_idx] * utility[nest_idx] * subset_mask) / vi
    
    return vi, ri



# import numpy as np
# import gurobipy as gp
# from gurobipy import GRB


# def revenue_order_nl(m, n, utility, price, v0, vi0, gamma):
#     """
#     Solves the optimal nested logit assortment problem by revenue order using Gurobi.

#     Args:
#         m (int): Number of nests.
#         n (int): Number of products per nest.
#         utility (np.ndarray): Utility matrix of shape (m, n).
#         price (np.ndarray): Price matrix of shape (m, n).
#         v0 (float): Outside option utility.
#         vi0 (np.ndarray or float): Outside option utility for each nest.
#         gamma (np.ndarray): Dissimilarity parameters for each nest.

#     Returns:
#         np.ndarray: Optimal assortment as a binary array of shape (1, n).
#     """
#     # Sort products in each nest by descending price
#     sorted_indices = np.argsort(-price, axis=1)
#     price = np.take_along_axis(price, sorted_indices, axis=1)
#     utility = np.take_along_axis(utility, sorted_indices, axis=1)
#     inverse_indices = np.argsort(sorted_indices, axis=1)  # 用于还原的索引

#     model = gp.Model("NestedLogitAssortment")
#     model.setParam("OutputFlag", 0)
#     # Add variable x: the revenue threshold to minimize
#     x = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")
#     # Add variables y[i]: auxiliary variables for each nest
#     y = {}
#     for i in range(m):
#         y[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{i}")
    
#     # Add constraint: v0 * x >= sum_i y_i
#     model.addConstr(v0 * x >= gp.quicksum(y[i] for i in range(m)), "c0")
    
#     # For each nest, add constraints for all possible revenue-ordered subsets
#     for i in range(m):
#         if isinstance(vi0, np.ndarray):
#             vi0_val = vi0[i]  
#         else:
#             vi0_val = vi0 

#         # Consider all possible subsets (including empty set) in revenue order
#         for k in range(n + 1):
#             si = np.zeros(n) 
#             if k > 0:
#                 # Select the first k products (highest revenue)
#                 si[:k] = 1
#                 vi = np.sum(utility[i] * si) + vi0_val  
#                 ri = np.sum(price[i] * utility[i] * si) / vi  
#             else:
#                 vi = vi0_val
#                 ri = 0
#             # Add constraint: y_i >= V_i^gamma_i * (R_i - x)
#             model.addConstr(y[i] >= (vi ** gamma[i]) * (ri - x), f"c_{i}_{k}")
    
#     # Set the objective to minimize x (the revenue threshold)
#     model.setObjective(x, GRB.MINIMIZE)
#     # Solve the model
#     model.optimize()
#     # print(f"Optimal objective value: {model.ObjVal}")

#     x_opt = model.getVarByName("x").X  # Get optimal threshold value
#     assortment = np.zeros((m, n))
    
#     for i in range(m):
#         # Get vi0 value for current nest
#         vi0_val = vi0[i] if isinstance(vi0, np.ndarray) else vi0
        
#         # Check each possible subset size
#         max_diff = -float('inf')
#         best_k = 0
        
#         for k in range(n + 1):
#             si = np.zeros(n)
#             if k > 0:
#                 si[:k] = 1  # First k products in revenue order
#                 vi = np.sum(utility[i] * si) + vi0_val
#                 ri = np.sum(price[i] * utility[i] * si) / vi
#             else:
#                 vi = vi0_val
#                 ri = 0
            
#             # Check if constraint is binding (approximately)
#             diff = abs((vi ** gamma[i]) * (ri - x_opt) - model.getVarByName(f"y_{i}").X)
  
#             if diff < 1e-4 and diff > max_diff:
#                 max_diff = diff
#                 best_k = k
        
#         # Set the optimal assortment for nest i
#         assortment[i, :best_k] = 1
#     assortment = np.take_along_axis(assortment, inverse_indices, axis=1)
#     assortment = assortment.reshape(1,-1)
#     return assortment
    


