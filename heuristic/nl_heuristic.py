import numpy as np
import gurobipy as gp
from gurobipy import GRB
import copy
import time


def revenue_order_nl(m, n, utility, price, v0, vi0, gamma):
    """
    Solves the optimal nested logit assortment problem by revenue order using Gurobi.

    Args:
        m (int): Number of nests.
        n (int): Number of products per nest.
        utility (np.ndarray): Utility matrix of shape (m, n).
        price (np.ndarray): Price matrix of shape (m, n).
        v0 (float): Outside option utility.
        vi0 (np.ndarray or float): Outside option utility for each nest.
        gamma (np.ndarray): Dissimilarity parameters for each nest.

    Returns:
        np.ndarray: Optimal assortment as a binary array of shape (1, n).
    """
    # Sort products in each nest by descending price
    sorted_indices = np.argsort(-price, axis=1)
    price = np.take_along_axis(price, sorted_indices, axis=1)
    utility = np.take_along_axis(utility, sorted_indices, axis=1)
    inverse_indices = np.argsort(sorted_indices, axis=1)  # 用于还原的索引

    model = gp.Model("NestedLogitAssortment")
    model.setParam("OutputFlag", 0)
    # Add variable x: the revenue threshold to minimize
    x = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")
    # Add variables y[i]: auxiliary variables for each nest
    y = {}
    for i in range(m):
        y[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{i}")
    
    # Add constraint: v0 * x >= sum_i y_i
    model.addConstr(v0 * x >= gp.quicksum(y[i] for i in range(m)), "c0")
    
    # For each nest, add constraints for all possible revenue-ordered subsets
    for i in range(m):
        if isinstance(vi0, np.ndarray):
            vi0_val = vi0[i]  
        else:
            vi0_val = vi0 

        # Consider all possible subsets (including empty set) in revenue order
        for k in range(n + 1):
            si = np.zeros(n) 
            if k > 0:
                # Select the first k products (highest revenue)
                si[:k] = 1
                vi = np.sum(utility[i] * si) + vi0_val  
                ri = np.sum(price[i] * utility[i] * si) / vi  
            else:
                vi = vi0_val
                ri = 0
            # Add constraint: y_i >= V_i^gamma_i * (R_i - x)
            model.addConstr(y[i] >= (vi ** gamma[i]) * (ri - x), f"c_{i}_{k}")
    
    # Set the objective to minimize x (the revenue threshold)
    model.setObjective(x, GRB.MINIMIZE)
    # Solve the model
    model.optimize()
    # print(f"Optimal objective value: {model.ObjVal}")

    x_opt = model.getVarByName("x").X  # Get optimal threshold value
    assortment = np.zeros((m, n))
    
    for i in range(m):
        # Get vi0 value for current nest
        vi0_val = vi0[i] if isinstance(vi0, np.ndarray) else vi0
        
        # Check each possible subset size
        max_diff = -float('inf')
        best_k = 0
        
        for k in range(n + 1):
            si = np.zeros(n)
            if k > 0:
                si[:k] = 1  # First k products in revenue order
                vi = np.sum(utility[i] * si) + vi0_val
                ri = np.sum(price[i] * utility[i] * si) / vi
            else:
                vi = vi0_val
                ri = 0
            
            # Check if constraint is binding (approximately)
            diff = abs((vi ** gamma[i]) * (ri - x_opt) - model.getVarByName(f"y_{i}").X)
  
            if diff < 1e-4 and diff > max_diff:
                max_diff = diff
                best_k = k
        
        # Set the optimal assortment for nest i
        assortment[i, :best_k] = 1
    assortment = np.take_along_axis(assortment, inverse_indices, axis=1)
    assortment = assortment.reshape(1,-1)
    return assortment
    


