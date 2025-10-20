import numpy as np
import gurobipy as gp
import time
from typing import Optional, Tuple, Dict, Union


# def your_mmnl_algorithm(m, n, u, price, v0, omega, constraint=None):
#     """
#     Your custom algorithm for MMNL assortment optimization
    
#     Args:
#         m (int): Number of customer segments
#         n (int): Number of products
#         u (np.ndarray): Utility matrix of shape (m, n)
#         price (np.ndarray): Product prices of shape (n,)
#         v0 (np.ndarray): No-purchase utilities of shape (m,)
#         omega (np.ndarray): Segment weights of shape (m,), sum(omega) = 1
#         constraint (optional): Linear constraint (A, B) where A @ x <= B
    
#     Returns:
#         np.ndarray: Binary assortment vector of shape (n,)
#     """
#     # Your implementation here
#     assortment = ...  # Shape: (n,)
#     return assortment




def conic_mmnl_warm_start(
    coef: np.ndarray,
    uti: np.ndarray,
    v_i0: Union[np.ndarray, float],
    p: np.ndarray,
    cardinality: int,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    time_limit: int = 600,
    x0: Optional[Union[list, np.ndarray]] = None
) -> Tuple[float, Optional[np.ndarray], Dict]:
    """
    Solve the assortment optimization problem for a Mixed Multinomial Logit (MMNL) model
    using a conic reformulation with optional warm start and additional linear constraints.

    Args:
        coef: Segment weights of shape (m,).
        uti: Utility matrix of shape (m, n) where m is number of segments, n is number of products.
        v_i0: Outside option utility for each segment. Scalar or array of shape (m,).
        p: Price vector of shape (n,) or (1, n).
        cardinality: Maximum number of products allowed in the assortment.
        A: Optional linear constraint matrix of shape (n_constr, n).
        b: Optional right-hand side vector of shape (n_constr,).
        time_limit: Time limit for solving in seconds. Default is 600.
        x0: Optional warm start initial solution. Either:
            - A list of product indices (0-based), e.g., [0, 3, 5].
            - A binary vector of length n, e.g., [1, 0, 0, 1, ...].

    Returns:
        opt_rev: Estimated optimal (or best feasible) revenue.
        ass: Optimal assortment as a binary array of shape (n,).
        info: Solver statistics including status, gap, solving time, and bounds.
    """
    start_time = time.time()

    # Input validation and standardization
    uti = np.asarray(uti, dtype=float)
    num_segments, num_products = uti.shape

    # Process price vector
    prices = _standardize_prices(p, num_products)
    max_price = float(np.max(prices))

    # Process segment coefficients
    coef = _standardize_array(coef, num_segments, "coef")

    # Process outside option utilities
    v_i0 = _standardize_outside_utility(v_i0, num_segments)

    # Build and configure Gurobi model
    model = _build_gurobi_model(
        num_segments, num_products, coef, uti, v_i0, prices, max_price,
        cardinality, A, b, time_limit
    )

    # Apply warm start if provided
    if x0 is not None:
        _apply_warm_start(model, x0, num_products, cardinality)

    # Solve the model
    model.optimize()
    
    # Extract and return results
    return _extract_results(model, uti, prices, v_i0, coef, num_products, start_time)


def _standardize_prices(p: np.ndarray, num_products: int) -> np.ndarray:
    """Standardize price vector to 1D array."""
    prices = np.asarray(p).reshape(-1)
    
    if prices.size == num_products:
        return prices
    elif prices.ndim == 1 and prices.size == 1 and hasattr(p, "__getitem__"):
        return np.asarray(p[0]).reshape(-1)
    else:
        raise ValueError(f"Price vector p must have shape ({num_products},) or (1, {num_products}).")


def _standardize_array(arr: np.ndarray, expected_size: int, name: str) -> np.ndarray:
    """Standardize array to expected size."""
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size != expected_size:
        arr = np.resize(arr, expected_size)
    return arr


def _standardize_outside_utility(v_i0: Union[np.ndarray, float], num_segments: int) -> np.ndarray:
    """Standardize outside utility to array of shape (num_segments,)."""
    v_i0 = np.asarray(v_i0, dtype=float).reshape(-1)
    
    if v_i0.size == 1:
        return np.full(num_segments, v_i0.item(), dtype=float)
    elif v_i0.size != num_segments:
        return np.resize(v_i0, num_segments)
    return v_i0


def _build_gurobi_model(
    num_segments: int, num_products: int, coef: np.ndarray, uti: np.ndarray,
    v_i0: np.ndarray, prices: np.ndarray, max_price: float,
    cardinality: int, A: Optional[np.ndarray], b: Optional[np.ndarray],
    time_limit: int
) -> gp.Model:
    """Build and configure the Gurobi optimization model."""
    model = gp.Model('ConicMMNL')
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    # model.Params.NonConvex = 2  # Allow non-convex (bilinear) constraints

    # Decision variables
    x = model.addVars(num_products, vtype=gp.GRB.BINARY, name='x')
    y = model.addVars(num_segments, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='y')
    z = model.addVars(num_segments * num_products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='z')
    w = model.addVars(num_segments, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='w')

    # Objective function
    obj = gp.quicksum(coef[i] * v_i0[i] * max_price * y[i] for i in range(num_segments))
    obj += gp.quicksum(
        coef[i] * uti[i, j] * (max_price - prices[j]) * z[i * num_products + j]
        for i in range(num_segments) for j in range(num_products)
    )
    model.setObjective(obj, gp.GRB.MINIMIZE)

    # Add constraints
    _add_basic_constraints(model, x, num_products, cardinality)
    _add_mmnl_constraints(model, x, y, z, w, num_segments, num_products, uti, v_i0, cardinality)
    
    if A is not None and b is not None:
        _add_linear_constraints(model, x, A, b, num_products)

    return model


def _add_basic_constraints(model: gp.Model, x: gp.tupledict, num_products: int, cardinality: int):
    """Add basic constraints to the model."""
    model.addConstr(
        gp.quicksum(x[j] for j in range(num_products)) >= 1,
        name='avoid_empty_assortment'
    )
    
    if cardinality is not None:
        model.addConstr(
            gp.quicksum(x[j] for j in range(num_products)) <= int(cardinality),
            name='cardinality'
        )


def _add_mmnl_constraints(
    model: gp.Model, x: gp.tupledict, y: gp.tupledict, z: gp.tupledict, w: gp.tupledict,
    num_segments: int, num_products: int, uti: np.ndarray, v_i0: np.ndarray, cardinality: int
):
    """Add MMNL-specific constraints to the model."""
    # Define w variables
    model.addConstrs(
        (w[i] == v_i0[i] + gp.quicksum(uti[i, j] * x[j] for j in range(num_products))
         for i in range(num_segments)),
        name='w_definition'
    )

    # Bilinear constraints
    model.addConstrs(
        (z[i * num_products + j] * w[i] >= x[j] * x[j]
         for i in range(num_segments) for j in range(num_products)),
        name='bilinear_zw'
    )

    model.addConstrs(
        (y[i] * w[i] >= 1 for i in range(num_segments)),
        name='reciprocal_y'
    )

    # Probability constraints
    model.addConstrs(
        (v_i0[i] * y[i] + gp.quicksum(uti[i, j] * z[i * num_products + j] for j in range(num_products)) >= 1
         for i in range(num_segments)),
        name='probability_mass'
    )

    # Upper bounds on z
    model.addConstrs(
        (z[i * num_products + j] <= (1.0 / max(v_i0[i] + uti[i, j], 1e-12)) * x[j]
         for i in range(num_segments) for j in range(num_products)),
        name='z_upper_bound'
    )

    # Refined bounds using top-k utilities
    _add_refined_z_bounds(model, z, y, x, num_segments, num_products, uti, v_i0, cardinality)


def _add_refined_z_bounds(
    model: gp.Model, z: gp.tupledict, y: gp.tupledict, x: gp.tupledict,
    num_segments: int, num_products: int, uti: np.ndarray, v_i0: np.ndarray, cardinality: int
):
    """Add refined upper and lower bounds on z using top-k utilities."""
    for i in range(num_segments):
        for j in range(num_products):
            others = np.append(uti[i, :j], uti[i, (j + 1):])
            
            if cardinality is not None:
                top_k_minus1_sum = np.sum(np.sort(others)[::-1][:max(int(cardinality) - 1, 0)])
                top_k_sum = np.sum(np.sort(others)[::-1][:max(int(cardinality), 0)])
            else:
                top_k_minus1_sum = np.sum(np.maximum(others, 0))
                top_k_sum = np.sum(np.maximum(others, 0))

            idx = i * num_products + j
            
            # Upper bound
            denom_upper = max(v_i0[i] + top_k_sum, 1e-12)
            model.addConstr(
                z[idx] <= y[i] - (1.0 / denom_upper) * (1 - x[j]),
                name=f"z_refined_upper[{i},{j}]"
            )

            # Lower bound
            denom_lower = max(v_i0[i], 1e-12)
            model.addConstr(
                z[idx] >= y[i] - (1.0 / denom_lower) * (1 - x[j]),
                name=f"z_refined_lower[{i},{j}]"
            )


def _add_linear_constraints(
    model: gp.Model, x: gp.tupledict, A: np.ndarray, b: np.ndarray, num_products: int
):
    """Add linear constraints Ax <= b to the model."""
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    
    if A.shape[1] != num_products:
        raise ValueError(f"A must have {num_products} columns, got {A.shape[1]}")
    if A.shape[0] != b.size:
        raise ValueError("Number of rows in A must match length of b")
    
    for i in range(A.shape[0]):
        model.addConstr(
            gp.quicksum(A[i, j] * x[j] for j in range(num_products)) <= b[i],
            name=f"linear_constraint[{i}]"
        )


def _apply_warm_start(model: gp.Model, x0: Union[list, np.ndarray], num_products: int, cardinality: int):
    """Apply warm start to the model."""
    x0_bin = _coerce_initial_x(x0, num_products, cardinality)
    if x0_bin is not None:
        x_vars = model.getVars()[:num_products]
        for j, var in enumerate(x_vars):
            var.Start = int(x0_bin[j])


def _extract_results(
    model: gp.Model, uti: np.ndarray, prices: np.ndarray, v_i0: np.ndarray,
    coef: np.ndarray, num_products: int, start_time: float
) -> Tuple[Optional[float], Optional[np.ndarray], Dict]:
    """Extract and format optimization results."""
    solving_time = time.time() - start_time
    best_bound = model.ObjBound if model.SolCount > 0 else None
    status = model.Status
    nodes = model.NodeCount

    # Optional: root relaxation bound
    z_root = None

    info = {
        "status": status,
        "nodes": nodes,
        "time": solving_time,
        "best_bound": best_bound,
        "z_root": z_root
    }

    # Extract solution if available
    if status == gp.GRB.OPTIMAL or status in (gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED):
        if model.SolCount > 0:
            x_vars = model.getVars()[:num_products]
            x_vals = np.array([var.X for var in x_vars])
            ass = (x_vals > 0.5).astype(int)

            # Calculate gaps
            end_gap = _calculate_gap(model.ObjVal, best_bound) if best_bound is not None else None
            root_gap = _calculate_gap(model.ObjVal, z_root) if z_root is not None else None
            
            info.update({
                "egap": end_gap,
                "rgap": root_gap,
                "z_opt": model.ObjVal
            })

            opt_rev = mmnl_rev(ass, uti, prices, v_i0, coef)
            return opt_rev, ass, info

    # No feasible solution found
    info.update({"egap": None, "rgap": None, "z_opt": None})
    return best_bound, None, info


def _calculate_gap(obj_val: float, bound: float) -> float:
    """Calculate optimality gap percentage."""
    if obj_val == 0:
        return None
    return abs(obj_val - bound) / max(abs(obj_val), 1e-12) * 100.0


def mmnl_rev(
    ass: np.ndarray,
    u: np.ndarray,
    prices: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray
) -> float:
    """
    Compute the expected revenue under the MMNL model for a given assortment.
    
    Args:
        ass: Binary vector of length n indicating which products are included.
        u: Utility matrix of shape (m, n), where m is segments and n is products.
        prices: Price array of shape (n,) or (n, 1).
        v0: Outside utility, scalar or array of shape (m,).
        omega: Segment weights or probabilities, array of shape (m,).
    
    Returns:
        Expected revenue for the given assortment.
    """
    ass = np.asarray(ass)
    assort = np.where(ass > 0)[0]
    
    num_segments = u.shape[0]
    
    # Standardize v0
    v0 = np.asarray(v0)
    if v0.size == 1:
        v0 = np.full(num_segments, v0.item())
    elif v0.size != num_segments:
        raise ValueError(f"v0 must be scalar or have length {num_segments}, got {v0.size}")
    
    # Validate omega
    if omega.size != num_segments:
        raise ValueError(f"omega must have length {num_segments}, got {omega.size}")
    
    # Extract selected prices
    selected_prices = prices[assort] if prices.ndim == 1 else prices[0, assort]

    # Calculate choice probabilities and revenue
    util_sel = u[:, assort]
    total_util = v0 + util_sel.sum(axis=1)
    choice_prob = util_sel / total_util[:, None]
    revenue = (omega[:, None] * choice_prob * selected_prices).sum()
    
    return revenue


def _coerce_initial_x(
    x0: Union[list, np.ndarray],
    n: int,
    cardinality: Optional[int]
) -> Optional[np.ndarray]:
    """
    Convert x0 to a binary numpy vector of length n.
    
    Args:
        x0: Initial solution, can be index list or binary vector.
        n: Number of products.
        cardinality: Maximum number of products to select.
    
    Returns:
        Binary vector of length n, or None if x0 is None.
    """
    if x0 is None:
        return None

    x0_arr = np.asarray(x0)
    
    # Handle binary vector input
    if x0_arr.ndim == 1 and x0_arr.size == n:
        x_bin = (x0_arr > 0.5).astype(int)
        sel_idx = np.flatnonzero(x_bin)
    else:
        # Handle index list input
        sel_idx = np.unique(x0_arr.astype(int))
        if np.any(sel_idx < 0) or np.any(sel_idx >= n):
            raise ValueError(
                f"x0 indices out of bounds. Allowed range: [0, {n-1}], received: {sel_idx}"
            )

    # Enforce cardinality constraint
    if cardinality is not None and len(sel_idx) > cardinality:
        sel_idx = sel_idx[:cardinality]
    
    # Ensure at least one product is selected
    if len(sel_idx) == 0:
        sel_idx = np.array([0], dtype=int)

    # Create binary vector
    x_bin = np.zeros(n, dtype=int)
    x_bin[sel_idx] = 1
    
    return x_bin


