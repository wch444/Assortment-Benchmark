import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time,pdb





def Conic_mmnl_warm_start(
    coef, uti, v_i0, p, cardinality, A=None, b=None, time_limit=600,
    x0=None):
    """
    Solve the assortment optimization problem for a Mixed Multinomial Logit (MMNL) model
    using a conic reformulation, with optional warm start and additional linear constraints.

    Args:
        coef (np.ndarray): Segment weights of shape (m,).
        uti (np.ndarray): Utility matrix of shape (m, n).
        v_i0 (np.ndarray or float): Outside option utility for each segment,  or a scalar applied to all segments.
        p (np.ndarray): Price vector of shape (n,) or (1, n).
        cardinality (int): Maximum number of products allowed in the assortment.
        A (np.ndarray, optional): Linear constraint matrix of shape (n_constr, n).
        b (np.ndarray, optional): Right-hand side vector of shape (n_constr,).
        time_limit (int, optional): Time limit for solving in seconds. Default is 600.
        x0 (list[int] or np.ndarray, optional): Warm start initial solution. Either:
            - A list of product indices (0-based), e.g. [0, 3, 5].
            - A binary vector of length n, e.g. [1, 0, 0, 1, ...].

    Returns:
        float: Estimated optimal (or best feasible) revenue.
        np.ndarray: Optimal assortment as a binary array of shape (n,).
        dict: Solver statistics including status, gap, solving time, and bounds.
    """
    start = time.time()

    # Standardize inputs
    uti = np.asarray(uti, dtype=float)
    num_of_segments, num_of_products = uti.shape

    prices = np.asarray(p).reshape(-1)
    if prices.size == num_of_products:
        pass
    elif prices.ndim == 1 and prices.size == 1 and hasattr(p, "__getitem__"):
        prices = np.asarray(p[0]).reshape(-1)
    else:
        raise ValueError("Price vector p must have shape (n,) or (1, n).")

    max_price = float(np.max(prices))

    coef = np.asarray(coef, dtype=float).reshape(-1)
    if coef.size != num_of_segments:
        coef = np.resize(coef, num_of_segments)

    v_i0 = np.asarray(v_i0, dtype=float).reshape(-1)
    if v_i0.size != num_of_segments:
        if v_i0.size == 1:
            v_i0 = np.full(num_of_segments, v_i0.item(), dtype=float)
        else:
            v_i0 = np.resize(v_i0, num_of_segments)

    # Build Gurobi model
    m = gp.Model('ConicMMNL')
    # m.setParam('MIPFocus', 2)
    # m.setParam('MIPGap', 0.015)
    # m.setParam('Heuristics', 0)
    # m.setParam('Cuts', 3)
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    # allow non-convex (bilinear) constraints
    m.Params.NonConvex = 2

    # Decision variables
    x = m.addVars(num_of_products, vtype=gp.GRB.BINARY, name='x')
    y = m.addVars(num_of_segments, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='y')
    z = m.addVars(num_of_segments * num_of_products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='z')
    w = m.addVars(num_of_segments, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='w')

    # Objective: equivalent minimization form
    obj = gp.quicksum(coef[i] * v_i0[i] * max_price * y[i] for i in range(num_of_segments))
    obj += gp.quicksum(
        coef[i] * uti[i, j] * (max_price - prices[j]) * z[i * num_of_products + j]
        for i in range(num_of_segments) for j in range(num_of_products)
    )
    m.setObjective(obj, gp.GRB.MINIMIZE)

    # Basic constraints
    m.addConstr(gp.quicksum(x[j] for j in range(num_of_products)) >= 1, name='avoid_0')
    if cardinality is not None:
        m.addConstr(gp.quicksum(x[j] for j in range(num_of_products)) <= int(cardinality), name='cardinality')

    # MMNL constraints
    m.addConstrs((w[i] == v_i0[i] + gp.quicksum(uti[i, j] * x[j] for j in range(num_of_products))
                  for i in range(num_of_segments)), name='w_def')

    m.addConstrs((z[i * num_of_products + j] * w[i] >= x[j] * x[j]
                  for i in range(num_of_segments) for j in range(num_of_products)), name='zw_x2')

    m.addConstrs((y[i] * w[i] >= 1 for i in range(num_of_segments)), name='y_w_ge_1')

    m.addConstrs((v_i0[i] * y[i] + gp.quicksum(uti[i, j] * z[i * num_of_products + j] for j in range(num_of_products)) >= 1
                  for i in range(num_of_segments)), name='mass_1')

    m.addConstrs((z[i * num_of_products + j] <= (1.0 / (v_i0[i] + uti[i, j])) * x[j]
                  for i in range(num_of_segments) for j in range(num_of_products)), name='z_upper_local')

    # Refined upper/lower bounds on z using top-k utilities
    for i in range(num_of_segments):
        for j in range(num_of_products):
            others = np.append(uti[i, :j], uti[i, (j + 1):])
            top_k_minus1_sum = np.sum(np.sort(others)[::-1][:(max(int(cardinality) - 1, 0))]) if cardinality is not None else np.sum(np.maximum(others, 0))
            top_k_sum = np.sum(np.sort(others)[::-1][:(max(int(cardinality), 0))]) if cardinality is not None else np.sum(np.maximum(others, 0))

            idx = i * num_of_products + j
            # z <= y - (1 / (v0 + sum top_k others)) * (1 - x_j)
            denom1 = v_i0[i] + top_k_sum
            denom1 = float(max(denom1, 1e-12))
            m.addConstr(z[idx] <= y[i] - (1.0 / denom1) * (1 - x[j]), name=f"z_y_ub[{i},{j}]")

            # z >= y - (1 / v0) * (1 - x_j)
            denom2 = float(max(v_i0[i], 1e-12))
            m.addConstr(z[idx] >= y[i] - (1.0 / denom2) * (1 - x[j]), name=f"z_y_lb[{i},{j}]")

    # Linear constraints Ax <= b
    if A is not None and b is not None:
        A = np.asarray(A)
        b = np.asarray(b).reshape(-1)
        if A.shape[1] != num_of_products:
            raise ValueError(f"A must have n={num_of_products} columns, got {A.shape[1]}")
        if A.shape[0] != b.size:
            raise ValueError("Number of rows in A must match length of b.")
        for i in range(A.shape[0]):
            m.addConstr(gp.quicksum(A[i, j] * x[j] for j in range(num_of_products)) <= b[i], name=f"Axb[{i}]")

    # Warm start: initialize x with x0
    x0_bin = _coerce_initial_x(x0, num_of_products, cardinality)
    if x0_bin is not None:
        for j in range(num_of_products):
            x[j].Start = int(x0_bin[j])
        
    # Solve the model
    m.optimize()
    t = time.time() - start
    best_bound = m.ObjBound if m.SolCount >= 0 else None
    status = m.Status
    nodes = m.NodeCount

    # 根松弛（可选）
    # try:
    #     m_relax = m.relax()
    #     m_relax.Params.OutputFlag = 0
    #     m_relax.optimize()
    #     z_root = m_relax.ObjVal if m_relax.Status == gp.GRB.OPTIMAL else None
    # except gp.GurobiError:
    z_root = None # Root relaxation (optional)

    # Collect results
    info = {
        "status": status,
        "nodes": nodes,
        "time": t,
        "best_bound": best_bound,
        "z_root": z_root
    }

    if status == gp.GRB.OPTIMAL or status in (gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED):
        # 读取 x
        x_vals = np.array([x[j].X for j in range(num_of_products)])
        ass = (x_vals > 0.5).astype(int)

        # gap
        if m.SolCount > 0 and best_bound is not None and m.ObjVal != 0:
            end_gap = abs(m.ObjVal - best_bound) / max(abs(m.ObjVal), 1e-12) * 100.0
        else:
            end_gap = None
        if z_root is not None and m.SolCount > 0 and m.ObjVal != 0:
            root_gap = abs(m.ObjVal - z_root) / max(abs(m.ObjVal), 1e-12) * 100.0
        else:
            root_gap = None
        info.update({"egap": end_gap, "rgap": root_gap, "z_opt": m.ObjVal})

        opt_rev =  mmnl_rev(ass, uti, prices, v_i0, coef)
        return opt_rev, ass, info

    # No feasible solution or failure
    info.update({"egap": None, "rgap": None, "z_opt": None})
    return best_bound, None, info

def mmnl_rev(ass, u,prices, v0, omega):
    """
    Compute the expected revenue under the MMNL model for a given assortment.
    
    Parameters:
    - ass: 0/1 vector of length n, indicating which products are included
    - u:   utility matrix of shape (n, m), where row j is the utility u_{ij} for product j in each segment i
    - p:   price array of shape (n,) or (n,1)
    - v0:  outside utility, scalar or array of shape (m,)
    - omega: segment weights or probabilities, array of shape (m,)
    """
    total_profit = 0
    ass = np.array(ass)
    assort = np.where(ass>0)[0]
    m = u.shape[0]
    if v0.size == 1:
        v0 = np.full(m, v0[0])  # all segments share the same v0 value, v0 shape -> (m,)
    elif v0.size != m:
        raise ValueError(f"v0 must be scalar or have length m={m}, got {v0.size}")
    # Validate omega
    if omega.size != m:
        raise ValueError(f"omega must have length m={m}, got {omega.size}")
    
    if prices.ndim > 1:
        selected_prices = prices[0, assort]
    else:
        selected_prices = prices[assort]


    util_sel = u[:, assort]
    total_util = v0 + util_sel.sum(axis=1)
    choice_prob = util_sel / total_util[:, None]
    revenue = (omega[:, None] * choice_prob * selected_prices).sum()
    return revenue


def _coerce_initial_x(x0, n, cardinality):
    """
    Convert x0 to a 0/1 numpy vector of length n.
    
    Args:
        x0: Initial solution, can be index list or 0/1 vector.
        n (int): Number of products.
        cardinality (int): Maximum number of products to select.
    
    Returns:
        np.ndarray: Binary vector of length n.
    """
    if x0 is None:
        return None

    x0_arr = np.asarray(x0)
    if x0_arr.ndim == 1 and x0_arr.size == n:
        x_bin = (x0_arr > 0.5).astype(int)
        sel_idx = np.flatnonzero(x_bin)
    else:
        sel_idx = np.unique(x0_arr.astype(int))
        if np.any(sel_idx < 0) or np.any(sel_idx >= n):
            raise ValueError(f"x0 indices out of bounds, allowed range [0, {n-1}], received {sel_idx}")

    if cardinality is not None and len(sel_idx) > cardinality:
        sel_idx = sel_idx[:cardinality]
    if len(sel_idx) == 0:
        sel_idx = np.array([0], dtype=int)

    x_bin = np.zeros(n, dtype=int)
    x_bin[sel_idx] = 1
    return x_bin

# def mmnl_revenue(assort: np.ndarray,
#                  uti: np.ndarray,
#                  prices: np.ndarray,
#                  v0: np.ndarray,
#                  omega: np.ndarray) -> float:
#     """
#     Calculate expected revenue of a binary assortment vector.
    
#     Args:
#         assort (np.ndarray): Binary assortment vector.
#         uti (np.ndarray): Utility matrix (m, n).
#         prices (np.ndarray): Price vector.
#         v0 (np.ndarray): Outside option utilities.
#         omega (np.ndarray): Segment weights.
    
#     Returns:
#         float: Expected revenue.
#     """
#     assort = assort.astype(bool)
#     if not assort.any():
#         return 0.0
#     if prices.ndim > 1:
#         selected_prices = prices[0, assort]
#     else:
#         selected_prices = prices[assort]
#     util_sel = uti[:, assort]
#     total_util = v0 + util_sel.sum(axis=1)
#     choice_prob = util_sel / total_util[:, None]
#     revenue = (omega[:, None] * choice_prob * selected_prices).sum()
#     return float(revenue)

# def greedy_initial(prices: np.ndarray,
#                    uti: np.ndarray,
#                    omega: np.ndarray,
#                    k: int) -> np.ndarray:
#     """A simple one‑shot heuristic: pick κ products with highest
#     aggregate *expected* revenue ignoring the denominator.
#     Provides a good starting incumbent for the solver.
#     """
#     # Score_j = Σ_i γ_i ν_{ij} ρ_j
#     score = (omega[:, None] * uti * prices).sum(axis=0)
#     idx = np.argsort(score)[::-1][:k]
#     x0 = np.zeros_like(prices, dtype=int)
#     x0[idx] = 1
#     return x0

# def conic_mmnl(coef: np.ndarray,
#                uti: np.ndarray,
#                v_i0: np.ndarray,
#                prices: np.ndarray,
#                cardinality: int,
#                num_of_products: int, 
#                num_of_segments: int,
#                A: np.ndarray | None = None,
#                b: np.ndarray | None = None,
#                time_limit: int = 600,
#                threads: int = 0,
#                use_greedy_start: bool = True):
#     """
#     Solve constrained assortment optimization under MMNL using conic formulation.

#     Args:
#         coef (np.ndarray): Segment coefficients.
#         uti (np.ndarray): Utility matrix (m, n).
#         v_i0 (np.ndarray): Outside option utilities.
#         prices (np.ndarray): Price vector.
#         cardinality (int): Maximum products to select.
#         num_of_products (int): Number of products.
#         num_of_segments (int): Number of segments.
#         A (np.ndarray, optional): Constraint matrix.
#         b (np.ndarray, optional): Constraint bounds.
#         time_limit (int): Time limit in seconds.
#         threads (int): Number of threads.
#         use_greedy_start (bool): Whether to use greedy initial solution.

#     Returns:
#         tuple: (opt_revenue, assortment, info)
#             opt_revenue (float): Optimal objective value.
#             assortment (np.ndarray): 0/1 optimal assortment or None.
#             info (dict): Solution statistics.
#     """
#     start = time.time()
    
#     prices = np.asarray(prices).flatten()
#     m_seg, n_prod = num_of_segments, num_of_products
#     prices = np.asarray(prices, dtype=float)

#     # Ensure input shapes are correct
#     coef = np.asarray(coef, dtype=float).reshape(-1)
#     v_i0 = np.asarray(v_i0, dtype=float).reshape(-1)
#     if coef.size != num_of_segments:
#         coef = np.resize(coef, num_of_segments)
#     if v_i0.size != num_of_segments:
#         v_i0 = np.resize(v_i0, num_of_segments)
    
#     # Compute bounds
#     y_u = 1.0 / v_i0
#     y_l = np.empty_like(v_i0)
#     for i in range(m_seg):
#         top_k = np.partition(uti[i], -cardinality)[-cardinality:]
#         y_l[i] = 1.0 / (v_i0[i] + top_k.sum())

#     rho_bar = prices.max()

#     # Build model
#     model = gp.Model("MMNL_CONIC_MC")
#     model.setParam("OutputFlag", 0)
#     model.Params.OutputFlag = 0
#     model.Params.TimeLimit = time_limit
#     if threads:
#         model.Params.Threads = threads
#     model.Params.Presolve = 2
#     model.Params.Cuts = 2
#     model.Params.MIPFocus = 1

#     # Variables
#     x = model.addVars(n_prod, vtype=GRB.BINARY, name="x")
#     y = model.addVars(m_seg, lb=y_l.tolist(), ub=y_u.tolist(), name="y")
#     z = model.addVars([(i, j) for i in range(m_seg) for j in range(n_prod)],
#                       lb=0.0, name="z")
#     w = model.addVars(m_seg, lb=0.0, name="w")

#     # Objective (Eq. 33)
#     term1 = gp.quicksum(coef[i] * v_i0[i] * rho_bar * y[i]
#                         for i in range(m_seg))
#     term2 = gp.quicksum(coef[i] * uti[i, j] * (rho_bar - prices[j]) * z[i, j]
#                         for i in range(m_seg) for j in range(n_prod))
#     model.setObjective(term1 + term2, GRB.MINIMIZE)

#     # Basic constraints
#     model.addConstr(gp.quicksum(x[j] for j in range(n_prod)) <= cardinality,
#                     name="capacity")
#     model.addConstr(gp.quicksum(x[j] for j in range(n_prod)) >= 1,
#                     name="non_empty")

#     # MMNL structural equations
#     for i in range(m_seg):
#         model.addConstr(w[i] == v_i0[i] + gp.quicksum(uti[i, j] * x[j]
#                                                      for j in range(n_prod)),
#                         name=f"def_w[{i}]")

#     for i in range(m_seg):
#         for j in range(n_prod):
#             model.addQConstr(z[i, j] * w[i] >= x[j], name=f"soc1[{i},{j}]")

#     for i in range(m_seg):
#         model.addQConstr(y[i] * w[i] >= 1, name=f"soc2[{i}]")

#     for i in range(m_seg):
#         lhs = v_i0[i] * y[i] + gp.quicksum(uti[i, j] * z[i, j]
#                                           for j in range(n_prod))
#         model.addConstr(lhs >= 1, name=f"soc3[{i}]")

#     # McCormick strengthening
#     for i in range(m_seg):
#         for j in range(n_prod):
#             model.addConstr(z[i, j] <= y_u[i] * x[j],
#                             name=f"mc_up[{i},{j}]")
#             model.addConstr(z[i, j] >= y_l[i] * x[j],
#                             name=f"mc_low[{i},{j}]")
#             model.addConstr(z[i, j] <= y[i] - y_l[i] * (1 - x[j]),
#                             name=f"mc_box_up[{i},{j}]")
#             model.addConstr(z[i, j] >= y[i] - y_u[i] * (1 - x[j]),
#                             name=f"mc_box_low[{i},{j}]")

#     # Extra linear constraints
#     if A is not None and b is not None:
#         A = np.asarray(A)
#         b = np.asarray(b)
#         if A.shape[1] != n_prod:
#             raise ValueError("A must have shape (*, n_products)")
#         if A.shape[0] != b.shape[0]:
#             raise ValueError("A and b row mismatch")
#         for r in range(A.shape[0]):
#             model.addConstr(gp.quicksum(A[r, j] * x[j] for j in range(n_prod))
#                             <= float(b[r]), name=f"extra[{r}]")


#     # Optimization
#     model.optimize()

#     # Output
#     t = time.time() - start
#     best_bound = model.ObjBound
#     status = model.Status 
#     nodes = model.NodeCount
#     print(f"[INFO] Gurobi optimization finished with status {status}, nodes: {nodes}, time: {t:.2f} seconds.")

#     if model.Status == GRB.OPTIMAL:
#         z_opt = model.ObjVal

#         root_gap = None
#         z_root = None
#         end_gap = abs(z_opt - best_bound) / abs(z_opt) * 100

#         info = {
#             "status" : status,
#             "nodes"  :  nodes,
#             "egap"   : end_gap,
#             "rgap"   : root_gap,
#             "time"   : t,
#             "z_opt" : z_opt,
#             "z_root": z_root if root_gap is not None else None,
#             "z_bb": best_bound
#         }

#         x_sol = np.array([x[j].X for j in range(n_prod)], dtype=int)
#         opt_rev = mmnl_revenue(x_sol, uti, prices, v_i0, coef)
#         return float(opt_rev), x_sol, info
    
#     else:
#         print(f"[Warning] Gurobi optimization failed with status {model.Status}.")
#         end_gap, root_gap = None, None    
#         z_opt = None

#         info = {
#             "status" : status,
#             "nodes"  :  nodes,
#             "IterCount": model.IterCount,
#             "egap"   : end_gap,
#             "rgap"   : root_gap,
#             "time"   : t,
#             "z_opt" : z_opt,
#             "z_root": z_root if root_gap is not None else None,
#             "z_bb": best_bound
#         }
        
#         print(f"[Warning] Gurobi optimization failed with status {status}")
#         return best_bound, None, info  # or -1.0, None means infeasible
