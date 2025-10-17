import numpy as np
from dataclasses import dataclass

@dataclass
class MMNLData:
    u: np.ndarray
    price: np.ndarray
    v0: np.ndarray
    omega: np.ndarray



def generate_data_mmnl(args):
    """
    Generate MMNL data according to the specified method in args.

    Args:
        args: an object with attributes:
            - generate_data_method (str): which data generation method to use
            - production_num (int): number of products
            - m (int): number of customer types (for some methods)
            - seed (int, optional): random seed

    Returns:
        u: the utility matrix of shape (m, num_prod)
        p: the price vector of shape (1, num_prod)
        v0: the baseline utility vector of shape (m,)
        omega: the choice probability vector of shape (m,)
    """
    seed = getattr(args, 'seed', None)
    if args.generate_data_method == 'mmnl_data_random':
        data = mmnl_data_random(args.production_num, args.m, seed=seed)
    elif args.generate_data_method == 'mmnl_data_easy':
        data = mmnl_data_easy(args.production_num, args.m, seed=seed)
    elif args.generate_data_method == 'mmnl_data_hard':
        data = mmnl_data_hard(args.production_num, seed=seed)

    elif args.generate_data_method == 'mmnl_data_v0_lognorm':
        data = mmnl_data_v0_lognorm(args.production_num, args.m, args.revenue_type, seed=seed)

    else:
        raise ValueError("Invalid data generation method")
    return data


def build_revenue_curve(num_prod: int, rs) -> np.ndarray:
    """
    Build a synthetic revenue curve for products.

    Args:
        num_prod (int): Number of products.
        rs (str): Revenue scheme type, one of {"RS1", "RS2", "RS3", "RS4"}.
            - RS1: Linear decrease.
            - RS2: Log-like (high at front, then steep drop).
            - RS3: Convex (steep early drop, then flatten).
            - RS4: Concave (fast initial drop, then slow).

    Returns:
        np.ndarray: Revenue curve of shape (1, num_prod). 
    """

    n = num_prod
    idx = np.arange(1, n + 1, dtype=float) 
    
    if rs == "RS1":
        # Linear decreasing: from 1.0 down to 0.2
        r = np.linspace(1.0, 0.2, n)
    elif rs == "RS2":
        # Log-like: high at start, sharper drop later
        r = 1.0 - np.log1p(idx) / np.log1p(n + 1)
        r = 0.2 + 0.8 * (r - r.min()) / (r.max() - r.min())
    elif rs == "RS3":
        # Convex: steep early, then slow (quadratic curve)
        base = np.linspace(0.0, 1.0, n)
        r = 1.0 - base**2
        r = 0.2 + 0.8 * (r - r.min()) / (r.max() - r.min())
    elif rs == "RS4":
         # Concave: fast early drop, then slow (square root curve)
        base = np.linspace(0.0, 1.0, n)
        r = 1.0 - np.sqrt(base)
        r = 0.2 + 0.8 * (r - r.min()) / (r.max() - r.min())
    else:
        raise ValueError("rs must be one of RS1–RS4")
    # Ensure strictly decreasing sequence (apply tiny perturbation if needed)
    r = np.maximum.accumulate(r[::-1])[::-1]
    r -= np.linspace(0, 1e-6, n)
    return r.reshape(1, -1)



def mmnl_data_v0_lognorm(num_prod, m, r_type, seed=None):
    """
    Lognormal v0 with heavy tail (mu=1, sigma=0.5), clipped to [1, 5].
    Captures continuous heterogeneity; a few segments get very large v0.

    Args:
        num_prod (int): Number of products .
        m (int): Number of customer segments (types).
        r_type (str): Revenue scheme type ("RS1"–"RS4"), used to build prices.
        seed (int, optional): Random seed for reproducibility.

    Returns:
       MMNLData: A data structure containing:
            u: the utility matrix of shape (m, num_prod)
            p: the price vector of shape (1, num_prod)
            v0: the baseline utility vector of shape (m,)
            omega: the choice probability vector of shape (m,)
     """

    rng = np.random.default_rng(seed)
    N = num_prod + 1  # include opt-out
    n_ec = 2
    sigma = 3

    # Create epsilon: [n_ec, N]
    epsilon = np.zeros((n_ec, N))
    ecs1 = int(N / 2)
    epsilon[0, :ecs1] = 1  # products 0-4
    epsilon[1, ecs1:] = 1  # products 5-9 + opt-out
    # Shared random components
    sigmas = np.ones(n_ec) * sigma
    eta = rng.normal(0, 1, size=(m, n_ec))  # η ~ N(0, 1), shape (m, n_ec)
    omega_ec = sigmas * eta  # ω = σ * η, shape (m, n_ec)

    V = omega_ec @ epsilon  # shape = (m, N)
    utility_full = np.exp(V)
    u = utility_full[:, :-1]

    price = build_revenue_curve(num_prod, r_type) 

    omega = rng.random(m)
    omega = omega / np.sum(omega)

    v0 = rng.lognormal(mean=1, sigma=0.5, size=m)
    v0 = np.clip(v0, 1.0, 5.0)  # keep numerics reasonable
    return MMNLData(u, price, v0, omega)



def mmnl_data_random(num_prod, m, seed=None):
    """
    Randomly generate MMNL data for a given number of products and customer type.

    Args:
        num_prod (int): Number of products
        m (int): Number of customer types
        seed (int, optional): Random seed

    Returns:
        MMNLData: A data structure containing:
            u: the utility matrix of shape (m, num_prod)
            p: the price vector of shape (1, num_prod)
            v0: the baseline utility vector of shape (m,)
            omega: the choice probability vector of shape (m,)
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=(m, num_prod))
    price = rng.uniform(1, 3, size=(1, num_prod))
    omega = rng.random(m)
    omega = omega / np.sum(omega)
    v0 = np.ones((m))
    ind = rng.choice(np.arange(m), size=int(m / 2), replace=False)
    v0[ind] = 5
    return MMNLData(u, price, v0, omega)

def mmnl_data_easy(num_prod, m, seed=None):
    """
    Generate easy MMNL data for a given number of products.

    Args:
        num_prod (int): Number of products
        m (int): Number of customer types
        seed (int, optional): Random seed

    Returns:
        MMNLData: A data structure containing:
            u: the utility matrix of shape (m, num_prod)
            p: the price vector of shape (1, num_prod)
            v0: the baseline utility vector of shape (m,)
            omega: the choice probability vector of shape (m,)
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=(m, num_prod))
    price = rng.uniform(1, 3, size=(1, num_prod))
    omega = np.full(m, 1 / m)
    v0 = np.full(m, 5)
    return MMNLData(u, price, v0, omega)

def mmnl_data_hard(num_prod, max_nonzero=11, seed=None):
    """
    Generate hard (sparse) MMNL data for a given number of products.

    Args:
        num_prod (int): Number of products
        max_nonzero (int): Maximum number of nonzero utilities per customer type
        seed (int, optional): Random seed

    Returns:
        MMNLData: A data structure containing:
            u: the utility matrix of shape (m, num_prod)
            p: the price vector of shape (1, num_prod)
            v0: the baseline utility vector of shape (m,)
            omega: the choice probability vector of shape (m,)
    """
    rng = np.random.default_rng(seed)
    m = num_prod
    u = np.zeros((m, num_prod))
    for i in range(m):
        k = min(max_nonzero, num_prod)
        indices = rng.choice(num_prod, k, replace=False)
        values = rng.uniform(0, 1, k)
        u[i, indices] = values
    price = rng.uniform(1, 3, size=(1, num_prod))
    omega = rng.uniform(0, 1, size=m)
    omega = omega / np.sum(omega)
    v0 = np.full(m, 1)
    return MMNLData(u, price, v0, omega)

