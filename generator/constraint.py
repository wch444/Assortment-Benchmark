import numpy as np

def generate_constraints(args):
    """
    Generate constraints (A, B) for assortment optimization problems under different constraint types.

    Args:
        args: An object with the required parameters:

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - A (ndarray): Constraint matrix.
            - B (ndarray): Constraint vector.

    Raises:
        ValueError: If input parameters are invalid.
        NotImplementedError: If an unsupported constraint type is requested.
    """
    # Map constraint type to generator functions
    n = args.production_num

    generators = {
        "cardinality": lambda: cardinality(n, args.cap) if args.cap_rate <= 1 else ValueError("Cardinality constraint cap_rate must be in (0,1)"),
        "cardinality_nested": lambda: (
            card_nested_logit(args.m, args.n, args.cap_nest)
            if args.choice_model == "nl" and args.cap_rate <= 1 else (_ for _ in ()).throw(ValueError("Nested logit cardinality constraints require choice_model='nl' and cap_rate must be in (0,1)."))
        ),
        "capacity_uniform1": lambda: cons_capacity(args.constraint_num, n, mode="uniform1"),
        "capacity_uniform2": lambda: cons_capacity(args.constraint_num, n, mode="uniform2"),
        "capacity_binary":   lambda: cons_capacity(args.constraint_num, n, mode="binary"),
        "capacity_hybrid":   lambda: cons_capacity(args.constraint_num, n, mode="hybrid"),
        "None":              lambda: (np.ones((1, n)), np.ones((1,)) * n) if args.cap_rate == 1 else (_ for _ in ()).throw(ValueError("No constraints require cap_rate=1")),
    }

    if args.constraint_type not in generators:
        raise NotImplementedError(f"Unsupported constraint type: {args.constraint_type}")

    A, B = generators[args.constraint_type]()
    return A, B

def cardinality(n, cap):
    """
    Generate cardinality constraints where at most `cap` products can be selected.

    Args:
        n (int): Number of products.
        cap (int): Maximum number of products allowed.

    Returns:
        A (np.ndarray): Constraint matrix of shape (1, n).
        B (np.ndarray): Constraint vector of shape (1,).
    """
    if not (1 <= cap < n):
        raise ValueError(f"Cardinality constraint cap must be in [1, {n}), got {cap}")
    A = np.ones((1, n), dtype=np.float32)
    B = np.full((1,), cap, dtype=np.float32)
    return A, B

def card_nested_logit(m,n, cap):
    """
    Generate constraints for a nested logit model where:
    - There are m nests
    - Each nest has n products
    - At most c products can be selected from each nest
    
    Args:
        m (int): Total number of nests
        n (int): Number of products in each nest
        c (int): Maximum number of products allowed in each nest
        
    Returns:
        A (np.ndarray): Constraint matrix of shape (m, m*n)
        B (np.ndarray): Constraint vector of shape (m,)
    """
    if not (1 <= cap < n):
        raise ValueError(f"Nested logit cardinality constraint cap must be in [1, {n}), got {cap}")
    total_products = m * n
    A = np.zeros((m, total_products), dtype=np.float32)
    
    # For each nest, create a constraint limiting the number of products
    for i in range(m):
        # Set coefficients for products in the current nest
        A[i, i * n : (i + 1) * n] = 1.0
    
    # Right-hand side: each nest can have at most c products
    B = np.full((m,), cap, dtype=np.float32)
    
    return A, B


def cons_capacity(m, n,mode='None', seed=42):
    """
    Generate capacity constraints under different random modes.

    Args:
        m (int): Number of constraints.
        n (int): Number of products.
        cap (int): Cardinality bound.
        mode (str): Mode for generating random constraints. 
                    Options: "uniform1", "uniform2", "binary", "hybrid", "None".
        seed (int): Random seed.

    Returns:
        A (np.ndarray): Constraint matrix of shape (m, n) or (1, n) if mode is invalid.
        B (np.ndarray): Constraint vector of shape (m,) or (1,) if mode is invalid.
    """
    rng = np.random.default_rng(seed)
    # Define capacity constraint generators
    generators = {
        "uniform1": lambda: (rng.uniform(0, 1, size=(m, n)), rng.uniform(1, 2, size=m)),
        "uniform2": lambda: (rng.uniform(1, 5, size=(m, n)), rng.uniform(2, 10, size=m)),
        "binary":   lambda: (rng.integers(0, 2, size=(m, n)), rng.integers(1, 6, size=m)),
        "hybrid":   lambda: _generate_hybrid_capacity(rng, m, n),
        "None":     lambda: (np.ones((1, n)), np.ones((1,)) * n),
    }

    if mode not in generators:
        raise ValueError(f"Invalid constraint mode: {mode}")

    A, B = generators[mode]()
    return A, B

def _generate_hybrid_capacity(rng, m, n):
    """
    Internal helper for generating hybrid constraints (half uniform2, half binary).
    """
    m_unif = m // 2
    m_bina = m - m_unif
    a_unif = rng.uniform(1, 5, size=(m_unif, n))
    b_unif = rng.uniform(2, 10, size=m_unif)
    a_bina = rng.integers(0, 2, size=(m_bina, n))
    b_bina = rng.integers(1, 6, size=m_bina)
    A = np.concatenate([a_unif, a_bina], axis=0)
    B = np.concatenate([b_unif, b_bina], axis=0)
    return A, B
