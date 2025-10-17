import numpy as np
from dataclasses import dataclass

@dataclass
class NLData:
    price: np.ndarray
    v: np.ndarray
    gamma: np.ndarray
    v0: float
    vi0: float

def generate_data_nested(args):
    """
    Generate nested data according to the specified method in args.

    Parameters:
        args: An object with attributes:
            - generate_data_method (str): which data generation method to use
            - m (int): The number of nests
            - n (int): The number of products in each nest
            - gamma_range (list):  range for the dissimilarity parameter
            - full_capture (bool): whether the utility in each nest is fully captured
            - seed (optional): Random seed 

    Returns:
        NLData: A data structure containing:
            price: the price matrix of shape (m, n) 
            v: the utility matrix of shape (m, n)
            gamma: the dissimilarity parameter vector for each nest, shape (m,)
            v0: the baseline utility value 
            vi0: the baseline utility value or vector in each nest
    """
    seed = getattr(args, 'seed', None)
    if args.generate_data_method == 'nested_data_complex':
        data = nested_data_complex(args.m, args.n,args.gamma_range, args.full_capture, seed=seed)
    elif args.generate_data_method == 'nested_data_random':
        data = nested_data_random(args.m, args.n, args.gamma_range, args.full_capture, seed=seed)
    elif args.generate_data_method == 'nested_data_NewBounds':
        data = nested_data_NewBounds(args.m, args.n, args.gamma_range, args.full_capture, seed=seed)
    elif args.generate_data_method == 'nl_data_vi0_uniform01':
        data = nl_data_vi0_uniform01(args.m, args.n, args.gamma_range, args.full_capture, seed=seed)
    elif args.generate_data_method == 'nl_data_vi0_uniform34':
        data = nl_data_vi0_uniform34(args.m, args.n, args.gamma_range, args.full_capture, seed=seed)
    elif args.generate_data_method == 'nl_data_vi0_lognormal':
        data = nl_data_vi0_lognormal(args.m, args.n, args.gamma_range, args.full_capture, seed=seed)
    else:
        raise ValueError("Invalid data generation method")
    return data


def nested_data_complex(m, n,gamma_range=[0.5,1], full_capture=None, epsilon=0.6, seed=None):
    """
    Generate complex nested structure data.

    Parameters:
        m (int): number of nests
        n (int): number of products in each nest
        gamma_range (list): range for the dissimilarity parameter
        full_capture (bool): whether the utility in each nest is fully captured
        epsilon (float): control parameter, default 0.6
        seed (int, optional): random seed, default None

    Returns:
        NLData: A data structure containing:
            price: the price matrix of shape (m, n) 
            v: the utility matrix of shape (m, n)
            gamma: the dissimilarity parameter vector for each nest, shape (m,)
            v0: the baseline utility scalar 
            vi0: the baseline utility scalar or vector in each nest
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 4, size=(m, n-1))
    x = rng.uniform(1, 10, size=(m, n-1))
    y = rng.uniform(0.2, 1.8, size=(m, n-1))
    price = (epsilon ** u) * x
    v = (epsilon ** (2-u)) * y

    price_n = np.zeros((m, 1), dtype=int)
    y_n = rng.uniform(0.2, 1.8, size=(m, 1))
    v_n = (epsilon ** (-1)) * y_n

    price = np.concatenate((price, price_n), axis=1)
    v = np.concatenate((v, v_n), axis=1)

    v0 = 1
    gamma = rng.uniform(gamma_range[0], gamma_range[1], size=m)
    if full_capture:
        vi0 = epsilon * 1e-4
    else:
        vi0 = v0

    sorted_indices = np.argsort(v, axis=1)
    price = np.take_along_axis(price, sorted_indices, axis=1)
    v = np.take_along_axis(v, sorted_indices, axis=1)
    return NLData(price, v, gamma, v0, vi0)

def nested_data_random(m, n, gamma_range, full_capture=None, price_range=[100,250], seed=None):
    """
    Generate complex nested structure data.

    Parameters:
        m (int): number of nests
        n (int): number of products in each nest
        gamma_range (list): range for the dissimilarity parameter
        full_capture (bool): whether the utility in each nest is fully captured
        price_range (list): range for the prices of products
        seed (int, optional): random seed, default None

    Returns:
        NLData: A data structure containing:
            price: the price matrix of shape (m, n) 
            v: the utility matrix of shape (m, n)
            gamma: the dissimilarity parameter vector for each nest, shape (m,)
            v0: the baseline utility scalar 
            vi0: the baseline utility scalar or vector in each nest
    """
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(gamma_range[0], gamma_range[1], size=m)
    price = rng.uniform(price_range[0], price_range[1], size=(m, n))
    v = rng.uniform(0, 10, size=(m, n))

    if full_capture:
        vi0 = 1e-4
    else:
        vi0 = rng.uniform(0, 4, size=m)
    v0 = 10

    sorted_indices = np.argsort(v, axis=1)
    price = np.take_along_axis(price, sorted_indices, axis=1)
    v = np.take_along_axis(v, sorted_indices, axis=1)
    return NLData(price, v, gamma, v0, vi0)

def nested_data_NewBounds(m, n, gamma_range, full_capture=None, seed=None):
    """
    Generate complex nested structure data.

    Parameters:
        m (int): number of nests
        n (int): number of products in each nest
        gamma_range (list): range for the dissimilarity parameter
        full_capture (bool): whether the utility in each nest is fully captured
        seed (int, optional): random seed, default None

    Returns:
        NLData: A data structure containing:
            price: the price matrix of shape (m, n) 
            v: the utility matrix of shape (m, n)
            gamma: the dissimilarity parameter vector for each nest, shape (m,)
            v0: the baseline utility scalar 
            vi0: the baseline utility scalar or vector in each nest
    """
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(gamma_range[0], gamma_range[1], size=m)
    u = rng.uniform(0, 1, size=(m, n))
    x = rng.uniform(0.75, 1.25, size=(m, n))
    y = rng.uniform(0.75, 1.25, size=(m, n))
    price = 10 * (u ** 2) * x
    v = 10 * (1-u) * y

    v0 = 10
    if full_capture:
        vi0 = 1e-4
    else:
        vi0 = v0

    sorted_indices = np.argsort(v, axis=1)
    price = np.take_along_axis(price, sorted_indices, axis=1)
    v = np.take_along_axis(v, sorted_indices, axis=1)
    return NLData(price, v, gamma, v0, vi0)



def _base_NewBounds(m, n, gamma_range, full_capture=None, seed=None):
    """
    Helper function to generate heterogeneous Nested Logit data with given parameters.
    """
    base = nested_data_NewBounds(m, n, gamma_range, full_capture, seed)
    return base.price, base.v, base.gamma, base.v0, base.vi0

def nl_data_vi0_uniform01(m, n, gamma_range, full_capture=None, seed=None):
    """
    Generate Nested Logit data with outside-option parameter vi0 ~ Uniform(0, 1).

    Args:
        m (int): number of nests
        n (int): number of products in each nest
        gamma_range (list): range for the dissimilarity parameter
        full_capture (bool): whether the utility in each nest is fully captured
        seed (int, optional): random seed, default None

    Returns:
        NLData: Data object with price, v, gamma, v0, and vi0.
    """
    price, v, gamma, v0, base = _base_NewBounds(m, n, gamma_range, full_capture, seed)
    rng = np.random.default_rng(seed)
    if full_capture:
        vi0 = base
    else:
        vi0 = rng.uniform(0, 1, size=m)
    return NLData(price, v, gamma, v0, vi0)

def nl_data_vi0_uniform34(m, n, gamma_range, full_capture=None, seed=None):
    """
    Generate Nested Logit data with outside-option parameter vi0 ~ Uniform(3, 4).

    Args:
        m (int): number of nests
        n (int): number of products in each nest
        gamma_range (list): range for the dissimilarity parameter
        full_capture (bool): whether the utility in each nest is fully captured
        seed (int, optional): random seed, default None

    Returns:
        NLData: Data object with price, v, gamma, v0, and vi0.
    """
    price, v, gamma, v0, base = _base_NewBounds(m, n, gamma_range, full_capture, seed)
    rng = np.random.default_rng(seed)
    if full_capture:
        vi0 = base
    else:
        vi0 = rng.uniform(3, 4, size=m)
    return NLData(price, v, gamma, v0, vi0)

def nl_data_vi0_lognormal(m, n, gamma_range, full_capture=None, seed=None):
    """
    Generate Nested Logit data with outside-option parameter vi0 ~ LogNormal(μ=1, σ=0.5),
    clipped to [1, 5] to keep numerical stability.
    
    Args:
        m (int): number of nests
        n (int): number of products in each nest
        gamma_range (list): range for the dissimilarity parameter
        full_capture (bool): whether the utility in each nest is fully captured
        seed (int, optional): random seed, default None

    Returns:
        NLData: Data object with price, v, gamma, v0, and vi0.
    """
    price, v, gamma, v0, base = _base_NewBounds(m, n, gamma_range, full_capture, seed)
    rng = np.random.default_rng(seed)
    if full_capture:
        vi0 = base
    else:
        vi0 = rng.lognormal(mean=1, sigma=0.5, size=m)
        vi0 = np.clip(vi0, 1.0, 5.0)
    return NLData(price, v, gamma, v0, vi0)