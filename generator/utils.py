from dataclasses import dataclass
from typing import List, Dict, Any
import json
import numpy as np  

@dataclass
class MMNLInstanceData:
    """Store data for a single MMNL instance"""
    m: int  # Number of customer types
    n: int  # Number of products
    seed: int  # Random seed
    max_rev: float  # Maximum revenue
    gap: float  # Gap value
    cap_rate: float
    u: np.ndarray  # Utility matrix
    price: np.ndarray  # Price vector
    v0: np.ndarray  # No-purchase option utility
    omega: np.ndarray  # Customer type probability
    
    def __repr__(self):
        return f"MMNLInstanceData(m={self.m}, n={self.n}, seed={self.seed}, cap_rate={self.cap_rate:.1f}, max_rev={self.max_rev:.4f}, gap={self.gap:.4f})"


def parse_config_to_instances_MNL(config: Dict[str, Any]) -> List[MMNLInstanceData]:
    """
    Parse config dictionary into a list of MMNLInstanceData instances
    
    Args:
        config: Configuration dictionary loaded from JSON file
        
    Returns:
        List containing all instance data
    """
    instances = []
    
    for m_key, m_data in config.items():
        m = int(m_key.split('_')[1])  # Extract 5 from "5_segments"
        n = int(m_key.split('_')[0])  # Extract 50 from "50_products"

        seeds = m_data['seeds']
        max_revs = m_data['max_rev']
        gaps = m_data['gap']
        cap_rate = m_data.get('cap_rate', 1.0)
        data_list = m_data['data']

        # Create an instance for each seed
        for i, seed in enumerate(seeds):
            data_dict = data_list[i]
            
            instance = MMNLInstanceData(
                m=m,
                n=n,
                seed=seed,
                cap_rate=cap_rate,
                max_rev=max_revs[i],
                gap=gaps[i],
                u=np.array(data_dict['u']),
                price=np.array(data_dict['price']),
                v0=np.array(data_dict['v0']),
                omega=np.array(data_dict['omega'])
            )
            
            instances.append(instance)

    return instances

def load_MNL_instances(input_json_path):
    """
    Load MMNL instances from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing instance data
    """

    with open(input_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    instances = parse_config_to_instances_MNL(config)

    return instances


@dataclass
class NLInstanceData:
    """Store data for a single NL instance"""
    m: int  # Number of nests
    n: int  # Number of products per nest
    seed: int  # Random seed
    max_rev: float  # Maximum revenue
    gap: float  # Gap value
    cap_rate: float  # Capacity rate
    v: np.ndarray  # Utility matrix (m x n)
    price: np.ndarray  # Price matrix (m x n)
    gamma: np.ndarray  # Dissimilarity parameter for each nest (m,)
    v0: float  # Global no-purchase option utility
    vi0: np.ndarray  # Within-nest no-purchase utility (m,) or scalar
    
    def __repr__(self):
        return f"NLInstanceData(m={self.m}, n={self.n}, seed={self.seed}, cap_rate={self.cap_rate:.1f}, max_rev={self.max_rev:.4f}, gap={self.gap:.4f})"


def parse_config_to_instances_NL(config: Dict[str, Any]) -> List[NLInstanceData]:
    """
    Parse config dictionary into a list of NLInstanceData instances
    
    Args:
        config: Configuration dictionary loaded from JSON file
        
    Returns:
        List containing all instance data
    """
    instances = []
    
    for key, data_group in config.items():
        # Parse key format: "m_n" (e.g., "5_25" means 5 nests, 25 products per nest)
        parts = key.split('_')
        m = int(parts[1])  # Number of nests
        n = int(parts[0])  # Number of products per nest
        
        seeds = data_group['seeds']
        max_revs = data_group['max_rev']
        gaps = data_group['gap']
        cap_rate = data_group.get('cap_rate', 1.0)
        data_list = data_group['data']
        
        # Create an instance for each seed
        for i, seed in enumerate(seeds):
            data_dict = data_list[i]
            
            # Handle vi0 which can be either a scalar or array
            vi0_data = data_dict['vi0']
            if isinstance(vi0_data, (list, np.ndarray)):
                vi0 = np.array(vi0_data)
            else:
                vi0 = vi0_data  # Keep as scalar
            
            instance = NLInstanceData(
                m=m,
                n=n,
                seed=seed,
                cap_rate=cap_rate,
                max_rev=max_revs[i],
                gap=gaps[i],
                v=np.array(data_dict['v']),
                price=np.array(data_dict['price']),
                gamma=np.array(data_dict['gamma']),
                v0=data_dict['v0'],
                vi0=vi0
            )
            
            instances.append(instance)
    
    return instances


def load_NL_instances(input_json_path: str) -> List[NLInstanceData]:
    """
    Load NL instances from a JSON file.
    
    Args:
        input_json_path: Path to the JSON file containing instance data
        
    Returns:
        List of NLInstanceData instances
        
    Example:
        >>> instances = load_NL_instances('hard_data/nl_unconstrained_uniform01_data.json')
        >>> print(f"Loaded {len(instances)} instances")
        >>> print(instances[0])
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    instances = parse_config_to_instances_NL(config)
    
    return instances
