import numpy as np

def to_x0_vector(x_candidate, n, cardinality=None, thresh=0.5):
    """
    Convert a candidate solution into a binary 0/1 vector of length n.

    Handles three possible input formats:
      1. A real-valued list/array of length n (interpreted as probabilities) 
         → binarize using a threshold.
      2. A binary 0/1 vector of length n → returned as-is.
      3. A list of indices → set corresponding positions to 1.

    Args:
        x_candidate (array-like): Candidate solution. Can be:
            - length-n float list/array (probabilities between 0 and 1),
            - length-n binary list/array (0/1 values),
            - list of indices of selected items.
        n (int): Dimension of the output vector.
        cardinality (int, optional): Max number of items allowed (truncate if exceeded).
        thresh (float, optional): Threshold for binarization if input is continuous. Default = 0.5.

    Returns:
        list[int]: Binary vector (length n) with values {0,1}.
    """
    x_candidate = np.asarray(x_candidate)

    if x_candidate.ndim == 1 and x_candidate.size == n and x_candidate.dtype != int:
        # Case 1: probability-like vector → binarize by threshold
        x_bin = (x_candidate > thresh).astype(int)
    elif x_candidate.ndim == 1 and x_candidate.size == n and set(np.unique(x_candidate)).issubset({0,1}):
        # Case 2: already a binary vector
        x_bin = x_candidate.astype(int)
    else:
        # Case 3: interpret as index list
        idx = np.unique(x_candidate.astype(int))
        if np.any(idx < 0) or np.any(idx >= n):
            raise ValueError(f"x0 索引越界，允许 [0,{n-1}]，收到 {idx}")
        x_bin = np.zeros(n, dtype=int)
        x_bin[idx] = 1

    # Safety check: ensure at least one item is selected
    if x_bin.sum() == 0:
        x_bin[0] = 1

    # Cardinality constraint: keep only up to `cardinality` items
    if cardinality is not None and x_bin.sum() > cardinality:
        on_idx = np.where(x_bin == 1)[0]
        # Here: keep first K items; can replace with greedy revenue-based selection if needed
        keep = on_idx[:int(cardinality)]
        x_bin[:] = 0
        x_bin[keep] = 1

    return x_bin.tolist()