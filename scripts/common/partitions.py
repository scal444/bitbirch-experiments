"""Compare labelings as partitions (ignores cluster numbering)."""
import numpy as np


def canonical(labels):
    """Renumber clusters by first appearance, so equal partitions give equal arrays."""
    _, first, inv = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inv]


def same_partition(a, b):
    return a.shape == b.shape and np.array_equal(canonical(a), canonical(b))
