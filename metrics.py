# metrics.py
import numpy as np

def gini_coefficient(values):
    values = np.array(values)
    if np.all(values == 0):
        return 0
    sorted_vals = np.sort(values)
    n = len(values)
    cumvals = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return gini

def fairness_index(resource_list):
    # Lower Gini = more fair
    return 1 - gini_coefficient(resource_list)
