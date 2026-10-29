import numpy as np

def median_ci(x, ci=(16, 84)):
    med = np.median(x)
    lo, hi = np.percentile(x, ci)
    return float(med), float(lo), float(hi)
