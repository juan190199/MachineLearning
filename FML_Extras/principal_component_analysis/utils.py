import numpy as np
from scipy import stats


# Create data
def pdf(x, e):
    return 0.5 * (stats.norm(scale=0.25 / e).pdf(x) + stats.norm(scale=4 / e).pdf(x))
