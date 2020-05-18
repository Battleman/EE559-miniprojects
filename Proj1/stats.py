"""As we can't use Numpy, this defines some useful statistical tools."""
from math import sqrt


def stddev(lst):
    """Compute the standard deviation of a list without Numpy."""
    return sqrt(sum((x - mean(lst))**2 for x in lst) / (len(lst)-1))


def mean(lst):
    """Compute the mean of a list without Numpy."""
    return sum(lst)/len(lst)
