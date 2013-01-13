"""Functions to implement convergence testing."""

import numpy as np
from nose.tools import eq_

def within_chain_variances(chain_set):
    """
    Takes a list of chains (each expressed as an array of positions,
    each transformed to linear, not log space. 

    Returns W

    Note that all chains should be the same length, n."""
    # TODO: Check/get chain lengths
    # Make sure that we have more than one chain!
    if len(chain_set) < 2:
        raise Exception("To calculate the convergence criterion, there must " +
                        "be more than one chain.")

    # Calculate each within-chain variance (s_j^2):
    chain_variances = []
    for chain in chain_set:
        chain_variances.append(np.var(chain, axis=0, ddof=1))

    # Calculate the average within-chain variance (W):
    W = np.mean(chain_variances, axis=0)

    return W

def between_chain_variances(chain_set):
    """
    Takes a list of chains (each expressed as an array of positions,
    each transformed to linear, not log space. 

    Returns B

    Note that all chains should be the same length, n."""
    # TODO: Check/get chain lengths
    # Make sure that we have more than one chain!
    if len(chain_set) < 2:
        raise Exception("To calculate the convergence criterion, there must " +
                        "be more than one chain.")

    # Calculate each within-chain average, that is, psi_{.j} :
    chain_averages = []
    for chain in chain_set:
        chain_averages.append(np.mean(chain, axis=0))

    # Calculate the between-chain variance (B):
    n = len(chain_set[0])
    B = n * np.var(chain_averages, axis=0, ddof=1)

    return B

def parameter_variance_estimates(chain_set):
    # TODO: Check/get chain lengths
    n = float(len(chain_set[0]))

    W = within_chain_variances(chain_set)
    B = between_chain_variances(chain_set)

    return (((n-1)/n) * W) + ((1/n) * B)

def convergence_criterion(chain_set):
    W = within_chain_variances(chain_set)
    var = parameter_variance_estimates(chain_set)

    return np.sqrt(var / W)