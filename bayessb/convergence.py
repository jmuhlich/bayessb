r"""
Implementation of the Gelman-Rubin convergence criterion calculations.

This implementation is based on the description on p. 296-7 of Gelman A, Carlin
JB, Stern HS, Rubin DB, *Bayesian Data Analysis, 2nd Ed.* (2004) Chapman &
Hall. What follows is a paraphrasing of the description from this book.

Suppose you have *m* parallel sequences, each of length *n* (after discarding
the steps for the burn-in period---Gelman et al. suggest throwing out the first
half of the sequence). For the vector of estimated parameters :math:`\psi`, we
label the simulation draws (i.e., the parameter values chosen at a simulation
step) as :math:`\psi_{ij} (i = 1,\ldots,n; j = 1,\ldots,m)`, and we compute *B*
and *W*, the between- and within-sequence variances, respectively. The
between-sequence variance is calculated by the function
:py:func:`between_chain_variances` and is defined as

.. math::

   B = \frac{n}{m-1} \sum_{j=1}^m \left(\overline{\psi}_{.j} - \overline{\psi}_{..}\right)^2,\ \mathrm{where}

   \overline{\psi}_{.j} = \frac{1}{n} \sum_{i=1}^n \psi_{ij},

   \overline{\psi}_{..} = \frac{1}{m} \sum_{j=1}^m \overline{\psi}_{.j}

:math:`\overline{\psi}_{.j}` are the mean parameter values averaged across all
steps for the single chain *j*, and :math:`\overline{\psi}_{..}` is the mean of
the :math:`\overline{\psi}_{.j}` across the *j* chains. These two quantities
are used to calculate the variance between chains, with the note that "the
between-sequence variance, *B*, contains a factor of *n* because it is based on
the variance of the within-sequence means, :math:`\overline{\psi}_{.j}`, each
of which is an average of *n* values :math:`\psi_{ij}`."

The within-sequence variance is calculated by the function
:py:func:`within_chain_variances` and is defined as:

.. math::

    W = \frac{1}{m} \sum_{j=1}^m s_j^2,\ \mathrm{where}

    s^2_j = \frac{1}{n-1} \sum_{i=1}^n
        \left(\psi_{ij} - \overline{\psi}_{.j} \right)^2

Once these variances have been calculated, the true variance for the parameters
can be estimated using the function :py:func:`parameter_variance_estimates`,
which calculates

.. math::

    \widehat{\mathrm{var}}^+ (\psi|y) = \frac{n-1}{n} W + \frac{1}{n} B

Gelman et al. state: "This quantity overestimates the marginal posterior
variance assuming the starting distribution (i.e., selection of starting
points) is overdispersed, but is unbiased under stationarity (that is, if the
starting distribution equals the target distribution), or in the limit as n goes
to infinity." Similarly, the within-chain variance *W* is an underestimate
of the total variance because the full spectrum of parameter values has likely
been undersampled.

The convergence criterion, calculated by the function
:py:func:`convergence_criterion`, is defined as

.. math::

    \hat{R} = \sqrt{\frac{\widehat{\mathrm{var}}^+ (\psi|y)} {W}},

which will decline to 1 as the number of steps is increased. The convergence
criterion therefore gives a measure of the likelihood that the parameter
estimates will approach their true distributions if the number of steps is
increased. As Gelman et al. state: "The condition of :math:`\hat{R}` being
'near' 1 depends on the problem at hand; for most examples, values below 1.1 are
acceptable, but for a final analysis in a critical problem, a higher level of
precision may be required."

.. note:: The parameter distributions should be (approximately) normal

    As Gelman et al. state: "Since our method our method of assessing
    convergence is based on means and variances, it is best where possible to
    transform the scalar estimands to be approximately normal (for example,
    take logarithms of all-positive quantities and logits of quantities that
    lie between 0 and 1)."

.. note:: The starting distribution is critical!

    "Even if an iterative simulation appears to converge and has passed all
    tests of convergence, it still may actually be far from convergence if
    important areas of the target distribution were not captured by the
    starting distribution and are not easily reachable by the simulation
    algorithm."

"""

import numpy as np
from nose.tools import eq_

def check_chain_lengths(chain_set):
    """Checks to make sure there is more than one chain in the set, and that
    all chains are the same length.
    """

    if len(chain_set) < 2:
        raise Exception("To calculate the convergence criterion, there must " +
                        "be more than one chain.")
    n = len(chain_set[0])
    for i in range(0, len(chain_set)):
        if len(chain_set[i]) != n:
            raise Exception("To calculate the within-chain variances, all " +
                "chains must be the same length.")

def within_chain_variances(chain_set):
    """
    Calculate the vector of average within-chain variances, *W*.

    Takes a list of chains (each expressed as an array of positions).
    Note that all chains should be the same length.
    """

    # Make sure that all chains are the same length
    check_chain_lengths(chain_set)

    # Calculate each within-chain variance (s_j^2):
    chain_variances = []
    for chain in chain_set:
        chain_variances.append(np.var(chain, axis=0, ddof=1))

    # Calculate the average within-chain variance (W):
    W = np.mean(chain_variances, axis=0)

    return W

def between_chain_variances(chain_set):
    """
    Calculate the vector of between-chain variances, *B*.

    Takes a list of chains (each expressed as an array of positions).
    Note that all chains should be the same length.
    """

    # Make sure that all chains are the same length
    check_chain_lengths(chain_set)

    # Calculate each within-chain average (i.e., psi_{.j}) :
    chain_averages = []
    for chain in chain_set:
        chain_averages.append(np.mean(chain, axis=0))

    # Calculate the between-chain variance (B):
    n = len(chain_set[0])
    B = n * np.var(chain_averages, axis=0, ddof=1)

    return B

def parameter_variance_estimates(chain_set):
    r"""
    Calculate the best estimate of the variances of the parameters given
    the chains that have been run, that is,

    .. math::

       \widehat{\mathrm{var}}^+ (\psi|y) = \frac{n-1}{n} W + \frac{1}{n} B

    Takes a list of chains (each expressed as an array of positions).
    Note that all chains should be the same length.
    """

    n = float(len(chain_set[0]))

    W = within_chain_variances(chain_set)
    B = between_chain_variances(chain_set)

    return (((n-1)/n) * W) + ((1/n) * B)

def convergence_criterion(mcmc_set, mask=False, thin=1):
    r"""Calculate the Gelman-Rubin convergence criterion, defined as

    .. math::

        \hat{R} = \sqrt{\frac{\widehat{\mathrm{var}}^+ (\psi|y)} {W}}

    which should decline to 1 as the number of simulation steps is increased.

    Parameters
    ----------
    mcmc_set : list of bayessb.MCMC
        The list of MCMC objects representing completed runs of estimation.
        There should be more than one, and all should be the same length.
    mask : bool/int, optional
        If True (default), the first half of the walk will be discarded.
        If False, none of the steps of the walk will be discarded.
        If an integer, specifies the number of steps to be discarded from the
        beginning of the walk.
    thin : int, optional
        The amount of thinning to be applied to the walk. For a given value
        k, only every k steps are sampled from the walk. The default is 1
        (no thinning). Thinning reduces unwanted autocorrelations in parameter
        values within a given walk.
    """

    # Use the number of steps from the first MCMC in the list
    if mask is True:
        mask = mcmc_set[0].options.nsteps / 2
    if mask is False:
        mask = 0

    # Iterate over the MCMC set, assembling a list of chains with the
    # specified mask and thinning
    chain_set = []
    min_accepts = np.inf
    for mcmc in mcmc_set:
        mixed_positions = mcmc.positions[mask:]
        mixed_accepts = mixed_positions[mcmc.accepts[mask:]]
        thinned_accepts = mixed_accepts[::thin]
        chain_set.append(thinned_accepts)

        if (len(thinned_accepts) < min_accepts):
            min_accepts = len(thinned_accepts)

    # Truncate the chains to make them all the length of the one with
    # with the fewest accepts
    for i, chain in enumerate(chain_set):
        chain_set[i] = chain[len(chain) - min_accepts:]

    # Run the calculations on the chain set
    W = within_chain_variances(chain_set)
    var = parameter_variance_estimates(chain_set)

    return np.sqrt(var / W)

# -- TESTS ---------------------------------------------------------
# 
# The following are tests of the convergence criterion calculations using
# the simple data given below. These tests can be run with, e.g.
#
# > nosetests convergence.py

test_data = [[1, 2, 3], [4, 5, 6]]

def test_within_chain_variances():
    """Check the within-chain variance calculation using a simple example."""
    # The average of the first chain is 2; the second is 5
    # The variance of the first chain is 1/2 * 2 = 1
    # The variance of the second chain is also 1/2 * 2 = 1
    # The average of the two variances is therefore 1.0
    eq_(1.0, within_chain_variances(test_data),
            "Failed to correctly calculate within-chain variance!")

def test_between_chain_variances():
    """Check the between-chain variance calculation using a simple example."""
    # The average of the first chain is 2; the second is 5
    # The average of the two averages is thus 3.5
    # The variance between the chains is thus
    # = 3/1 * [(3.5-2)^2 + (3.5-5)^2]
    # = 3 * [9/4 + 9/4]
    # = 54/4 = 13.5
    eq_(13.5, between_chain_variances(test_data),
            "Failed to correctly calculate between-chain variance!")

def test_parameter_variance_estimates():
    """Check the parameter variance estimate calculation using a simple
    example."""
    # For the test data, W is 1.0 and B is 13.5 (see tests above)
    # The variance estimate is thus
    # = (2/3 * 1.0) + (1/3 * 13.5)
    # = 2/3 + 18/4
    # = 8/12 + 54/12
    # = 62/12
    eq_(62/12., parameter_variance_estimates(test_data),
            "Failed to correctly calculate parameter variance estimates!")

def test_convergence_criterion():
    """Check the convergence criterion calculation using a simple example."""
    # For the test data, the estimated variance is 62/12, and the
    # within-chain variance is 1.0 (see other tests above)
    # The convergence criterion is therefore
    # sqrt( (62/12) / 1.0)
    eq_(np.sqrt(62/12.), convergence_criterion(test_data),
            "Failed to correctly calculate the convergence criterion!")

