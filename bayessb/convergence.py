"""
Implementation of the Gelman-Rubin convergence criterion calculations.

The implementation is based on the description on p. 296-7 of 

Gelman A, Carlin JB, Stern HS, Rubin DB, *Bayesian Data Analysis, 2nd Ed.*
(2004) Chapman & Hall. 

What follows is a paraphrase of the description from this book.

Suppose you have *m* parallel sequences, each of length *n* (after discarding
the steps for the burn-in period--they suggest throwing out the first half of
the sequence). For for the vector of parameters :math:`\psi`, we label the
simulation draws (i.e., the parameter values chosen at a simulation step) as
:math:`\psi_{ij} (i = 1,\ldots,n; j = 1,\ldots,m)`, and we compute *B* and *W*,
the between- and within-sequence variances. The between-sequence variance is
defined as

.. math::

    B = \frac{n}{m-1} \sum_{j=1}^m \left(\overline{\psi}_{.j} -
             \overline{\psi}_{..}\right)^2,\ \mathrm{where}

    \overline{\psi}_{.j} = \frac{1}{n} \sum_{i=1}^n \psi_{ij},

    \overline{\psi}_{..} = \frac{1}{m} \sum_{j=1}^m \overline{\psi}_{.j}

In words, :math:`\overline{\psi}_{.j}` are the mean parameter values averaged
across all steps for the single chain *j*, and :math:`\overline{\psi}_{..}` is
the mean of the :math:`\overline{\psi}_{.j}` across the *j* chains. These two
quantities are used to calculate the variance between chains in the usual way,
with the note that "the between-sequence variance, *B*, contains a factor of *n*
because it is based on the variance of the within-sequence means,
:math:`\overline{\psi}_{.j}, each of which is an average of *n* values
:math:`\psi_{ij}."

The within-sequence variance is calculated as follows:

.. math::

    W = \frac{1}{m} \sum_{j=1}^m s_j^2,\ \mathrm{where}

    s^2_j = \frac{1}{n-1} \sum_{i=1}^n
        \left(\psi_{ij} - \overline{\psi}_{.j} \right)^2

Once these variances have been calculated, the true variance for the parameters
can be estimated using

.. math::

    \widehat{\mathrm{var}}^+ (\psi|y) = \frac{n-1}{n} W + \frac{1}{n} B

Gelman et al. state: "This quantity overestimates the marginal posterior
variance assuming the starting distribution (i.e., selection of starting
points) is overdispersed, but is unbiased under stationarity (that is, if the
starting distribution equals the target distribution, or in the limit as n goes
to infinity." Similarly, the within-chain variance *W* is an underestimate
of the total variance because the full spectrum of parameter values has likely
been undersampled.

The convergence criterion is then calculated as

.. math::

    \hat{R} = \sqrt{\frac{\widehat{\mathrm{var}}^+ (\psi|y)} {W}}

which will decline to 1 as the number of steps is increased. The convergence
criterion therefore gives a measure of the likelihood that the parameter
estimates will approach their true distributions if the number of steps is
increased. As Gelman et al. state: "The condition of :math:`\hat{R}` being
'near' 1 depends on the problem at hand; for most examples, values below 1.1 are
acceptable, but for a final analysis in a critical problem, a higher level of
precision may be required."

.. note:: The parameter distributions should be normal

    As Gelman et al. state: "Since our method our method of assessing
    convergence is based on means and variances, it is best where possible to
    transform the scalar estimands to be approximately normal (for example,
    take logarithms of all-positive quantities and logits of quantities that
    lie between 0 and 1."

.. note:: The starting distribution is critical!

    "Even if an iterative simulation appears to converge and has passed all
    tests of convergence, it still may actually be far from convergence if
    important areas of the target distribution were not captured by the
    starting distribution and are not easily reachable by the simulation
    algorithm."

"""

import numpy as np

def check_chain_lengths(chain_set):
    """Checks to make sure there is more than one chain in the set, and that
    all chains are the same length.
    """

    if len(chain_set) < 2:
        raise Exception("To calculate the convergence criterion, there must " +
                        "be more than one chain.")
    n = len(chain_set[0])
    for i in range(0, n):
        if len(chain_set[i]) != i:
            raise Exception("To calculate the within-chain variances, all " +
                "chains must be the same length.")

def within_chain_variances(chain_set):
    """
    Calculate the vector of average within-chain variances, *W*.

    Takes a list of chains (each expressed as an array of positions,
    presumed to have already been transformed to linear (not log) space.
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

    Takes a list of chains (each expressed as an array of positions,
    presumed to have already been transformed to linear (not log) space.
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
    """
    Calculate the best estimate of the variances of the parameters given
    the chains that have been run, that is,

    .. math::

    \widehat{\mathrm{var}}^+ (\psi|y) = \frac{n-1}{n} W + \frac{1}{n} B

    Takes a list of chains (each expressed as an array of positions,
    presumed to have already been transformed to linear (not log) space.
    Note that all chains should be the same length.
    """

    n = float(len(chain_set[0]))

    W = within_chain_variances(chain_set)
    B = between_chain_variances(chain_set)

    return (((n-1)/n) * W) + ((1/n) * B)

def convergence_criterion(chain_set):
    """Calculate the Gelman-Rubin convergence criterion, defined as

    .. math::

        \hat{R} = \sqrt{\frac{\widehat{\mathrm{var}}^+ (\psi|y)} {W}}

    which should decline to 1 as the number of simulation steps is increased.
    """

    W = within_chain_variances(chain_set)
    var = parameter_variance_estimates(chain_set)

    return np.sqrt(var / W)
