import numpy as np
from matplotlib import pyplot as plt
from pysb.report import reporter, Result
from bayessb import convergence

@reporter('Number of chains')
def num_chains(mcmc_set):
    return Result(len(mcmc_set.chains), None)

@reporter('Conv. Criterion')
def convergence_criterion(mcmc_set):
    # TODO Make plots of parameter traces
    return Result(convergence.convergence_criterion(mcmc_set), None)

@reporter('Maximum likelihood')
def maximum_likelihood(mcmc_set):
    # Get the maximum likelihood
    (max_likelihood, max_likelihood_position) = mcmc_set.maximum_likelihood()

    # Plot the maximum likelihood fit
    if hasattr(mcmc_set.chains[0], 'fit_plotting_function'):
        mcmc_set.chains[0].fit_plotting_function(
                                        position=max_likelihood_position)
        filename = '%s_max_likelihood_plot.png' % mcmc_set.name
        plt.savefig(filename)
    else:
        filename = None

    return Result(max_likelihood, filename)

@reporter('Maximum posterior')
def maximum_posterior(mcmc_set):
    # Get the maximum posterior
    (max_posterior, max_posterior_position) = mcmc_set.maximum_posterior()

    # Plot the maximum posterior fit
    if hasattr(mcmc_set.chains[0], 'fit_plotting_function'):
        mcmc_set.chains[0].fit_plotting_function(
                                        position=max_posterior_position)
        filename = '%s_max_posterior_plot.png' % mcmc_set.name
        plt.savefig(filename)
    else:
        filename = None

    # Return the max posterior along with link to the plot
    return Result(max_posterior, filename)

