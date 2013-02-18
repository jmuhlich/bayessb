import numpy as np
from pysb.report import reporter
from bayessb import convergence

@reporter('Number of chains')
def num_chains(mcmc_set):
    return len(mcmc_set.chains)

@reporter('Conv. Criterion')
def convergence_criterion(mcmc_set):
    return convergence.convergence_criterion(mcmc_set)

@reporter('Maximum likelihood')
def maximum_likelihood(mcmc_set):
    return mcmc_set.maximum_likelihood()
    #return np.nanmin(mcmc.likelihoods)

@reporter('Maximum posterior')
def maximum_posterior(mcmc_set):
    return mcmc_set.maximum_posterior()
    #return np.nanmin(mcmc.posteriors)



