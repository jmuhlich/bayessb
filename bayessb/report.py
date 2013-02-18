import numpy as np
from pysb.report import reporter

@reporter('Maximum likelihood')
def maximum_likelihood(mcmc_set):
    return mcmc_set.maximum_likelihood()
    #return np.nanmin(mcmc.likelihoods)

@reporter('Maximum posterior')
def maximum_posterior(mcmc_set):
    return mcmc_set.maximum_posterior()
    #return np.nanmin(mcmc.posteriors)

