import numpy as np
from pysb.report import reporter

@reporter('Maximum likelihood')
def maximum_likelihood(mcmc):
    return np.nanmin(mcmc.likelihoods)

@reporter('Maximum posterior')
def maximum_posterior(mcmc):
    return np.nanmin(mcmc.posteriors)

