import bayessb
import numpy as np
import scipy.integrate
import itertools
import multiprocessing
from fit_1_3_standalone import build_opts


def run_chain(args):
    print 'run_chain' + str(args)
    temperature, sample = args
    opts = master_opts.copy()
    opts.thermo_temp = temperature
    opts.seed = sample
    mcmc = bayessb.MCMC(opts)
    mcmc.run()
    num_likelihoods = mcmc.acceptance / 2
    return mcmc.likelihoods[mcmc.accepts][num_likelihoods:]


if __name__ == '__main__':

    master_opts = build_opts()
    master_opts.step_fn = None
    num_temps = 16
    num_chains = 4
    temperatures = np.logspace(-3, 0, num_temps)
    inputs = itertools.product(temperatures, xrange(num_chains))

    pool = multiprocessing.Pool()
    results = pool.map(run_chain, inputs)
    pool.close()
    pool.join()

    likelihood_means = np.empty_like(temperatures)
    likelihood_stds = np.empty_like(temperatures)
    for i, temperature in enumerate(temperatures):
        likelihood_sets = []
        for c in xrange(num_chains):
            likelihood_sets.append(results[i * num_chains + c])
        likelihoods = np.hstack(likelihood_sets)
        likelihood_means[i] = likelihoods.mean()
        likelihood_stds[i] = likelihoods.std()

    num_samples = 1000
    sample_iter = itertools.imap(np.random.normal, likelihood_means,
                                 likelihood_stds, itertools.repeat(num_samples))
    # FIXME needlessly creates an intermediate list
    samples = np.array(list(sample_iter)).T
    bayes_factors = scipy.integrate.simps(samples, temperatures)
