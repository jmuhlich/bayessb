#!/usr/bin/env python
"""Calculate evidence using EARM 1.3 and single-cell data, using thermodynamic
integration. Will use all available local CPU cores."""


from __future__ import division
import bayessb
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import itertools
import multiprocessing
import sys
from fit_1_3_standalone import build_opts


def run_chain(args):
    temperature, sample = args
    # Start with master option set, then override temperature and seed
    opts = master_opts.copy()
    opts.thermo_temp = temperature
    opts.seed = sample
    # Build an MCMC object and run it
    mcmc = bayessb.MCMC(opts)
    mcmc.run()
    # Return likelihoods of accepted moves in the latter half of the walk
    num_likelihoods = mcmc.acceptance // 2
    return mcmc.likelihoods[mcmc.accepts][num_likelihoods:]


def print_progress_bar(fraction):
    percentage = fraction * 100
    bar_size = int(fraction * 50)
    sys.stdout.write('%3d%% [%-51s]\r' % (percentage, '=' * bar_size + '>'))
    sys.stdout.flush()


if __name__ == '__main__':

    print "Performing thermodynamic integration:"

    # Build master option set
    master_opts = build_opts()
    # Don't print anything out, as we'll have many simultaneous workers
    master_opts.step_fn = None
    # Choose the number of temperatures to sample and chains to run at each
    num_temps = 8
    num_chains = 3
    # Sample temperatures on a log scale, from 1e-3 to 1
    temperatures = np.logspace(-3, 0, num_temps)
    # Produce tuples of input arguments to run_chain
    inputs = itertools.product(temperatures, xrange(num_chains))

    # Launch a parallel processing pool to run the chains
    pool = multiprocessing.Pool()
    result = pool.map_async(run_chain, inputs)
    # Print a progress bar while the pool is still working
    num_chunks = result._number_left
    while not result.ready():
        try:
            outputs = result.get(timeout=1)
        except multiprocessing.TimeoutError:
            pass
        except KeyboardInterrupt as e:
            pool.terminate()
            raise
        print_progress_bar((num_chunks - result._number_left) / num_chunks),
    print
    pool.close()
    pool.join()

    # Calculate mean of likelihoods from all chains at each temperature, and
    # standard deviation on the means from each chain at each temperature
    likelihood_means = np.empty_like(temperatures)
    likelihood_stds = np.empty_like(temperatures)
    for i, temperature in enumerate(temperatures):
        likelihood_sets = []
        for c in xrange(num_chains):
            # Extract the right likelihood vectors from the pool output,
            # negating the values to obtain positive log-likelihood values
            likelihood_sets.append(-1 * outputs[i * num_chains + c])
        # Mean of all likelihood values
        likelihood_means[i] = np.mean(np.hstack(likelihood_sets))
        # Standard deviation on the means
        likelihood_stds[i] = np.std(map(np.mean, likelihood_sets))

    # Produce a number of sampled trajectories from the means and stds
    num_samples = 1000
    sample_iter = itertools.imap(np.random.normal, likelihood_means,
                                 likelihood_stds, itertools.repeat(num_samples))
    # FIXME this needlessly creates an intermediate list
    samples = np.array(list(sample_iter)).T

    # Integrate sampled trajectories to obtain log-evidence i.e.
    # log(P(Data|Model))
    log_evidences = scipy.integrate.simps(samples, temperatures)
    # Plot histogram of evidence values
    counts, bins, _ = plt.hist(np.exp(log_evidences), bins=40)
    print 'Histogram of evidence values:'
    for b, c in zip(bins, counts):
        print '%-8.3g: %d' % (b, c)
    plt.xlabel('Evidence')
    plt.ylabel('Count')
    plt.show()
