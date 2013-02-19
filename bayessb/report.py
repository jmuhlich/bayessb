import numpy as np
from matplotlib import pyplot as plt
from pysb.report import reporter, Result
from bayessb import convergence
from matplotlib.font_manager import FontProperties

@reporter('Number of chains')
def num_chains(mcmc_set):
    return Result(len(mcmc_set.chains), None)

@reporter('Conv. Criterion')
def convergence_criterion(mcmc_set):
    # Make plots of 
    return Result(convergence.convergence_criterion(mcmc_set), None)

@reporter('Maximum likelihood')
def maximum_likelihood(mcmc_set):
    # Get the maximum likelihood
    (max_likelihood, max_likelihood_position) = mcmc_set.maximum_likelihood()

    # Plot the maximum likelihood fit
    model = mcmc_set.chains[0].options.model
    tspan = mcmc_set.chains[0].options.tspan
    x = mcmc_set.chains[0].simulate(position=max_likelihood_position,
                                    observables=True)
    plt.figure()
    for o in model.observables:
        plt.plot(tspan, x[o.name], label=o.name)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    fontP = FontProperties() 
    fontP.set_size('small')
    plt.legend(loc='upper center', prop=fontP, ncol=5, bbox_to_anchor=(0.5, 1.1),
         fancybox=True, shadow=True)
    filename = '%s_max_likelihood_plot.png' % mcmc_set.name
    plt.savefig(filename)

    # Return the max likelihood along with link to the plot
    return Result(max_likelihood, filename)

@reporter('Maximum posterior')
def maximum_posterior(mcmc_set):
    # Get the maximum posterior
    (max_posterior, max_posterior_position) = mcmc_set.maximum_posterior()

    # Plot the maximum posterior fit
    model = mcmc_set.chains[0].options.model
    tspan = mcmc_set.chains[0].options.tspan
    x = mcmc_set.chains[0].simulate(position=max_posterior_position,
                                    observables=True)
    plt.figure()
    for o in model.observables:
        plt.plot(tspan, x[o.name], label=o.name)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    fontP = FontProperties() 
    fontP.set_size('small')
    plt.legend(loc='upper center', prop=fontP, ncol=5, bbox_to_anchor=(0.5, 1.1),
         fancybox=True, shadow=True)
    filename = '%s_max_posterior_plot.png' % mcmc_set.name
    plt.savefig(filename)

    # Return the max posterior along with link to the plot
    return Result(max_posterior, filename)


