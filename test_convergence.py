"""A simple model and MCMC execution routine for testing the properties of the
Gelman-Rubin convergence criterion."""

from pysb import *
from pysb.integrate import odesolve
from matplotlib import pyplot as plt
import numpy as np
import bayessb
import pysb.util
import pickle

# Define a simple exponential decay model with two parameters, the initial
# value A_0 and the rate parameter k
Model()
Monomer('A')
Initial(A(), Parameter('A_0', 2))
Rule('Decay_A', A() >> None, Parameter('k', 1))
Observable('A_', A())

# Define the time span, number of steps, and other globals
tmax = 10
tspan = np.linspace(0, tmax, 100)
random_seed = 1
sigma = 0.1
nsteps = 8000
num_chains = 5

# TODO: Need to actually get real synthetic data!
#synthetic_data = pysb.util.synthetic_data(model, tspan, seed=random_seed)
synthetic_data = odesolve(model, tspan)

def do_fit(iteration):
    """Runs MCMC on the globally defined model."""

    def likelihood(mcmc, position):
        yout = mcmc.simulate(position, observables=True)
        err = np.sum((synthetic_data['A_'] - yout['A_'])**2 / (2*sigma**2))
        return err

    # Set the random number generator seed
    random = np.random.RandomState(random_seed)

    # Initialize the MCMC arguments
    opts = bayessb.MCMCOpts()
    opts.model = model
    opts.tspan = tspan
    # Because there is a degradation reaction, there is a __source_0
    # parameter in the model that we need to ignore
    opts.estimate_params = [p for p in model.parameters
                                    if p.name != '__source_0']
    # Choose MCMC start values randomly from [0, 10)
    opts.initial_values = np.random.uniform(0, 10, 2)
    opts.nsteps = nsteps
    opts.likelihood_fn = likelihood
    opts.step_fn = step
    opts.use_hessian = True
    opts.hessian_period = opts.nsteps / 10
    opts.seed = random_seed
    mcmc = bayessb.MCMC(opts)

    mcmc.run()

    # Pickle it!
    basename = 'convergence_test'
    output_basename = '%s_%d_steps_seed_%d_iter_%d' % \
                      (basename, opts.nsteps, random_seed, iteration)
    mcmc.options.likelihood_fn = None
    output_file = open('%s.pck' % output_basename, 'w')
    pickle.dump(mcmc, output_file)
    output_file.close()

    # Show best fit params
    mcmc.position = mcmc.positions[np.argmin(mcmc.likelihoods)]
    best_fit_params = mcmc.cur_params(position=mcmc.position)
    p_name_vals = zip([p.name for p in model.parameters], best_fit_params)
    print('\n'.join(['%s: %g' % (p_name_vals[i][0], p_name_vals[i][1])
                     for i in range(0, len(p_name_vals))]))

    return mcmc

def step(mcmc):
    """The function to call at every iteration. Currently just prints
    out a few progress indicators.
    """
    if mcmc.iter % 200 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, mcmc.acceptance/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)

def plot_model_data():
    """Plots a simulation of the model along with the data."""
    plt.ion()
    x = odesolve(model, tspan)
    plt.plot(tspan, x['A_'])
    plt.plot(tspan, synthetic_data['A_'])
    plt.show()

if __name__ == '__main__':
    #run_model()

    # Run a series of chains
    chain_set = []
    for i in range(0, num_chains):
        mcmc = do_fit(i)

        # Get the positions
        mixed_start = nsteps / 2
        mixed_positions = mcmc.positions[mixed_start:,:]
        #mixed_accepts = mcmc.accepts[mixed_start:]
        #mixed_accept_positions = mixed_positions[mixed_accepts]

        # Convert to linear scale
        mixed_positions_linear = 10**mixed_positions
        #mixed_accept_positions_linear = 10**mixed_accept_positions

        # Add to list
        chain_set.append(mixed_positions_linear)



