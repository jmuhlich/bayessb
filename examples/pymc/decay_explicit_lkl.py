from pysb import *
import numpy as np
from pysb.integrate import odesolve, Solver
from pymc import Uniform, stochastic, deterministic, MvNormal, Lognormal, MCMC
import pymc
from pymc import Matplot
from matplotlib import pyplot as plt

# Define the model: simple exponential decay from 1 to 0
Model()
Monomer('A')
Parameter('A_0', 1)
Parameter('k', 0.2)
Rule('A_decay', A() >> None, k)
Initial(A(), A_0)
Observable('A_obs', A())

# Simulate the model to generate the synthetic data
ntimes = 100; 
tspan = np.linspace(0, 40, ntimes);
ysim = odesolve(model, tspan)

# Add error to the underlying data
seed = 2
random = np.random.RandomState(seed)
sigma = 0.1; 
ydata = ysim['A_obs'] * (random.randn(len(ysim['A_obs'])) * sigma + 1);

solver = Solver(model, tspan)

# The prior distribution for our rate parameter, a stochastic variable
# with either:
# - a uniform distribution between 0 and 1, or
# - a lognormal distribution centered at 0.1, with a variance of 3
#   log10 units on either side
#k = Uniform('k', lower=0, upper=1)
k = Lognormal('k', mu=np.log(1e-1), tau=(1/(np.log(10)*np.log(1e3))))

variances = np.array([0.1] * len(tspan))

# In this approach, the "model" is a stochastic random variable, marked by
# @stochastic decorator. This is required because we need to tell PyMC that
# decay_model is our observed variable, that is, the one we wish to compare
# to data--and variables that are considered observed must be stochastic;
# see http://pymc-devs.github.com/pymc/modelbuilding.html#data
#
# So the way to think about this function is as a likelihood probability
# distribution, yielding the probability of the data given the
# model/parameters. So we can do the simulation and the error/chi-squared
# calculation in one function and return the log-likelihood.
#
# IMPORTANT! IMPORTANT! The function must return the log-likehood,
# NOT THE NEGATIVE LOG-LIKELIHOOD. This means that the sum of the errors
# actually requires a negative sign in front of it. PyMC is trying to
# maximize the log likelihood/posterior, not minimize the negative log-
# likelihood/posterior.
@stochastic(plot=False, observed=True)
def decay_model(k=k, value=ydata):
    # The solver object needs all of the parameters, including the initial
    # condition parameter A_0 that is not being fit; the final parameter
    # value of 1.0 is the __source_0 parameter which is required because
    # we are using a degradation rule.
    # NOTE: Make sure the parameter values passed to the solver are in
    # the right order!
    solver.run(np.array([A_0.value, k, 1.0]))
    # return the sum of the squared error divided by 2 * the variance
    return -np.sum(((value - solver.yobs['A_obs'])**2) / (2 * variances))

if __name__ == '__main__':
    # Build a model
    # NOTE: Be careful to avoid namespace clashes with pysb.Model!
    pymc_model = pymc.Model([k, decay_model])

    # Initialize an MCMC object from the model
    mcmc = MCMC(pymc_model)

    # Sample
    mcmc.sample(iter=15000, burn=5000, thin=10)

    # Plot the posterior distribution of the parameter k
    plt.ion()
    Matplot.plot(mcmc)

    # Plot the original data (underlying and noisy)
    # along with the sampled trajectories
    plt.figure()
    plt.plot(tspan, ysim, color='r')
    plt.plot(tspan, ydata, color='g')
    num_to_plot = 1000
    k_vals = mcmc.trace('k')[:]
    if num_to_plot > len(k_vals):
        num_to_plot = len(k_vals)

    for i in range(num_to_plot):
        solver.run(np.array([A_0.value, k_vals[i], 1.0]))
        plt.plot(tspan, solver.yobs['A_obs'], alpha=0.05, color='b')

