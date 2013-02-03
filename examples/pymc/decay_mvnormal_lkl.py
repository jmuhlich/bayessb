from pysb import *
import numpy as np
from pysb.integrate import odesolve, Solver
from pymc import Uniform, deterministic, MvNormal, Lognormal, MCMC
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
k = Uniform('k', lower=0, upper=1)
#k = Lognormal('k', mu=np.log(1e-1), tau=(1/(np.log(10)*np.log(1e3))))

# Our "model" is a deterministic random variable that is determined by the
# value of k that is passed in as an argument
@deterministic(plot=False)
def decay_model(k=k):
    # The solver object needs all of the parameters, including the initial
    # condition parameter A_0 that is not being fit; the final parameter
    # value of 1.0 is the __source_0 parameter which is required because
    # we are using a degradation rule.
    # NOTE: Make sure the parameter values passed to the solver are in
    # the right order!
    solver.run(np.array([A_0.value, k, 1.0]))
    y = solver.yobs['A_obs']
    # The decay_model variable returns a vector of predicted values for
    # the timecourse given the current value of the decay parameter k
    return y

# The "precision" matrix tau is a square matrix (with dimension len(tspan))
# that has as entries the reciprocals of the various covariances between the
# different datapoints in the timecourse. Since we only have made one
# observation (one synthetic data series), we use the known variance of 0.1
# (precision of 1 / 0.1 = 10) for the diagonals, and 0 for all other elements
tau = np.eye(len(tspan)) * 10

# Our output variable is a multivariate normal distribution with len(tspan)
# dimensions. Since it has been observed, we set observed=True, and assign
# the synthetic data ("ydata") to it as its value. The predicted trajectory
# from decay_model is given as the proposed mean of the distribution, with
# tau as the precision matrix.
timecourse = MvNormal('timecourse', mu=decay_model, tau=tau, observed=True,
                  value=ydata, plot=True)

if __name__ == '__main__':
    # Build a model
    # NOTE: Be careful to avoid namespace clashes with pysb.Model!
    pymc_model = pymc.Model([k, decay_model, timecourse])

    # Initialize an MCMC object from the model
    mcmc = MCMC(pymc_model)

    # Sample
    mcmc.sample(iter=10000, burn=5000)

    # Plot the posterior distribution of the parameter k
    plt.ion()
    Matplot.plot(mcmc)

    # Plot the original data (underlying and noisy)
    # along with the sampled trajectories
    plt.figure()
    plt.plot(tspan, ysim, color='r')
    plt.plot(tspan, ydata, color='g')
    num_timecourses = 1000
    num_iterations_sampled = mcmc.trace('decay_model')[:].shape[0]
    plt.plot(tspan, mcmc.trace('decay_model')[num_iterations_sampled -
                                              num_timecourses:,:].T,
                                              alpha=0.05, color='b')

