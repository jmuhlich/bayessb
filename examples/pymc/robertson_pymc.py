from pymc import deterministic, stochastic, MvNormal, Normal, Lognormal, Uniform, \
                 MCMC
import pymc
import numpy as np
from pysb.examples.robertson import model
from pysb.integrate import odesolve, Solver
from matplotlib import pyplot as plt

# Generate the synthetic data
seed = 2
random = np.random.RandomState(seed)
sigma = 0.1;
ntimes = 20;
tspan = np.linspace(0, 40, ntimes);
ysim = odesolve(model, tspan)
ysim_array = ysim.view(float).reshape(len(ysim), -1)
yspecies = ysim_array[:, :len(model.species)]
ydata = yspecies * (random.randn(*yspecies.shape) * sigma + 1);
ysim_max = yspecies.max(0)
ydata_norm = ydata / ysim_max

solver = Solver(model, tspan)

# Set up the parameter vector for the solver
nominal_rates = [model.parameters[n].value for n in ('A_0', 'B_0', 'C_0')]

# Stochastic variables for the rate parameters.
# Given lognormal priors with their correct order-of-mag mean but with a
# variance of 10 base 10 log units
k1 = Lognormal('k1', mu=np.log(1e-2), tau=1/(np.log(10)*np.log(1e10)),
               value=1e-2, plot=True)
k2 = Lognormal('k2', mu=np.log(1e7), tau=1/(np.log(10)*np.log(1e10)),
               value=1e7, plot=True)
k3 = Lognormal('k3', mu=np.log(1e4), tau=1/(np.log(10)*np.log(1e10)),
               value=1e4, plot=True)

# The model is set up as a deterministic variable
@deterministic(plot=False)
def robertson_model(k1=k1, k2=k2, k3=k3):
    solver.run(np.concatenate((np.array([k1, k2, k3]), nominal_rates)))
    yout = solver.y / ysim_max # Normalize the simulation
    return yout.flatten()
    
# The precision (1/variance) matrix
tau = np.eye(len(tspan)*3) * 10

output = MvNormal('output', mu=robertson_model, tau=tau, observed=True,
                   value=ydata_norm.flatten())

if __name__ == '__main__':
    # Create the MCMC object and start sampling
    pymc_model = pymc.Model([k1, k2, k3, robertson_model, output])
    mcmc = MCMC(pymc_model)
    mcmc.sample(iter=10000, burn=5000, thin=5)

    # Show the pymc histograms and autocorrelation plots
    plt.ion()
    pymc.Matplot.plot(mcmc)
    plt.show()

    # Plot the original data along with the sampled trajectories
    plt.figure()
    plt.plot(tspan, ydata_norm[:,0], 'r')
    plt.plot(tspan, ydata_norm[:,1], 'g')
    plt.plot(tspan, ydata_norm[:,2], 'b')

    num_timecourses = 1000
    num_iterations_sampled = mcmc.trace('robertson_model')[:].shape[0]

    plt.plot(tspan, mcmc.trace('robertson_model')[num_iterations_sampled -
               num_timecourses:,0::3].T, alpha=0.05, color='r')
    plt.plot(tspan, mcmc.trace('robertson_model')[num_iterations_sampled -
               num_timecourses:,1::3].T, alpha=0.05, color='g')
    plt.plot(tspan, mcmc.trace('robertson_model')[num_iterations_sampled -
               num_timecourses:,2::3].T, alpha=0.05, color='b')
    
    # Show k1/k3 scatter plot
    plt.figure()
    plt.scatter(mcmc.trace('k1')[:], mcmc.trace('k3')[:])
