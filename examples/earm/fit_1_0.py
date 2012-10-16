# Fits EARM 1.0 (Albeck et. al 2008) against single-cell measurements
# of caspase reporters and MOMP timing.

import bayessb
import pysb.integrate
import numpy
import matplotlib.pyplot as plt
import os
import itertools

from pysb.examples.earm_1_0 import model


def normalize(trajectories):
    """Rescale a matrix of model trajectories to 0-1"""
    ymin = trajectories.min(0)
    ymax = trajectories.max(0)
    return (trajectories - ymin) / (ymax - ymin)

def extract_records(recarray, names):
    """Convert a record-type array and list of names into a float array"""
    return numpy.vstack([recarray[name] for name in names]).T

def likelihood(mcmc, position):
    """Distance between model trajectories and experimental data"""
    ysim = mcmc.simulate(position, observables=True)
    ysim_array = extract_records(ysim, obs_names)
    ysim_norm = normalize(ysim_array)
    return numpy.sum((ydata_norm - ysim_norm) ** 2 / (2 * exp_var ** 2))

def prior(mcmc, position):
    """Distance to original parameter values"""
    return numpy.sum((position - prior_mean) ** 2 / ( 2 * prior_var))

def step(mcmc):
    """Print out some statistics every 20 steps"""
    if mcmc.iter % 20 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)

# data is pre-processed to filter noise and rescale the values to 0-1
data_filename = os.path.join(os.path.dirname(__file__), 'experimental_data.npy')
ydata_norm = numpy.load(data_filename)
exp_var = 0.2  # take variance of experimental measurements to be constant

t_end = 5.5 * 3600  # 5.5 hours, in seconds
tspan = numpy.linspace(0, t_end, len(ydata_norm))

obs_names = ['tBid_total', 'CPARP_total', 'cSmac_total']

opts = bayessb.MCMCOpts()
opts.model = model
opts.tspan = tspan

scenario = 1

# A few estimation scenarios:
if scenario == 1:
    # estimate rates only (not initial conditions)
    opts.estimate_params = model.parameters_rules()
elif scenario == 2:
    # estimate first 10 rates and use hessian
    opts.estimate_params = model.parameters_rules()[0:10]
    # Warning: hessian-guidance is expensive when fitting many parameters -- the
    # time to calculate the hessian increases with the square of the number of
    # parameters to fit!
    opts.use_hessian = True
    opts.hessian_period = opts.nsteps / 3
else:
    raise RuntimeError("unknown scenario number")

# values for prior calculation
prior_mean = [p.value for p in opts.estimate_params]
prior_var = 6.0

opts.nsteps = 1000
opts.likelihood_fn = likelihood
opts.prior_fn = prior
opts.step_fn = step
opts.seed = 1
opts.atol=1e-6
opts.rtol=1e-3
mcmc = bayessb.MCMC(opts)

mcmc.run()

# print some information about the maximum-likelihood estimate parameter set
print
print '%-10s %-12s %-12s %s' % ('parameter', 'actual', 'fitted', 'log10(fit/actual)')
fitted_values = mcmc.cur_params()[mcmc.estimate_idx]
for param, new_value in zip(opts.estimate_params, fitted_values):
    change = numpy.log10(new_value / param.value)
    values = (param.name, param.value, new_value, change)
    print '%-10s %-12.2g %-12.2g %-+6.2f' % values

# plot data and simulated cleaved PARP trajectories before and after the fit
colors = ('r', 'g', 'b')
patterns = ('k--', 'k', 'k:x')
# generate a legend with the deconvolved colors and styles
for style in colors + patterns:
    plt.plot(numpy.nan, numpy.nan, style)
plt.legend(('A', 'B', 'C', 'initial', 'final', 'data'), loc='lower right')
# simulate initial and final trajectories and plot those along with the data
yinitial = pysb.integrate.odesolve(model, tspan)
yinitial_array = extract_records(yinitial, obs_names)
yfinal_array = extract_records(mcmc.simulate(observables=True), obs_names)
initial_lines = plt.plot(tspan, normalize(yinitial_array))
data_lines = plt.plot(tspan, ydata_norm)
final_lines = plt.plot(tspan, normalize(yfinal_array))
for il, dl, fl, c in zip(initial_lines, data_lines, final_lines, colors):
    il.set_color(c)
    dl.set_color(c)
    fl.set_color(c)
    il.set_linestyle('--')
    dl.set_linestyle(':')
    dl.set_marker('x')
plt.show()
