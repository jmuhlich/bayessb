import mcmc_hessian
from pysb.examples.robertson import model
from pysb.integrate import odesolve
import numpy
import matplotlib.pyplot as plt
import sys

scenario = 2
if len(sys.argv) > 1:
    scenario = int(sys.argv[1])

seed = 2
random = numpy.random.RandomState(seed)
sigma = 0.1;
ntimes = 20;
tspan = numpy.linspace(0, 40, ntimes);
ysim = odesolve(model, tspan)
ysim_array = ysim.view().reshape(len(tspan), len(ysim.dtype))
yspecies = ysim_array[:, :len(model.species)]
ydata = yspecies * (random.randn(*yspecies.shape) * sigma + 1);
ysim_max = yspecies.max(0)
ydata_norm = ydata / ysim_max

def likelihood(mcmc, position):
    yout = mcmc.simulate(position)
    yout_norm = yout / ysim_max
    if scenario == 3:
        # fit to "perfect" data
        ret = numpy.sum((yspecies / ysim_max - yout_norm) ** 2 / (2 * sigma ** 2))
    else:
        # fit to noisy data
        ret = numpy.sum((ydata_norm - yout_norm) ** 2 / (2 * sigma ** 2))
    return ret

def prior(mcmc, position):
    if scenario == 1:
        est = [1e-2, 1e7, 1e4, 1, -5, -5]
    elif scenario == 2 or scenario == 3:
        est = [1e-2, 1e7, 1e4]
    elif scenario == 4:
        est = [1e-2, 1e7]
    mean = numpy.log10(est)
    var = 10
    return numpy.sum((position - mean) ** 2 / ( 2 * var))

def step(mcmc):
    if mcmc.iter % 20 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1), mcmc.accept_likelihood,
             mcmc.accept_prior, mcmc.accept_posterior)

def print_fit(position):
    new_values = 10 ** position
    print
    print '%-10s %-12s %-12s %-12s' % ('param', 'actual', 'fitted', '% error')
    for param, new_value in zip(opts.estimate_params, new_values):
        error = abs(1 - param.value / new_value) * 100
        values = (param.name, param.value, new_value, error)
        print '%-10s %-12g %-12g %-12g' % values

def plot_fit(position):
    plt.figure()
    colors = ('r', 'g', 'b')
    labels = ('A', 'B', 'C')
    real_lines = plt.plot(tspan, yspecies / ysim_max)
    data_lines = plt.plot(tspan, ydata_norm)
    sim_lines = plt.plot(tspan, mcmc.simulate(position) / ysim_max)
    for rl, dl, sl, c, l in zip(real_lines, data_lines, sim_lines, colors,
                                labels):
        rl.set_color(c)
        dl.set_color(c)
        sl.set_color(c)
        rl.set_linestyle('--')
        dl.set_linestyle(':')
        dl.set_marker('x')
        sl.set_label(l)
    plt.legend(loc='lower right')
    plt.show()


print "Running scenario", scenario
print "=================="

opts = mcmc_hessian.MCMCOpts()
opts.model = model
opts.tspan = tspan

# Note: actual parameter values are [4e-2, 3e7, 1e4, 1, 0, 0]

# A few estimation scenarios:
if scenario == 1:
    # estimate all parameters from wild guesses (orders of magnitude off)
    opts.estimate_params = model.parameters
    opts.initial_values = [1e-4, 1e3, 1e6, 1e-1, 1e-1, 1e-1]
elif scenario == 2 or scenario == 3:
    # estimate rates only (not initial conditions) from wild guesses
    opts.estimate_params = [p for p in model.parameters if p.name.startswith('k') ]
    opts.initial_values = [1e-4, 1e3, 1e6]
elif scenario == 4:
    # estimate k1 and k2 only
    opts.estimate_params = [model.parameters['k1'], model.parameters['k2']]
    opts.initial_values = [1e-4, 1e3]
else:
    raise RuntimeError("unknown scenario number")

opts.nsteps = 10000
opts.likelihood_fn = likelihood
opts.prior_fn = prior
opts.step_fn = step
opts.use_hessian = True
opts.hessian_period = opts.nsteps / 10
opts.seed = seed
mcmc = mcmc_hessian.MCMC(opts)

mcmc.run()

estimate = numpy.median(mcmc.positions[mcmc.accepts], 0)

print_fit(estimate)
plot_fit(estimate)
