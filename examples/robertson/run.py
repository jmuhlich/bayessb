import mcmc_hessian
from pysb.examples.robertson import model
from pysb.integrate import odesolve
import numpy
import matplotlib.pyplot as plt


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
    return numpy.sum((ydata_norm - yout_norm) ** 2 / (2 * sigma ** 2))

def prior(mcmc, position):
    mean = math.log10([1e-2, 1e7])
    var = [100, 100]
    return numpy.sum((position - means) ** 2 / ( 2 * var))

def step(mcmc):
    if mcmc.iter % 20 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, mcmc.acceptance/(mcmc.iter+1), mcmc.accept_likelihood,
             mcmc.accept_prior, mcmc.accept_posterior)


scenario = 1

opts = mcmc_hessian.MCMCOpts()
opts.model = model
opts.tspan = tspan

# A few estimation scenarios:
if scenario == 1:
    # estimate all parameters from wild guesses (orders of magnitude off)
    opts.estimate_params = model.parameters
    opts.initial_values = [1e-4, 1e3, 1e6, 1e-1, 1e-1, 1e-1]
elif scenario == 2:
    # estimate rates only (not initial conditions) from wild guesses
    opts.estimate_params = [p for p in model.parameters if p.name.startswith('k') ]
    opts.initial_values = [1e-4, 1e3, 1e6]
else:
    raise RuntimeError("unknown scenario number")

opts.nsteps = 5000
opts.likelihood_fn = likelihood
opts.step_fn = step
opts.use_hessian = True
opts.hessian_period = opts.nsteps / 10
opts.seed = seed
mcmc = mcmc_hessian.MCMC(opts)

mcmc.run()

print
print '%-10s %-12s %-12s %-12s' % ('param', 'actual', 'fitted', '% error')
fitted_values = mcmc.cur_params()[mcmc.estimate_idx]
for param, new_value in zip(opts.estimate_params, fitted_values):
    error = abs(1 - param.value / new_value) * 100
    values = (param.name, param.value, new_value, error)
    print '%-10s %-12g %-12g %-12g' % values

colors = ('r', 'g', 'b')
real_lines = plt.plot(tspan, yspecies / ysim_max)
data_lines = plt.plot(tspan, ydata_norm)
sim_lines = plt.plot(tspan, mcmc.simulate() / ysim_max)
for rl, dl, sl, c in zip(real_lines, data_lines, sim_lines, colors):
    rl.set_color(c)
    dl.set_color(c)
    sl.set_color(c)
    rl.set_linestyle('--')
    dl.set_linestyle(':')
    dl.set_marker('x')
plt.show()
