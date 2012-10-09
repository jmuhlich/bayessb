import biomc
import pysb.integrate
import numpy
import matplotlib.pyplot as plt
import os

from pysb.examples.earm_1_0 import model


seed = 2

data_filename = os.path.join(os.path.dirname(__file__), 'experimental_data.npy')
ydata_norm = numpy.load(data_filename)
sigma = 0.2
t_end = 5.5 * 3600  # 5.5 hours, in seconds
tspan = numpy.linspace(0, t_end, ydata_norm.shape[0])
obs_names = ['tBid_total', 'CPARP_total', 'cSmac_total']

def normalize(trajectories):
    ymin = trajectories.min(0)
    ymax = trajectories.max(0)
    return (trajectories - ymin) / (ymax - ymin)

def extract_records(recarray, names):
    return numpy.vstack([recarray[name] for name in names]).T

def likelihood(mcmc, position):
    ysim = mcmc.simulate(position, observables=True)
    ysim_array = extract_records(ysim, obs_names)
    ysim_norm = normalize(ysim_array)
    return numpy.sum((ydata_norm - ysim_norm) ** 2 / (2 * sigma ** 2))

#def prior(mcmc, position):
#    mean = math.log10([1e-2, 1e7])
#    var = [100, 100]
#    return numpy.sum((position - means) ** 2 / ( 2 * var))

def step(mcmc):
    if mcmc.iter % 20 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, mcmc.acceptance/(mcmc.iter+1), mcmc.accept_likelihood,
             mcmc.accept_prior, mcmc.accept_posterior)

scenario = 3

opts = biomc.MCMCOpts()
opts.model = model
opts.tspan = tspan

# A few estimation scenarios:
if scenario == 1:
    # estimate all parameters starting from values stored in model
    opts.estimate_params = model.parameters
elif scenario == 2:
    # estimate rates only (not initial conditions)
    opts.estimate_params = [p for p in model.parameters if not p.name.endswith('_0') ]
elif scenario == 3:
    # estimate first 10 rates
    opts.estimate_params = [p for p in model.parameters if not p.name.endswith('_0') ][0:10]
else:
    raise RuntimeError("unknown scenario number")

opts.nsteps = 320
opts.likelihood_fn = likelihood
opts.step_fn = step
opts.use_hessian = True
opts.hessian_period = opts.nsteps / 3
opts.seed = seed
mcmc = biomc.MCMC(opts)

mcmc.run()

print
print '%-10s %-12s %-12s %s' % ('parameter', 'actual', 'fitted', 'log10(fit/actual)')
fitted_values = mcmc.cur_params()[mcmc.estimate_idx]
for param, new_value in zip(opts.estimate_params, fitted_values):
    change = np.log10(new_value / param.value)
    values = (param.name, param.value, new_value, change)
    print '%-10s %-12.2g %-12.2g %-+6.2f' % values

colors = ('r', 'g', 'b')
yreal = pysb.integrate.odesolve(model, tspan)
yreal_array = extract_records(yreal, obs_names)
ysim_array = extract_records(mcmc.simulate(observables=True), obs_names)
real_lines = plt.plot(tspan, normalize(yreal_array))
data_lines = plt.plot(tspan, ydata_norm)
sim_lines = plt.plot(tspan, normalize(ysim_array))
for rl, dl, sl, c in zip(real_lines, data_lines, sim_lines, colors):
    rl.set_color(c)
    dl.set_color(c)
    sl.set_color(c)
    rl.set_linestyle('--')
    dl.set_linestyle(':')
    dl.set_marker('x')
plt.show()
