import biomc
import pysb.integrate
import numpy as np
import matplotlib.pyplot as plt
import os
from pysb.core import ComponentSet
from pysb.util import get_param_num

from earm_1_3_standalone import Model

# TODO: make this work like RUN_ME.m
# http://www.cdpcenter.org/wordpress/wp-content/uploads/2011/08/RUN_ME.m

def likelihood(mcmc, position):
    param_values = mcmc.cur_params(position)
    _, yobs = model.simulate(mcmc.options.tspan, param_values, view=True)
    cparp_sim_norm = yobs['CPARP_total'] / parp_initial
    return np.sum((exp_ecrp - cparp_sim_norm) ** 2 / (2 * exp_ecrp_var ** 2))

def prior(mcmc, position):
    return np.sum((position - prior_mean) ** 2 / ( 2 * prior_var))

def step(mcmc):
    if mcmc.iter % 20 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, mcmc.acceptance/(mcmc.iter+1), mcmc.accept_likelihood,
             mcmc.accept_prior, mcmc.accept_posterior)

def filter_params(prefix):
    """Return a list of model parameters whose names begin with a prefix"""
    return [p for p in model.parameters if p.name.startswith(prefix)]

def sort_params(params):
    """Sort parameters to be consistent with the model's parameter order"""
    return [p for p in model.parameters if p in params]

def param_stats(params):
    """Calculate mean and variance of log10-transformed parameter values"""
    transformed_values = np.log10([p.value for p in params])
    return np.mean(transformed_values), np.var(transformed_values)

def calculate_prior_stats():
    """Create vectors of means and variances for calculating the prior

    Mean and variance are calculated separately across each of four classes:
    kf-fast, kf-slow, kr, and kc."""
    kf_fast = []
    kf_slow = []
    for p in kf:
        if p.value >= 1e-3:
            kf_fast.append(p)
        else:
            kf_slow.append(p)
    mean_kf_fast, _ = param_stats(kf_fast)
    mean_kf_slow, _ = param_stats(kf_slow)
    mean_kr, _ = param_stats(kr)
    mean_kc, var_kc = param_stats(kc)
    mean = np.empty(len(estimate_params))
    var = np.empty_like(mean)
    var.fill(2.0)
    for i, p in enumerate(estimate_params):
        if p in kf_fast:
            mean[i] = mean_kf_fast
        elif p in kf_slow:
            mean[i] = mean_kf_slow
        elif p in kr:
            mean[i] = mean_kr
        elif p in kc:
            mean[i] = mean_kc
            var[i] = 2 * var_kc
        else:
            raise RuntimeError("unexpected parameter: " + str(p))
    return mean, var
    

model = Model()

# EC-RP trajectory (experimental data, arbitrary units)
exp_ecrp = np.array([
    0.4820,0.4540,0.4500,0.4430,0.4440,0.4420,0.4430,0.4470,0.4390,0.4490,0.4450,
    0.4450,0.4390,0.4410,0.4370,0.4450,0.4470,0.4390,0.4430,0.4410,0.4390,0.4410,0.4420,
    0.4430,0.4490,0.4420,0.4460,0.4480,0.4500,0.4420,0.4460,0.4450,0.4520,0.4480,0.4520,
    0.4460,0.4470,0.4500,0.4500,0.4470,0.4490,0.4570,0.4500,0.4530,0.4550,0.4510,0.4560,
    0.4560,0.4600,0.4530,0.4540,0.4510,0.4590,0.4570,0.4490,0.4510,0.4640,0.4580,0.4560,
    0.4610,0.4620,0.4600,0.4590,0.4640,0.4640,0.4650,0.4750,0.4760,0.4760,0.4780,0.4760,
    0.4810,0.4830,0.4930,0.4870,0.4890,0.4890,0.5180,0.6120,0.7280,0.8050,0.9120,0.9570,
    0.9980,1.025,1.025,1.068,1.074,1.061,1.056,1.049,1.064,1.092,1.086,1.044,1.023,1.062,
    1.058,1.077,1.081,1.068,1.066,1.078,1.103,1.116,1.099,1.118,1.087,1.086,1.066,1.076,
    1.075,1.069,1.080,1.077,1.086,1.079,1.088,1.084,1.092,1.088,1.099,1.093,1.108,1.101,
    1.104,1.106,1.099,1.110,1.102,1.104,1.111,1.113,1.118,1.117,1.118,1.114,1.131,1.140,
    1.131,1.135,1.143,1.144,1.146,1.145,1.146,1.137,1.149,1.164,1.157,1.150,1.148,1.151,
    1.164,1.148,1.157,1.163,1.141,1.152,1.151,1.159,1.164,1.154,1.152,1.141,1.143,1.163,
    1.158,1.152,1.144,1.151,1.156,1.154,1.151,1.150,1.148,1.150,1.150,1.147,1.148,1.161,
    1.154,1.160,1.158,1.160,1.140,1.149,1.152,1.161,1.152,1.158,1.156,1.154,1.156,1.157,
    1.148,1.156,1.159,1.160,1.160,1.150,1.153,1.163,1.156,1.171,1.147,1.166,1.155,1.159,
    1.163,1.157,1.154,1.154,1.159,1.151,1.162,1.167,1.166,1.159,1.160,1.170,1.165,1.171,
    1.179,1.164,1.176,1.175,1.178,1.175,1.170,1.178,1.173,1.168,1.168,1.169,1.174,1.177,
    1.178,1.179,1.181,1.182,1.180,1.173,1.185,1.189,1.173,1.167,1.176,1.176,1.182,1.175,
    1.186,1.188,1.169,1.189,1.179,1.183,1.192,1.194,1.188,1.169,1.182,1.188,1.185,1.185,
    1.195,1.190,1.177,1.187,1.187,1.191,1.192,1.204,1.193,1.199,1.193,1.196,1.190,1.189,
    1.196,1.196,1.198,1.186,1.190,1.190,1.198,1.198,1.186,1.190,1.197,1.191,1.208,1.197,
    1.206,1.197,1.209,1.198,1.200,1.196,1.212,1.200,1.205,1.205,1.205,1.206,1.198,1.209,
    1.207,1.206,1.203,1.190,1.211,1.200,1.204,1.202,1.188,1.207,1.212,1.204,1.203,1.202,
    1.199,1.202,1.209,1.210,1.199,1.199,1.218,1.216,1.213,1.197,1.207,1.210,1.224,1.226,
    1.226,1.202,1.216,1.186,1.209,1.203,1.195,1.194,1.199,1.209,1.198,1.213,1.194,1.195,
    1.203,1.194,1.160,1.192
    ])
# index for beginning and end of MOMP (where the EC-RP signal spikes)
momp_start_idx = 75
momp_end_idx = 86
# data was collected at evenly-spaced points over 12 hours
tspan = np.linspace(0, 12 * 3600, len(exp_ecrp))
kf = filter_params('kf')
kr = filter_params('kr')
kc = filter_params('kc')
estimate_params = sort_params(kf + kr + kc)
parp_initial = [p.value for p in model.parameters if p.name == 'PARP_0'][0]

### XXX temp
for i, p in enumerate(model.parameters):
    if p.name == 'kf31':
        model.parameters[i] = p._replace(value=1e-2)
    #if p.name == 'kdeg_CPARP':
    #    model.parameters[i] = p._replace(value=0)

prior_mean, prior_var = calculate_prior_stats()

# clean up the noise in the latter part of the EC-RP signal and rescale to 0-1
# ----------
# shift the range down to begin at 0
exp_ecrp -= exp_ecrp.min()
# take discrete difference
ecrp_dd = np.diff(exp_ecrp)
# get first index after momp start where the signal drops (i.e. dd is negative)
plateau_idx = [i for i in range(momp_start_idx, len(ecrp_dd)) if ecrp_dd[i]<0][0]
# rescale the portion before that index to a max of 1
exp_ecrp[:plateau_idx] /= exp_ecrp[plateau_idx]
# clamp the latter portion directly to 1
exp_ecrp[plateau_idx:] = 1.0

# set up a vector of variance values for the ecrp signal
exp_ecrp_var = np.empty_like(tspan)
# start with a single value for the variance at all time points
exp_ecrp_var[:] = 0.0272
# use a higher value for the switching window to reflect less certainty there
exp_ecrp_var[momp_start_idx:momp_end_idx+1] = 0.1179

opts = biomc.MCMCOpts()
opts.model = model
opts.tspan = tspan
opts.nsteps = 100
opts.likelihood_fn = likelihood
opts.prior_fn = prior
opts.step_fn = step
opts.estimate_params = estimate_params
# the specific value of seed isn't important, it just makes the run reproducible
opts.seed = 1  


mcmc = biomc.MCMC(opts)
mcmc.run()

_, yobs_initial = model.simulate(tspan, [p.value for p in model.parameters])
_, yobs_final = model.simulate(tspan, mcmc.cur_params(mcmc.position))
plt.plot(exp_ecrp, 'o', mec='red', mfc='none')
plt.plot(yobs_initial['CPARP_total']/parp_initial, 'b');
plt.plot(yobs_final['CPARP_total']/parp_initial, 'k');
plt.legend(('data', 'original model', 'fitted model'), loc='lower right')
plt.show()

"""

print
print '%-10s %-12s %-12s %-12s' % ('param', 'actual', 'fitted', '% error')
fitted_values = mcmc.cur_params()[mcmc.estimate_idx]
for param, new_value in zip(opts.estimate_params, fitted_values):
    error = abs(1 - param.value / new_value) * 100
    values = (param.name, param.value, new_value, error)
    print '%-10s %-12g %-12g %-12g' % values

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
"""
