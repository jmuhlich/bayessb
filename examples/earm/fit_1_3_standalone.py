# Fits EARM 1.3 (Gaudet et. al 2012) against a single-cell time course
# measurement of an executioner caspase reporter (a proxy for Caspase-3 activity
# i.e. PARP cleavage). The model is contained in earm_1_3_standalone.py which
# was produced via export from a PySB implementation of the model
# (pysb.examples.earm_1_3 in the PySB distribution).

import biomc
import numpy as np
import matplotlib.pyplot as plt
import os

from earm_1_3_standalone import Model


def likelihood(mcmc, position):
    """TODO"""
    param_values = mcmc.cur_params(position)
    _, yobs = model.simulate(mcmc.options.tspan, param_values, view=True)
    cparp_sim_norm = yobs['CPARP_total'] / parp_initial
    return np.sum((exp_ecrp - cparp_sim_norm) ** 2 / (2 * exp_ecrp_var ** 2))

def prior(mcmc, position):
    """TODO ...mean and variance from calculate_prior_stats"""
    return np.sum((position - prior_mean) ** 2 / ( 2 * prior_var))

def step(mcmc):
    """Print out some statistics every 20 steps"""
    if mcmc.iter % 20 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)

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
    # note: estimate_params must be sorted here, in the same order that biomc
    # maintains the position vector
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
    

# instantiate an instance of our model
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
    1.178,1.179,1.181,1.182
    ])
# index for beginning and end of MOMP (where the EC-RP signal spikes)
momp_start_idx = 75
momp_end_idx = 86

### XXX temp
for i, p in enumerate(model.parameters):
    if p.name == 'kf31':
        # original value of 1e-3 gives a much better initial fit, but starting
        # from 1e-2 helps converge to a better fit faster.
        model.parameters[i] = p._replace(value=1e-2)
    if False and p.name == 'kdeg_CPARP':
        # we process the post-momp signal to force a flat plateau, but CPARP
        # degradation makes CPARP slope down in the simulation. if we disable
        # the degradation then the simulated trajectory is much flatter.
        model.parameters[i] = p._replace(value=0)

# data was collected at evenly-spaced points over 12 hours
tspan = np.linspace(0, 12 * 3600, len(exp_ecrp))
# select the forward/reverse/catalytic parameters for estimation
kf = filter_params('kf')
kr = filter_params('kr')
kc = filter_params('kc')
estimate_params = sort_params(kf + kr + kc)
# grab the initial amount of parp, for normalizing CPARP trajectories
parp_initial = [p.value for p in model.parameters if p.name == 'PARP_0'][0]

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

# set up our MCMC options
opts = biomc.MCMCOpts()
opts.model = model
opts.tspan = tspan
opts.nsteps = 200
opts.likelihood_fn = likelihood
opts.prior_fn = prior
opts.step_fn = step
opts.estimate_params = estimate_params
# the specific value of seed isn't important, it just makes the run reproducible
opts.seed = 1  
# note that solver tolerance values are set in earm_1_3_standalone.py


# run the chain
mcmc = biomc.MCMC(opts)
mcmc.run()


# print some information about the maximum-likelihood estimate parameter set
print
print '%-10s %-12s %-12s %s' % ('parameter', 'original', 'fitted', 'log10(fit/orig)')
fitted_values = mcmc.cur_params()[mcmc.estimate_idx]
changes = np.log10(fitted_values / [p.value for p in opts.estimate_params])
for param, new_value, change in zip(opts.estimate_params, fitted_values, changes):
    values = (param.name, param.value, new_value, change)
    print '%-10s %-12.2g %-12.2g %-+6.2f' % values

# plot data and simulated cleaved PARP trajectories before and after the fit
_, yobs_initial = model.simulate(tspan, [p.value for p in model.parameters])
_, yobs_final = model.simulate(tspan, mcmc.cur_params(mcmc.position))
plt.plot(tspan, exp_ecrp, 'o', mec='red', mfc='none')
plt.plot(tspan, yobs_initial['CPARP_total']/parp_initial, 'b');
plt.plot(tspan, yobs_final['CPARP_total']/parp_initial, 'k');
plt.xlim(0, 2.5e4)
plt.legend(('data', 'original model', 'fitted model'), loc='upper left')
plt.show()
