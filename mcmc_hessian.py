import numpy as np
import math
import pysb.integrate

class MCMC(object):

    def __init__(self, options):
        self.options = self.validate(options)
        self.initial_values = None
        self.initial_position = None
        self.position = None  # log-transform of "actual" phase space
        self.test_position = None
        self.num_estimate = None
        self.estimate_idx = None
        self.acceptance = None
        self.T_decay = None
        self.T = None
        self.sig_value = 1
        self.iter = 0
        self.start_iter = 0
        self.ode_options = {}

        # likelihood, prior, and posterior are also all log-transformed
        self.initial_likelihood = None
        self.initial_prior = None
        self.initial_posterior = None
        self.accept_prior = None
        self.accept_likelihood = None
        self.accept_posterior = None
        self.test_likelihood = None
        self.test_prior = None
        self.test_posterior = None

        self.delta_lkls = None
        self.delta_prior = None
        self.hessian = None

        # "logs" for some values we'd like to keep track of across iterations
        self.positions = None
        self.priors = None
        self.likelihoods = None
        self.posteriors = None
        self.alphas = None
        self.sigmas = None
        self.delta_posteriors = None
        self.ts = None
        self.accepts = None
        self.rejects = None
        self.total_times = None
        self.hessians = None
    
    def run(self):
        self.initialize()
        self.estimate()
        
    def validate(self, options):
        """Validates options and applies some defaults, and returns the
        resulting options dict."""

        if not options.model:
            raise Exception("model must be a PySB model")

        if not options.estimate_params or not len(options.estimate_params):
            raise Excepction("estimate_params must contain a list of parameters")
            
        # clamp hessian_period to actual number of steps
        if options.use_hessian:
            options.hessian_period = min(options.hessian_period, options.nsteps)
        else:
            options.hessian_period = np.inf

        if options.anneal_length is None:
            # default for anneal_length if unspecified
            if options.use_hessian:
                # if using hessian, anneal until we start using it
                options.anneal_length = options.hessian_period
            else:
                # otherwise, anneal for 10% of the run
                options.anneal_length = np.floor(options.nsteps * 0.10)
        else:
            # clamp it to actual number of steps
            options.anneal_length = min(options.anneal_length, options.nsteps)
            
        # default for sigma_adj_interval if unspecified
        if options.sigma_adj_interval is None:
            # default to 10 adjustments throughout the annealing phase
            options.sigma_adj_interval = max(int(options.anneal_length / 10), 1)
            
        return options
        
    def initialize(self):
        # create list of starting values from initial parameter values given by
        # user. vector only contains values which are to be estimated!
        self.num_estimate = len(self.options.estimate_params)
        if self.options.initial_values is not None:
            self.initial_values = self.options.initial_values
        else:
            # if no explicit values given, take values from model
            self.initial_values = [p.value for p in self.options.estimate_params]
        # indices of parameters to be estimated
        self.estimate_idx = [i for i, p in enumerate(self.options.model.parameters)
                             if p in self.options.estimate_params]
            
        # we actually work in a log-transformed phase space
        self.initial_position = np.log10(self.initial_values)
        self.position = self.initial_position
            
        self.initial_posterior, self.initial_prior, self.initial_likelihood = \
            self.calculate_posterior(self.initial_position)

        self.accept_prior = self.initial_prior
        self.accept_likelihood = self.initial_likelihood
        self.accept_posterior = self.initial_posterior

        self.T_decay = -math.log10(1e-6) / self.options.anneal_length;
            
        self.ode_options = {};
        if self.options.reltol is not None:
            self.ode_options['reltol'] = self.options.reltol
        if self.options.abstol is not None:
            self.ode_options['abstol'] = self.options.abstol

        self.start_iter = 0;
        self.acceptance = 0;
        self.T = self.options.T_init;
        self.delta_posteriors = np.empty(self.options.nsteps)
        self.ts = np.empty(self.options.nsteps)
        self.priors = np.empty(self.options.nsteps)
        self.likelihoods = np.empty(self.options.nsteps)
        self.posteriors = np.empty(self.options.nsteps)
        self.positions = np.empty((self.options.nsteps, self.num_estimate))
        self.alphas = np.empty(self.options.nsteps)
        self.sigmas = np.empty(self.options.nsteps)
        self.accepts = np.zeros(self.options.nsteps, dtype=bool)
        self.rejects = np.zeros(self.options.nsteps, dtype=bool)
        self.total_times = np.empty(self.options.nsteps)
        # FIXME only store distinct hessians -- it doesn't change on every iter
        self.hessians = np.empty((self.options.nsteps, self.num_estimate, self.num_estimate))

    def estimate(self):
        # this is the heart of the algorithm

        self.iter = self.start_iter;
        while self.iter < self.options.nsteps:

            # update hessian
            if self.options.use_hessian and self.iter % self.options.hessian_period == 0:
                try:
                    self.hessian = self.calculate_hessian()
                    print 'updating hessian (iter=%d)' % self.iter
                except Exception as e:
                    # don't update if new hessian failed to calculate properly
                    print 'keeping previous hessian (iter=%d)' % self.iter

            # choose test position and calculate posterior there
            self.test_position = self.generate_new_position()
            (self.test_posterior, self.test_prior, self.test_likelihood) = \
                self.calculate_posterior(self.test_position)

            # ------------------METROPOLIS-HASTINGS ALGORITHM-------------------
            delta_posterior = self.test_posterior - self.accept_posterior
            if math.e ** delta_posterior < 1:
                self.accept_move()
            else:
                alpha = np.random.random()
                self.alphas[self.iter] = alpha;  # log the alpha value
                if math.e ** (-delta_posterior/self.T) > alpha:
                    self.accept_move()
                else:
                    self.reject_move()

            # -------ADJUSTING SIGMA & TEMPERATURE (ANNEALING)--------
            if self.iter < self.options.anneal_length \
               and self.iter % self.options.sigma_adj_interval == 0:
                if self.acceptance / (self.iter + 1) < self.options.accept_rate_target:
                    if self.sig_value > self.options.sigma_min:
                        self.sig_value -= self.options.sigma_step
                    elif self.sig_value < self.options.sigma_max:
                        self.sig_value += self.options.sigma_step
                self.T = 1 + (self.options.T_init - 1) * math.e ** (-self.iter * self.T_decay);
                
            # log some interesting variables
            self.positions[self.iter,:] = self.test_position
            self.priors[self.iter] = self.test_prior
            self.likelihoods[self.iter] = self.test_likelihood
            self.posteriors[self.iter] = self.test_posterior
            self.delta_posteriors[self.iter] = delta_posterior
            self.sigmas[self.iter] = self.sig_value
            self.ts[self.iter] = self.T
            self.hessians[self.iter,:,:] = self.hessian;
                
            # call user-callback step function
            if self.options.step_fn:
                self.options.step_fn(self)
            
            self.iter += 1
        
    def accept_move(self):
        self.accept_prior = self.test_prior
        self.accept_likelihood = self.test_likelihood
        self.accept_posterior = self.test_posterior
        self.position = self.test_position
        self.acceptance += 1
        self.accepts[self.iter] = 1

    def reject_move(self):
        self.rejects[self.iter] = 1;

    def simulate(self, position=None):
        if position is None:
            position = self.position
        ysim = pysb.integrate.odesolve(self.options.model, self.options.tspan, self.cur_params(position))
        ysim_array = ysim.view().reshape(len(self.options.tspan), len(ysim.dtype))
        yspecies = ysim_array[:, :len(self.options.model.species)]
        return yspecies

    def cur_params(self, position=None):
        """Return the parameter values corresponding to a position in phase
        space."""
        if position is None:
            position = self.position
        # start with the original values
        values = np.array([p.value for p in self.options.model.parameters])
        # now "overlay" any rates we are estimating, by extracting them from
        # position and inverting the log transform
        values[self.estimate_idx] = 10 ** position
        return values

    def generate_new_position(self):
        """Sample from num_estimate independent gaussians and normalize the
        resulting vector to obtain a vector sampled uniformly on the unit
        hypersphere. Then scale by norm_step_size and ig_value, and add the
        resulting vector to our current position."""
        step = np.random.randn(self.num_estimate)
        if not self.options.use_hessian \
                or self.iter < self.options.hessian_period \
                or self.hessian is None:
            step /= math.sqrt(step.dot(step))
            step *= self.options.norm_step_size * self.sig_value
        else:
            # FIXME: make the 0.25 a user option
            # FIXME make 0.0855 (hess_ss) a user option
            eig_val, eig_vec = np.linalg.eig(self.hessian)
            # clamp eigenvalues to a lower bound of 0.25
            adj_eig_val = np.maximum(abs(eig_val), 0.25)
            step = (eig_vec / adj_eig_val ** 0.5).dot(step) * 0.085
        return self.position + step

    def calculate_prior(self, position=None):
        if position is None:
            position = self.position
        if self.options.prior_fn:
            return self.options.prior_fn(self, position)
        else:
            # default is a flat prior
            return 0

    def calculate_likelihood(self, position=None):
        if position is None:
            position = self.position
        return self.options.likelihood_fn(self, position)

    def calculate_posterior(self, position=None):
        prior = self.calculate_prior(position)
        likelihood = self.calculate_likelihood(position)
        posterior = prior + likelihood
        return posterior, prior, likelihood

    def calculate_hessian(self, position=None):
    # http://en.wikipedia.org/wiki/Finite_difference#Finite_difference_in_several_variables
    # TODO: look into these codes:
    # http://www.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation
    # http://www.mathworks.com/matlabcentral/fileexchange/11870-numerical-derivative-of-analytic-function
        if position is None:
            position = self.position
        d = 0.1
        hessian = np.empty((self.num_estimate, self.num_estimate))
        # iterate over diagonal
        for i in range(self.num_estimate):
            position_f = position.copy()
            position_b = position.copy()
            position_f[i] += d
            position_b[i] -= d
            f = self.calculate_posterior(position_f)[0]
            c = self.calculate_posterior(position)[0]
            b = self.calculate_posterior(position_b)[0]
            hessian[i,i] = (f - 2*c + b) / d ** 2
        # iterate over elements above diagonal
        for i in range(self.num_estimate-1):
            for j in range(i, self.num_estimate):
                position_ff = position.copy()
                position_fb = position.copy()
                position_bf = position.copy()
                position_bb = position.copy()
                position_ff[i] += d;
                position_ff[j] += d;
                position_fb[i] += d;
                position_fb[j] -= d;
                position_bf[i] -= d;
                position_bf[j] += d;
                position_bb[i] -= d;
                position_bb[j] -= d;
                ff = self.calculate_posterior(position_ff)[0]
                fb = self.calculate_posterior(position_fb)[0]
                bf = self.calculate_posterior(position_bf)[0]
                bb = self.calculate_posterior(position_bb)[0]
                hessian[i,j] = (ff - fb - bf + bb) / (4 * d ** 2)
                # hessian is symmetric, so we copy the value to the transposed location
                hessian[j,i] = hessian[i,j]
        if np.any(np.isnan(hessian)):
            raise Exception("NaN encountered in hessian calculation")
        return hessian


class MCMCOpts(object):

    def __init__(self):
        self.rhs_fn             = None    # model ODE right-hand side fn (t,y,params)
        self.estimate_params    = None    # boolean array indicating which parameters to estimate
        self.initial_values     = None    # starting values for parameters to estimate
        self.tspan              = None    # start/end times for model integration, or list of times
        self.step_fn            = None    # user callback, called on every MCMC iteration
        self.likelihood_fn      = None    # model likelihood fn (project,position)
        self.prior_fn           = None    # model prior fn (project,position)
        self.nsteps             = None    # number of MCMC iterations to perform
        self.use_hessian        = False   # whether to use the Hessian to guide the walk
        self.start_random       = False   #  whether to start from random rates and ICs
        self.boundary_option    = False # whether to enforce hard boundaries on the walk trajectory
        self.reltol             = None    # relative tolerance for ODE solver
        self.abstol             = None    # absolute tolerance for ODE solver (scalar or vector)
        self.norm_step_size     = 0.75  # MCMC step size
        self.hessian_period     = 25000 # number of MCMC steps between Hessian recalculations
        self.hessian_scale      = 0.085 # scaling factor used in generating Hessian-guided steps
        self.sigma_adj_interval = None    # how often to adust sig_value while annealing
        self.anneal_length      = None    # length of initial "burn-in" annealing period
        self.T_init             = 10    # initial temperature for annealing
        self.accept_rate_target = 0.3   # desired acceptance rate during annealing
        self.sigma_max          = 1     # max value for sigma (MCMC step size scaling factor)
        self.sigma_min          = 0.25  # min value for sigma
        self.sigma_step         = 0.125 # increment for sigma adjustments, to retain accept_rate_target
