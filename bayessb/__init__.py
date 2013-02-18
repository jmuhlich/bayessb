import numpy as np
import math

_use_pysb = False
try:
    import pysb.core
    import pysb.integrate
    _use_pysb = True
except ImportError:
    pass


__all__ = ['MCMC', 'MCMCOpts']


class MCMC(object):

    """An interface for Markov-Chain Monte Carlo parameter estimation.

    Parameters
    ----------
    options : bayessb.MCMCOpts
        Option set -- defines the problem and sets some parameters to control
        the MCMC algorithm.

    Attributes
    ----------
    options : bayessb.MCMCOpts
        Validated copy of options passed to constructor.
    num_estimate : int
        Number of parameters to estimate.
    estimate_idx : list of int
        Indices of parameters to estimate in the model's full parameter list.
    initial_values : list of float
        Starting values for parameters to estimate, taken from the parameters'
        nominal values in the model or explicitly specified in `options`.
    initial_position : list of float
        Starting position of MCMC walk in parameter space (log10 of
        `initial_values`).
    position : list of float
        Current position of MCMC walk in parameter space, i.e. the most recently
        accepted move).
    test_position : list of float
        Proposed MCMC move.
    acceptance : int
        Number of accepted moves.
    T : float
        Current value of the simulated annealing temperature.
    T_decay : float
        Constant for exponential decay of `T`, automatically calculated such
        that T will decay from `options.T_init` down to 1 over the first
        `options.anneal_length` steps.
    sig_value : float
        Current value of 'Sigma', the scaling factor for the proposal
        distribution. The MCMC algorithm dynamically tunes this to maintain the
        acceptance rate specified in `options.accept_rate_target`.
    iter : int
        Current MCMC step number.
    start_iter : int
        Starting MCMC step number.
    ode_options : dict
        Options for the ODE integrator, currently just 'rtol' for relative
        tolerance and 'atol' for absolute tolerance.
    random : numpy.random.RandomState
        Random number generator. Seeded with `options.seed` for reproducible
        runs.
    solver : pysb.integrate.Solver
        ODE solver.
    initial_prior : float
        Starting prior value, i.e. the value at `initial_position`.
    initial_likelihood : float
        Starting likelihood value, i.e. the value at `initial_position`.
    initial_posterior : float
        Starting posterior value, i.e. the value at `initial_position`.
    accept_prior : float
        Current prior value, i.e the value at `position`.
    accept_likelihood : float
        Current likelihood value, i.e the value at `position`.
    accept_posterior : float
        Current posterior value, i.e the value at `position`.
    test_prior : float
        Prior value at `test_position`.
    test_likelihood : float
        Likelihood value at `test_position`.
    test_posterior : float
        Posterior value at `test_position`.
    hessian : numpy.ndarray of float
        Current hessian of the posterior landscape. Size is
        `num_estimate`x`num_estimate`.
    positions : numpy.ndarray of float
        Trace of all proposed moves. Size is `num_estimate`x`nsteps`.
    priors, likelihoods, posteriors : numpy.ndarray of float
        Trace of all priors, likelihoods, and posteriors corresponding to
        `positions`. Length is `nsteps`.
    alphas, sigmas, delta_posteriors, ts : numpy.ndarray of float
        Trace of various MCMC parameters and calculated values. Length is
        `nsteps`.
    accepts, rejects : numpy.ndarray of bool
        Trace of whether each propsed move was accepted or rejected. Length is
        `nsteps`.
    hessians : numpy.ndarray of float
        Trace of all hessians. Size is `num_estimate`x`num_estimate`x`num_hessians`
        where num_hessians is the actual number of hessians to be calculated.

    Notes
    -----

    """

    def __init__(self, options):
        self.options = self.validate(options)
    
    def __getstate__(self):
        # clear solver since it causes problems with pickling
        state = self.__dict__.copy()
        del state['solver']
        return state

    def __setstate__(self, state):
        # re-init the solver which we didn't pickle
        self.__dict__.update(state)
        self.init_solver()

    def run(self):
        """Initialize internal state and runs the parameter estimation."""
        self.initialize()
        self.estimate()
        
    def validate(self, options):
        """Return a validated copy of options with defaults applied."""
        # FIXME should this live in MCMCOpts?

        options = options.copy()

        if options.model is None:
            raise Exception("model not defined")

        if options.estimate_params is None or not len(options.estimate_params):
            raise Exception("estimate_params must contain a list of parameters")
            
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
        """Initialize internal state from the option set."""

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
            
        # need to do this before init_solver
        self.ode_options = {};
        if self.options.rtol is not None:
            self.ode_options['rtol'] = self.options.rtol
        if self.options.atol is not None:
            self.ode_options['atol'] = self.options.atol

        # create solver so we can calculate the posterior
        self.init_solver()

        self.initial_posterior, self.initial_prior, self.initial_likelihood = \
            self.calculate_posterior(self.initial_position)

        self.accept_prior = self.initial_prior
        self.accept_likelihood = self.initial_likelihood
        self.accept_posterior = self.initial_posterior

        self.T_decay = -math.log10(1e-6) / self.options.anneal_length;
            
        self.random = np.random.RandomState(self.options.seed)

        self.start_iter = 0;
        self.acceptance = 0;
        self.T = self.options.T_init;
        self.sig_value = 1.0
        self.hessian = None

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
        hessian_steps = self.options.nsteps - self.options.anneal_length
        num_hessians = int(math.ceil(float(hessian_steps)
                                     / self.options.hessian_period))
        # initialize to zeros so we can see where there were failures
        self.hessians = np.zeros((num_hessians,
                                  self.num_estimate, self.num_estimate))

    def init_solver(self):
        """Initialize solver from model and tspan."""
        if _use_pysb and isinstance(self.options.model, pysb.core.Model):
            self.solver = pysb.integrate.Solver(self.options.model,
                                                self.options.tspan,
                                                **self.ode_options)

    def estimate(self):
        """Execute the MCMC parameter estimation algorithm."""

        self.iter = self.start_iter;
        while self.iter < self.options.nsteps:

            # update hessian
            if (self.options.use_hessian
                and self.iter >= self.options.anneal_length
                and self.iter % self.options.hessian_period == 0):
                try:
                    self.hessian = self.calculate_hessian()
                    #self.hessian = self.calculate_inverse_covariance()
                    hessian_num = ((self.iter - self.options.anneal_length)
                                   // self.options.hessian_period)
                    self.hessians[hessian_num,:,:] = self.hessian;

                except HessianCalculationError:
                    pass

            # choose test position and calculate posterior there
            self.test_position = self.generate_new_position()
            (self.test_posterior, self.test_prior, self.test_likelihood) = \
                self.calculate_posterior(self.test_position)

            # ------------------METROPOLIS-HASTINGS ALGORITHM-------------------
            delta_posterior = self.test_posterior - self.accept_posterior
            if delta_posterior < 0:
                self.accept_move()
            else:
                alpha = self.random.rand()
                self.alphas[self.iter] = alpha;  # log the alpha value
                if math.e ** (-delta_posterior/self.T) > alpha:
                    self.accept_move()
                else:
                    self.reject_move()

            # -------ADJUSTING SIGMA & TEMPERATURE (ANNEALING)--------
            # XXX why did I move this first bit outside the iter<anneal_length test (below)?
            if self.iter % self.options.sigma_adj_interval == 0:

                # Calculate the acceptance rate only over the recent steps
                # unless we haven't done enough steps yet
                window = self.options.accept_window
                if self.iter < window:
                    accept_rate = float(self.acceptance) / (self.iter + 1)
                else:
                    accept_rate = np.sum(self.accepts[(self.iter - window): 
                                            self.iter]) / float(window)

                if accept_rate < self.options.accept_rate_target:
                    if self.sig_value > self.options.sigma_min:
                        #self.sig_value -= self.options.sigma_step
                        self.sig_value *= self.options.sigma_step
                else:
                    if self.sig_value < self.options.sigma_max:
                        #self.sig_value += self.options.sigma_step
                        self.sig_value *= (1. / self.options.sigma_step)

            if self.iter < self.options.anneal_length:
                self.T = 1 + (self.options.T_init - 1) * math.e ** (-self.iter * self.T_decay)
                
            # log some interesting variables
            self.positions[self.iter,:] = self.test_position
            self.priors[self.iter] = self.test_prior
            self.likelihoods[self.iter] = self.test_likelihood
            self.posteriors[self.iter] = self.test_posterior
            self.delta_posteriors[self.iter] = delta_posterior
            self.sigmas[self.iter] = self.sig_value
            self.ts[self.iter] = self.T
                
            # call user-callback step function
            if self.options.step_fn:
                self.options.step_fn(self)
            
            self.iter += 1
        
    def accept_move(self):
        """Accept the current proposed move."""
        self.accept_prior = self.test_prior
        self.accept_likelihood = self.test_likelihood
        self.accept_posterior = self.test_posterior
        self.position = self.test_position
        self.acceptance += 1
        self.accepts[self.iter] = 1

    def reject_move(self):
        """Reject the current proposed move."""
        self.rejects[self.iter] = 1;

    def simulate(self, position=None, observables=False):
        """Simulate the model.

        Parameters
        ----------
        position : list of float, optional
            log10 of the values of the parameters being estimated. (See
            the `cur_params` method for details)
        observables : boolean, optional
            If true, return a record array containing the trajectories of the
            model's observables. If false, return a float array of all species
            trajectories. Defaults to false.

        """
        if position is None:
            position = self.position
        self.solver.run(self.cur_params(position))
        if observables:
            return self.solver.yobs
        else:
            return self.solver.y

    def cur_params(self, position=None):
        """Return a list of the values of all model parameters.

        For a given set of values for the parameters to be estimated, this
        method returns an array containing the actual (not log-transformed)
        values of all model parameters, not just those to be estimated, in the
        same order as specified in the model. This is helpful when simulating
        the model at a given position in parameter space.

        Parameters
        ----------
        position : list of float, optional
            log10 of the values of the parameters being estimated. If omitted,
            `self.position` (the most recent accepted MCMC move) will be
            used. The model's nominal values will be used for all parameters
            *not* being estimated, regardless.

        """
        if position is None:
            position = self.position
        # start with the original values
        values = np.array([p.value for p in self.options.model.parameters])
        # now "overlay" any rates we are estimating, by extracting them from
        # position and inverting the log transform
        values[self.estimate_idx] = np.power(10, position)
        return values

    def generate_new_position(self):
        """Generate a sample from the proposal distribution."""
        # sample from num_estimate independent gaussians
        step = self.random.randn(self.num_estimate)
        if not self.options.use_hessian \
                or self.iter < self.options.hessian_period \
                or self.hessian is None:
            # normalize to obtain a vector sampled uniformly on the unit
            # hypersphere
            step /= math.sqrt(step.dot(step))
            # scale by norm_step_size and sig_value.
            step *= self.options.norm_step_size * self.sig_value
        else:
            # FIXME make the 0.25 a user option
            # FIXME call eig() only when hessian changes and store results
            eig_val, eig_vec = np.linalg.eig(self.hessian)
            # clamp eigenvalues to a lower bound of 0.25
            adj_eig_val = np.maximum(abs(eig_val), 0.25)
            # transform into eigenspace, with length scaled by the inverse
            # square root of the eigenvalues. length is furthermore scaled down
            # by a constant factor.
            step = (eig_vec / adj_eig_val ** 0.5).dot(step) \
                * self.options.hessian_scale * self.sig_value
        # the proposed position is our most recent accepted position plus the
        # step we just calculated
        return self.position + step

    def calculate_prior(self, position=None):
        """Return the prior for a position in parameter space.

        If `options.prior_fn` is None, return 0 (i.e. a flat prior).

        position : list of float, optional
            log10 of the values of the parameters being estimated. (See
            the `cur_params` method for details)

        """
        if position is None:
            position = self.position
        if self.options.prior_fn:
            return self.options.prior_fn(self, position)
        else:
            # default is a flat prior
            return 0

    def calculate_likelihood(self, position=None):
        """Return the likelihood for a position in parameter space.

        position : list of float, optional
            log10 of the values of the parameters being estimated. (See
            the `cur_params` method for details)

        """
        if position is None:
            position = self.position
        return self.options.likelihood_fn(self, position)

    def calculate_posterior(self, position=None):
        """Return the prior, likelihood and posterior for a position.

        Calculates the prior and likelihood and adds them together (with the
        likelihood scaled by the thermodynamic integration temperature, if being
        used) to generate the posterior. Returns a tuple of the posterior, prior
        and likelihood as all three are always tracked together by the MCMC
        algorithm.

        position : list of float, optional
            log10 of the values of the parameters being estimated. (See
            the `cur_params` method for details)

        """
        prior = self.calculate_prior(position)
        likelihood = self.calculate_likelihood(position)
        posterior = prior + likelihood * self.options.thermo_temp
        return posterior, prior, likelihood

    def calculate_inverse_covariance(self):
        covariance_matrix = np.cov(self.positions, rowvar=0)
        return np.linalg.inv(covariance_matrix)

    def calculate_hessian(self, position=None):
        """Calculate the hessian of the posterior landscape.

        Note that this is very expensive -- O(n^2) where n is the number of
        parameters being estimated. The calculation is performed using second
        order central finite differences.

        position : list of float, optional
            log10 of the values of the parameters being estimated. (See
            the `cur_params` method for details)

        """
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
            for j in range(i + 1, self.num_estimate):
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
                # hessian is symmetric, so mirror the value across the diagonal
                hessian[j,i] = hessian[i,j]
        if np.any(np.isnan(hessian) | np.isinf(hessian)):
            raise HessianCalculationError(
                "numerical instability encountered in hessian calculation")
        return hessian

    def prune(self, burn, thin=1):
        """Truncates the chain to the thinned, mixed, accepted positions.

        Side Effects:

            - After this method is called, self.positions is (destructively) set
              to the mixed, accepted positions.
            - The ``self.pruned`` field is set to True, indicating that the walk
              has been irreversibly pruned.
            - The indices of the thinned, mixed, accepted steps are recorded in
              the field ``self.thinned_accept_steps``. This can be useful for
              plotting walks from multiple chains.
            - The values of the ``burn`` and ``thin`` parameters are recorded
              in the options object as ``self.options.burn`` and
              ``self.options.thin``.
        """

        (thinned_accepts, thinned_accept_steps) = \
                            self.get_mixed_accepts(burn, thin)
        self.positions = thinned_accepts
        self.options.burn = burn
        self.options.thin = thin
        self.pruned = True
        self.thinned_accept_steps = thinned_accept_steps

    def get_mixed_accepts(self, burn, thin=1):
        """A helper function that returns the thinned,
        accepted positions after a user-specified burn-in period; also returns
        the indices (step numbers) of each of the returned positions.

        Use this method instead of ``self.prune`` if you want to get the
        mixed accepted steps without discarding all of the other ones.

        Parameters
        ----------
        burn : int
            An integer specifying the number of steps to cut off from the
            beginning of the walk.
        thin : int
            An integer specifying how to thin the accepted steps of the walk.
            If 1, returns every step; if 2, every other step; if 5, every
            fifth step, etc.

        Returns
        -------
        A tuple of numpy.array objects. The first element in the tuple contains
        the array of accepted positions, burned and thinned as required; the
        second element contains a list of integers which indicate the indices
        associated with each of the steps returned.
        """

        mixed_steps = np.array(range(burn, self.options.nsteps))    
        mixed_positions = self.positions[burn:]

        mixed_accepts = mixed_positions[self.accepts[burn:]]
        mixed_accept_steps = mixed_steps[self.accepts[burn:]]

        thinned_accepts = mixed_accepts[::thin]
        thinned_accept_steps = mixed_accept_steps[::thin]

        return (thinned_accepts, thinned_accept_steps)

class HessianCalculationError(RuntimeError):
    pass


class MCMCOpts(object):

    """Options for defining a bayessb.MCMC project/run.

    Constructor takes no options. Interface is via direct manipulation of
    attributes on instances.

    Attributes
    ----------
    model : pysb.Model (or similar)
        The model to estimate. If you do not wish to use a PySB model, you may
        instead provide any object with a `parameters` attribute holding a list
        of all model parameters. The parameter objects in turn must each have a
        `value` attribute containing the parameter's numerical value. If you are
        not using a PySB model you must rely on your own code to simulate the
        model in your likelihood function instead of calling `MCMC.simulate`.
    estimate_params : list of pysb.Parameter
        List of parameters to estimate, all of which must also be listed in
        `model.parameters`.
    initial_values : list of float, optional
        Starting values for parameters to estimate. If omitted, will use the
        nominal values from `model.parameters`.
    tspan : list of float
        List of time points over which to integrate the model. Ignored if not
        using a PySB model.
    step_fn : callable f(mcmc), optional
        User callback, called on every MCMC iteration.
    likelihood_fn : callable f(mcmc, position)
        User likelihood function.
    prior_fn : callable f(mcmc, position), optional
        User prior function. If omitted, a flat prior will be used.
    nsteps : int
        Number of MCMC iterations to perform.
    use_hessian : bool, optional
        Whether to use the Hessian to guide the walk. Defaults to false.
    start_random : bool, optional
        Whether to start from a random point in parameter space. Defaults to
        false. (NOT IMPLEMENTED)
    boundary_option : bool, optional
        Whether to enforce hard boundaries on the walk trajectory. Defaults to
        false. (NOT IMPLEMENTED)
    rtol : float or list of float, optional
        Relative tolerance for ODE solver.
    atol : float or list of float, optional
        Absolute tolerance for ODE solver.
    norm_step_size : float, optional
        MCMC step size. Defaults to a reasonable value.
    hessian_period : int, optional
        Number of MCMC steps between Hessian recalculations. Defaults to a
        reasonable but fairly large value, as hessian calculation is expensive.
    hessian_scale : float, optional
        Scaling factor used in generating Hessian-guided steps. Defaults to a
        reasonable value.
    sigma_adj_interval : int, optional
        How often to adust `MCMC.sig_value` while annealing to meet
        `accept_rate_target`. Defaults to a reasonable value.
    anneal_length : int, optional
        Length of initial "burn-in" annealing period. Defaults to 10% of
        `nsteps`, or if `use_hessian` is true, to `hessian_period` (i.e. anneal
        until first hessian is calculated).
    T_init : float, optional
        Initial temperature for annealing. Defaults to a reasonable value.
    accept_rate_target : float, optional
        Desired acceptance rate during annealing. Defaults to a reasonable
        value. See also `sigma_adj_interval` above.
    sigma_max : float, optional
        Maximum value for `MCMC.sig_value`. Defaults to a reasonable value.
    sigma_min : float, optional
        Minimum value for `MCMC.sig_value`. Defaults to a reasonable value.
    sigma_step : float, optional
        Increment for `MCMC.sig_value` adjustments. Defaults to a reasonable
        value. To eliminate adaptive step size, set sigma_step to 1.
    thermo_temp : float in the range [0,1], optional
        Temperature for thermodynamic integration support. Used to scale
        likelihood when calculating the posterior value. Defaults to 1.0,
        i.e. no effect.
    seed : int or list of int, optional
        Seed for random number generator. Defaults to using a non-deterministic
        seed (see numpy.random.RandomState). If you want reproducible runs, you
        must set this to a constant value.
    accept_window : int
        The number of steps over which to calculate the current "local"
        accept rate. If the local acceptance rate is too low or too high,
        the step size is adjusted.

    """

    def __init__(self):
        self.model              = None    
        self.estimate_params    = None
        self.initial_values     = None
        self.tspan              = None
        self.step_fn            = None
        self.likelihood_fn      = None
        self.prior_fn           = None
        self.nsteps             = None
        self.use_hessian        = False
        self.start_random       = False
        self.boundary_option    = False
        self.rtol               = None
        self.atol               = None
        self.norm_step_size     = 0.75
        self.hessian_period     = 25000
        self.hessian_scale      = 0.085
        self.sigma_adj_interval = None
        self.anneal_length      = None
        self.T_init             = 10
        self.accept_rate_target = 0.3
        self.sigma_max          = 1
        self.sigma_min          = 0.25
        self.sigma_step         = 0.125
        self.thermo_temp        = 1
        self.seed               = None
        self.accept_window      = 200

    def copy(self):
        new_options = MCMCOpts()
        new_options.__dict__.update(self.__dict__)
        return new_options
