import mcmc_hessian
from pysb.examples.robertson import model
from pysb.integrate import odesolve
import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
import functools
import sys

def likelihood(mcmc, position, data, scale_factor, sigma):
    yout = mcmc.simulate(position)
    yout_norm = yout / scale_factor
    # fit to first two species
    return numpy.sum((data[:,0:2] - yout_norm[:,0:2]) ** 2 / (2 * sigma ** 2))

def prior(mcmc, position):
    est = [1e-2, 1e7, 1e4]
    mean = numpy.log10(est)
    var = 10
    return numpy.sum((position - mean) ** 2 / ( 2 * var))

def step(mcmc):
    if mcmc.iter % 20 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1), mcmc.accept_likelihood,
             mcmc.accept_prior, mcmc.accept_posterior)

def scatter(mcmc, mask=True):
    """
    Display a grid of scatter plots for each 2-D projection of an MCMC walk.

    Parameters
    ----------
    mcmc : mcmc_hessian.MCMC
        The MCMC object to display.
    mask : bool/int, optional
        If True (default) the annealing phase of the walk will be discarded
        before plotting. If False, nothing will be discarded and all points will
        be plotted. If an integer, specifies the number of steps to be discarded
        from the beginning of the walk.

    """

    # number of dimensions in position vector
    ndims = mcmc.num_estimate
    # vector of booleans indicating accepted MCMC moves
    accepts = mcmc.accepts.copy()
    # mask off the annealing (burn-in) phase, or up to a user-specified step
    if mask is True:
        mask = mcmc.options.anneal_length
    if mask is False:
        mask = 0
    accepts[0:mask] = 0
    # grab position vectors and posterior values from accepted moves
    positions = mcmc.positions[accepts]
    posteriors = mcmc.posteriors[accepts]
    # calculate actual range of values on each dimension
    maxes = positions.max(0)
    mins = positions.min(0)
    ranges = abs(maxes - mins)
    # use 2% of the maximum range as a margin for all scatter plots
    margin = max(ranges) * 0.02
    # calculate upper and lower plot limits based on min/max plus the margin
    lims_top = maxes + margin
    lims_bottom = mins - margin
    # calculate new ranges based on limits
    lim_ranges = abs(lims_top - lims_bottom)

    plt.figure()
    # build a GridSpec which allocates space based on these ranges
    import matplotlib.gridspec as mgridspec
    gs = mgridspec.GridSpec(ndims, ndims, width_ratios=lim_ranges,
                            height_ratios=lim_ranges[-1::-1])
    # build an axis locator for each dimension
    locators = []
    for i, r in enumerate(lim_ranges):
        # place ticks on the integers, unless there is no integer within the
        # given dimension's calculated range
        nbins = numpy.ceil(r) * 5 + 1
        locators.append(mticker.MaxNLocator(nbins=nbins, steps=[2, 10]))

    fignum = 0
    # reverse the param list along the y axis so we end up with the "origin"
    # (i.e. the first param) at the bottom left instead of the top left. note
    # that y==0 will be along the bottom now, but the figure numbers in the
    # gridspec still begin counting at the top.
    for y, py in reversed(list(enumerate(mcmc.options.estimate_params))):
        for x, px in enumerate(mcmc.options.estimate_params):
            ax = plt.subplot(gs[fignum])
            ax.tick_params(left=False, right=True, top=True, bottom=False,
                           labelleft=False, labelright=False, labeltop=False,
                           labelbottom=False, direction='in')
            ax.yaxis.set_major_locator(locators[y])
            ax.xaxis.set_major_locator(locators[x])
            if x == y:
                # 1-D histograms along the diagonal
                ax.hist(positions[:,x], bins=100, histtype='stepfilled',
                        color='salmon', ec='tomato')
                ax.set_xlim(lims_bottom[x], lims_top[x])
                ax.yaxis.set_major_locator(mticker.NullLocator())
            else:
                # 2-D scatter plots off the diagonal
                ax.plot(positions[:, x], positions[:, y], color='darkblue',
                        alpha=0.2)
                ax.scatter(positions[:, x], positions[:, y], s=1, color='darkblue',
                           alpha=0.2)
                ax.set_xlim(lims_bottom[x], lims_top[x])
                ax.set_ylim(lims_bottom[y], lims_top[y])
            # parameter name labels along left and bottom edge of the grid
            if x == 0:
                ax.set_ylabel(py.name, weight='black', size='large',
                              labelpad=10, rotation='horizontal',
                              horizontalalignment='right')
            if y == 0:
                ax.set_xlabel(px.name, weight='black', size='large',
                              labelpad=10,)
            # tick labels along the right and top edge of the grid
            if x == ndims - 1:
                ax.tick_params('y', labelright=True)
            if y == ndims - 1:
                ax.tick_params('x', labeltop=True)
            # move to next figure in the gridspec
            fignum += 1
    # TODO: would axis('scaled') force the aspect ratio we want?
    plt.show()


def prediction(mcmc, n, species_idx, scale_factor, data, plot_samples=False):
    plt.figure()
    positions = mcmc.positions[-n:]
    accepts = mcmc.accepts[-n:]
    accept_positions = positions[accepts]
    tspan = mcmc.options.tspan
    ysamples = numpy.empty((len(accept_positions), len(tspan)))
    for i, pos in enumerate(accept_positions):
        ysim = mcmc.simulate(pos)
        ysamples[i] = ysim[:, species_idx] / scale_factor
    ymean = numpy.mean(ysamples, 0)
    ystd = numpy.std(ysamples, 0)
    if plot_samples:
        for y in ysamples:
            plt.plot(tspan, y, c='gray', alpha=.01)
    plt.plot(tspan, ymean, 'b:', linewidth=2)
    std_interval = ystd[:, None] * [+1, -1]
    plt.plot(tspan, ymean[:, None] + std_interval * 0.842, 'g-.', linewidth=2)
    plt.plot(tspan, ymean[:, None] + std_interval * 1.645, 'k-.', linewidth=2)
    plt.show()


def main():
    global mcmc

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

    opts = mcmc_hessian.MCMCOpts()
    opts.model = model
    opts.tspan = tspan

    # estimate rates only (not initial conditions) from wild guesses
    opts.estimate_params = [p for p in model.parameters if p.name.startswith('k') ]
    opts.initial_values = [1e-4, 1e3, 1e6]

    opts.nsteps = 10000
    opts.likelihood_fn = functools.partial(likelihood, data=ydata,
                                           scale_factor=ysim_max, sigma=sigma)
    opts.prior_fn = prior
    opts.step_fn = step
    opts.use_hessian = True
    opts.hessian_period = opts.nsteps / 10
    opts.seed = seed
    mcmc = mcmc_hessian.MCMC(opts)

    mcmc.run()

    estimate = numpy.median(mcmc.positions[mcmc.accepts], 0)

    # show scatter plot
    scatter(mcmc, opts.nsteps / 2)
    # show prediction for C trajectory, which was not fit to
    prediction(mcmc, opts.nsteps / 2, 2, ysim_max[2])

if __name__ == '__main__':
    main()
