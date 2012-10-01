import biomc
from pysb.examples.robertson import model
import pysb.integrate
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

def scatter(mcmc, mask=True, example_pos_r=None, example_pos_g=None,
            show_model=False):
    """
    Display a grid of scatter plots for each 2-D projection of an MCMC walk.

    Parameters
    ----------
    mcmc : biomc.MCMC
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
                #
                # distribute 200 total bins across all histograms,
                # proportionally by their width, such that the bin density looks
                # consistent across the different histograms
                bins = 200 * lim_ranges[x] / numpy.sum(lim_ranges)
                ax.hist(positions[:,x], bins=bins, histtype='stepfilled',
                        color='salmon', ec='tomato')
                if example_pos_r is not None:
                    ax.vlines(example_pos_r[x], *ax.get_ylim(),
                              color='red', linewidth=2)
                if example_pos_g is not None:
                    ax.vlines(example_pos_g[x], *ax.get_ylim(),
                              color='green', linewidth=2)
                arrow_scale = ax.get_ylim()[1] / lim_ranges[x]
                arrow_len = arrow_scale * 0.1
                arrow_head_l = arrow_len * 0.4
                arrow_head_w = min(lim_ranges) * .1
                ax.arrow(numpy.log10(px.value), arrow_len, 0, -arrow_len,
                         head_length=arrow_head_l, head_width=arrow_head_w,
                         ec='k', fc='k', length_includes_head=True)
                ax.set_xlim(lims_bottom[x], lims_top[x])
                #ax.yaxis.set_major_locator(mticker.NullLocator())
                ax.yaxis.set_major_locator(mticker.LinearLocator())
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
            if True:#x == ndims - 1: # XXX
                ax.tick_params('y', labelright=True)
            if y == ndims - 1:
                ax.tick_params('x', labeltop=True)
            # move to next figure in the gridspec
            fignum += 1
    # TODO: would axis('scaled') force the aspect ratio we want?


def prediction(mcmc, n, species_idx, scale_factor, data_std, plot_samples=False):
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
    plt.errorbar(tspan, ymean, yerr=data_std, fmt=None, ecolor='red')
    plt.xlim(tspan[0] - 1, tspan[-1] + 1)


def data(mcmc, data_norm, scale_factor, data_species_idxs):
    plt.figure()
    colors = ('r', 'g', 'b')
    labels = ('A', 'B', 'C')
    tspan = mcmc.options.tspan
    true_pos = numpy.log10([p.value for p in mcmc.options.estimate_params])
    true_norm = mcmc.simulate(true_pos) / scale_factor
    for i, (rl, dl, c, l) in enumerate(zip(true_norm.T, data_norm.T,
                                           colors, labels)):
        plt.plot(tspan, rl, color=c, label=l)
        if i in data_species_idxs:
            plt.plot(tspan, dl, linestyle=':', marker='o', color=c, ms=4, mew=0)


def main():
    seed = 2
    random = numpy.random.RandomState(seed)
    sigma = 0.1;
    ntimes = 20;
    tspan = numpy.linspace(0, 40, ntimes);
    solver = pysb.integrate.Solver(model, tspan)
    solver.run()
    ydata = solver.y * (random.randn(*solver.y.shape) * sigma + 1);
    ysim_max = solver.y.max(0)
    ydata_norm = ydata / ysim_max

    opts = biomc.MCMCOpts()
    opts.model = model
    opts.tspan = tspan

    # estimate rates only (not initial conditions) from wild guesses
    opts.estimate_params = [p for p in model.parameters if p.name.startswith('k') ]
    opts.initial_values = [1e-4, 1e3, 1e6]

    opts.nsteps = 10000
    opts.likelihood_fn = functools.partial(likelihood, data=ydata_norm,
                                           scale_factor=ysim_max, sigma=sigma)
    opts.prior_fn = prior
    opts.step_fn = step
    opts.use_hessian = True
    opts.hessian_period = opts.nsteps / 10
    opts.seed = seed
    mcmc = biomc.MCMC(opts)

    mcmc.run()

    mixed_nsteps = opts.nsteps / 2
    mixed_positions = mcmc.positions[-mixed_nsteps:]
    mixed_accepts = mcmc.accepts[-mixed_nsteps:]
    mixed_accept_positions = mixed_positions[mixed_accepts]
    marginal_mean_pos = numpy.mean(mixed_accept_positions, 0)

    # position is far from marginal mean, but posterior is good (determined by
    # trial and error and some interactive plotting)
    interesting_step = 8830

    print "\nGenerating figures..."
    # show scatter plot
    scatter(mcmc, opts.nsteps / 2, mcmc.positions[interesting_step],
            marginal_mean_pos)
    # show prediction for C trajectory, which was not fit to
    prediction(mcmc, opts.nsteps / 2, 2, ysim_max[2], sigma, plot_samples=True)
    plt.title("Prediction for C")
    # show "true" trajectories and noisy data
    data(mcmc, ydata_norm, ysim_max, [0, 1])
    plt.title("True trajectories and noisy data")
    # show all plots at once
    plt.show()

if __name__ == '__main__':
    main()
