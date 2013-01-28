import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
import matplotlib
import mpl_toolkits.mplot3d as mplot3d
import numpy
import math
import itertools
import multiprocessing
import pysb.integrate


__all__ = ['scatter', 'surf', 'sample']


def scatter(mcmc, mask=True):
    """
    Display a grid of scatter plots for each 2-D projection of an MCMC walk.

    Parameters
    ----------
    mcmc : bayessb.MCMC
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
    gs = mgridspec.GridSpec(ndims, ndims, width_ratios=lim_ranges,
                            height_ratios=lim_ranges[-1::-1])
    # build an axis locator for each dimension
    locators = []
    for i, r in enumerate(lim_ranges):
        # place ticks on the integers, unless there is no integer within the
        # given dimension's calculated range
        nbins = math.ceil(r) + 1
        want_integer = math.ceil(lims_bottom[i]) <= math.floor(lims_top[i])
        locators.append(mticker.MaxNLocator(nbins=nbins, integer=want_integer))

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
                ax.hist(positions[:,x], color='salmon', ec='tomato')
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

def surf(mcmc, dim0, dim1, mask=True, square_aspect=True, gridsize=20):
    """
    Display the posterior of an MCMC walk as a 3-D surface.

    Parameters
    ----------
    mcmc : bayessb.MCMC
        The MCMC object to display.
    dim0, dim1 : indices of parameters to display
    mask : bool/int, optional
        If True (default) the annealing phase of the walk will be discarded
        before plotting. If False, nothing will be discarded and all points will
        be plotted. If an integer, specifies the number of steps to be discarded
        from the beginning of the walk.

    """
    # mask off the annealing (burn-in) phase, or up to a user-specified step
    if mask is True:
        mask = mcmc.options.anneal_length
    elif mask is False:
        mask = 0
    # create masked versions of a few vectors of interest
    accepts = mcmc.accepts[mask:]
    rejects = mcmc.rejects[mask:]
    posteriors = mcmc.posteriors[mask:]
    # filter out the position elements we aren't plotting
    positions = mcmc.positions[mask:, (dim0, dim1)]
    # build grid of points for sampling the posterior surface
    pos_min = positions.min(0)
    pos_max = positions.max(0)
    if square_aspect:
        # enforce square aspect ratio in X-Y plane by recomputing pos_min/max
        pos_max_range = numpy.max(pos_max - pos_min)
        pos_mean = numpy.mean([pos_min, pos_max], 0)
        pos_min = pos_mean - pos_max_range / 2
        pos_max = pos_mean + pos_max_range / 2
    margin = (pos_max - pos_min) * 0.1
    pos_min -= margin
    pos_max += margin
    p0_vals = numpy.linspace(pos_min[0], pos_max[0], gridsize)
    p1_vals = numpy.linspace(pos_min[1], pos_max[1], gridsize)
    p0_mesh, p1_mesh = numpy.meshgrid(p0_vals, p1_vals)
    # calculate posterior value at all gridsize*gridsize points
    posterior_mesh = numpy.empty_like(p0_mesh)
    position_base = numpy.median(mcmc.positions, axis=0)
    # use multiprocessing to make use of multiple cores
    idx_iter = itertools.product(range(gridsize), range(gridsize))
    inputs = ((p0_mesh[i0, i1], p1_mesh[i0, i1]) for i0, i1 in idx_iter)
    inputs = itertools.product([mcmc], [position_base], [dim0], [dim1], inputs)
    try:
        pool = multiprocessing.Pool()
    except KeyboardInterrupt:
        pool.terminate()
        raise
    outputs = pool.map(surf_calc_mesh_pos, inputs)
    pool.close()
    for i0 in range(gridsize):
        for i1 in range(gridsize):
            posterior_mesh[i0, i1] = outputs[i0 * gridsize + i1]
    # plot 3-D surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(p0_mesh, p1_mesh, posterior_mesh, rstride=1, cstride=1,
                    cmap=matplotlib.cm.jet, linewidth=0, alpha=0.2)
    ax.plot(positions[accepts,0], positions[accepts,1], posteriors[accepts],
            c='k')
    ax.scatter(positions[rejects,0], positions[rejects,1], posteriors[rejects],
               marker='x', c='k', alpha=0.3)
    ax.set_xlabel('ln(%s)' % mcmc.options.estimate_params[dim0].name)
    ax.set_ylabel('ln(%s)' % mcmc.options.estimate_params[dim1].name)
    ax.set_zlabel('-ln(posterior)')
    plt.show()

def surf_calc_mesh_pos(args):
    mcmc, position_base, dim0, dim1, param_vals = args
    p0_val, p1_val = param_vals
    position = position_base.copy()
    position[dim0] = p0_val
    position[dim1] = p1_val
    return mcmc.calculate_posterior(position)[0]

def sample(mcmc, n, colors, norm_factor=None):
    """
    Display model trajectories of parameter sets sampled from the MCMC walk.

    Parameters
    ----------
    mcmc : bayessb.MCMC
        MCMC object to sample from.
    n : int
        Number of samples.
    norm_factor : vector-like, optional
        Vector of scaling factors, the same length as the number of species in
        the model. Each simulated trajectory will be divided by the respective
        value before plotting. If omitted, trajectories will be rescaled to the
        maximum value of each trajectory across all samples and all time points.

    Notes
    -----
    This is not currently useful for models with many species. It should
    probably plot observables instead (and allow selection of specific ones).

    """
    # FIXME plot observables instead, and allow a list of observables or
    # observable names as an argument.
    plt.figure()
    tspan = mcmc.options.tspan
    accept_positions = mcmc.positions[mcmc.accepts]
    if isinstance(n, int):
        idx = range(n)
    ysamples = numpy.empty((len(idx), len(tspan),
                            len(mcmc.options.model.species)))
    for i in idx:
        ysamples[i] = mcmc.simulate(accept_positions[-(i+1)])
    if norm_factor is None:
        norm_factor = ysamples.max(1).max(0)
    ysamples /= norm_factor
    for yout in ysamples:
        for y, c in zip(yout.T, colors):
            plt.plot(tspan, y, c=c, linewidth=1, alpha=.01)
    true_params = [p.value for p in mcmc.options.estimate_params]
    true_position = numpy.log10(true_params)
    true_yout = mcmc.simulate(true_position) / norm_factor
    plt.plot(tspan, true_yout, c='k', linewidth=2)
    plt.show()
