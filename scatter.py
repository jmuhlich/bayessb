import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
import math

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
    plt.show()
