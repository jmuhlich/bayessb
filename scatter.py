import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def scatter(mcmc):

    n_est = mcmc.num_estimate
    accepts = mcmc.accepts.copy()
    accepts[0:2650] = 0
    positions = mcmc.positions[accepts]
    posteriors = mcmc.posteriors[accepts]
    maxes = positions.max(0)
    mins = positions.min(0)
    ranges = abs(maxes - mins)
    margin = max(ranges) * 0.1
    lims_top = maxes + margin
    lims_bottom = mins - margin
    lim_ranges = abs(lims_top - lims_bottom)
    gs = gridspec.GridSpec(n_est, n_est, width_ratios = lim_ranges,
                           height_ratios = lim_ranges)
    fignum = 0
    for y, py in enumerate(mcmc.options.estimate_params):
        for x, px in enumerate(mcmc.options.estimate_params):
            ax = plt.subplot(gs[fignum])
            if x != y:
                ax.plot(positions[:, x], positions[:, y], color='darkblue', alpha=0.2)
                ax.scatter(positions[:, x], positions[:, y], s=1, color='darkblue', alpha=0.2)
                ax.set_xlim(lims_bottom[x], lims_top[x])
                ax.set_ylim(lims_bottom[y], lims_top[y])
            ax.tick_params(left=False, right=True, top=False, bottom=True,
                           labelleft=False, labelright=False, labeltop=False,
                           labelbottom=False, direction='in')
            if x == y:
                ax.hist(positions[:,x], color='salmon')
                ax.set_xlim(lims_bottom[x], lims_top[x])
            if x == 0:
                ax.set_ylabel(py.name, rotation='horizontal',
                              horizontalalignment='right', weight='black',
                              size='large')
            if y == 0:
                ax.set_xlabel(px.name, weight='black', size='large')
                ax.xaxis.set_label_position('top')
            if x == n_est - 1:
                ax.tick_params('y', labelright=True)
            if y == n_est - 1:
                ax.tick_params('x', labelbottom=True)
            fignum += 1
    plt.show()
