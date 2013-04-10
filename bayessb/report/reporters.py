import numpy as np
from matplotlib import pyplot as plt
from bayessb.report import reporter, Result, FloatListResult, ThumbnailResult
from bayessb import convergence
from StringIO import StringIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from bayessb.multichain import NoPositionsException

reporter_group_name = "Estimation"
num_samples = 100

@reporter('Number of chains')
def num_chains(mcmc_set):
    return Result(len(mcmc_set.chains), None)

@reporter('Estimation parameters')
def estimation_parameters(mcmc_set):
    chain = mcmc_set.chains[0]
    opts = chain.options
    output = StringIO()
    output.write("<html><head /><body><pre>")
    output.write("nbd_sites: %s\n" % chain.nbd_sites)
    output.write("nbd_observables: %s\n" % chain.nbd_observables)
    output.write("model: %s\n" % chain.builder.model.name)
    output.write("use_hessian: %s\n" % opts.use_hessian)
    output.write("hessian_period: %s\n" % opts.hessian_period)
    output.write("hessian_scale: %s\n" % opts.hessian_scale)
    output.write("norm_step_size: %s\n" % opts.norm_step_size)
    output.write("anneal_length: %s\n" % opts.anneal_length)
    output.write("T_init: %s\n" % opts.T_init)
    output.write("thermo_temp: %s\n" % opts.thermo_temp)
    output.write("accept_rate_target: %s\n" % opts.accept_rate_target)
    output.write("sigma_max: %s\n" % opts.sigma_max)
    output.write("sigma_min: %s\n" % opts.sigma_min)
    output.write("sigma_step: %s\n" % opts.sigma_step)
    output.write("sigma_adj_interval: %s\n" % opts.sigma_adj_interval)
    output.write("initial_values: %s\n" % opts.initial_values)
    output.write("</pre></body></html>")

    # Write the estimation parameter description to a file
    param_file_name = '%s_estimation_params.html' % mcmc_set.name
    with open(param_file_name, 'w') as f:
        f.write(output.getvalue())

    return Result(None, param_file_name)

@reporter('Conv. Criterion')
def convergence_criterion(mcmc_set):
    """Returns the vector of Gelman-Rubin convergence criterion values and a
    link to an HTML file containing plots of the traces of the walk for each
    parameter fitted."""

    # Prepare html for page showing plots of parameter traces
    html_str = "<html><head><title>Parameter traces for %s</title></head>\n" \
               % mcmc_set.name
    html_str += "<body><p>Parameter traces for %s</p>\n" \
                % mcmc_set.name
    img_str_list = []

    # Useful variables
    num_estimate = mcmc_set.chains[0].num_estimate
    fontP = FontProperties()
    fontP.set_size('small')
    legend_kwargs = {'prop':fontP, 'ncol':1, 'bbox_to_anchor':(1, 1),
                     'fancybox': True, 'shadow': True}

    # Make plot of posterior
    fig = Figure()
    ax = fig.gca()
    for i, chain in enumerate(mcmc_set.chains):
        color = cm.jet(i/float(num_estimate - 1))
        label = 's%d' % chain.options.seed
        if chain.pruned:
            line = ax.plot(chain.thinned_accept_steps, chain.posteriors, color=color,
                    label=label)
        else:
            line = ax.plot(chain.posteriors, color=color, label=label)
    ax.set_title("Posterior traces")
    ax.legend(loc='upper left', **legend_kwargs)
    posterior_plot_filename = '%s_posterior_trace.png' % mcmc_set.name
    canvas = FigureCanvasAgg(fig)
    fig.set_canvas(canvas)
    fig.savefig(posterior_plot_filename)

    # Make plots of parameter traces
    for i in range(num_estimate):
        param_name = mcmc_set.chains[0].options.estimate_params[i].name
        fig = Figure()
        ax = fig.gca()
        for j, chain in enumerate(mcmc_set.chains):
            color = cm.jet(j/float(num_estimate - 1))
            label = 's%d' % chain.options.seed
            if chain.pruned:
                line = ax.plot(chain.thinned_accept_steps, chain.positions[:,i],
                               color=color, label=label)
            else:
                line = ax.plot(chain.positions[:, i], color=color, label=label)

        ax.set_title("Parameter: %s" % param_name)
        ax.legend(loc='upper left', **legend_kwargs)
        plot_filename = '%s_trace_%s.png' % (mcmc_set.name, param_name)
        canvas = FigureCanvasAgg(fig)
        fig.set_canvas(canvas)
        fig.savefig(plot_filename)
        img_str_list.append(plot_filename)

    # Make the html file
    html_str += '<a href="%s"><img src="%s" width=400 /></a>' % \
                (posterior_plot_filename, posterior_plot_filename)
    html_str += '\n'.join([
        '<a href="%s"><img src="%s" width=400 /></a>' %
        (i, i) for i in img_str_list])
    html_str += "</body></html>"
    html_filename = '%s_convergence.html' % mcmc_set.name
    with open(html_filename, 'w') as f:
        f.write(html_str)

    return FloatListResult(convergence.convergence_criterion(mcmc_set),
                           html_filename)

@reporter('Maximum likelihood')
def maximum_likelihood(mcmc_set):
    # Get the maximum likelihood
    try:
        (max_likelihood, max_likelihood_position) = mcmc_set.maximum_likelihood()
    except NoPositionsException as npe:
        return Result(None, None)

    return show_fit_at_position(mcmc_set, max_likelihood,
                                max_likelihood_position, 'max_likelihood')

@reporter('Maximum posterior')
def maximum_posterior(mcmc_set):
    # Get the maximum posterior
    try:
        (max_posterior, max_posterior_position) = mcmc_set.maximum_posterior()
    except NoPositionsException as npe:
        return Result(None, None)

    return show_fit_at_position(mcmc_set, max_posterior,
                                max_posterior_position, 'max_posterior')

def show_fit_at_position(mcmc_set, fit_value, position, fit_name):
    """Create the result page showing the fit quality at the given position.

    Parameters
    ----------
    mcmc_set : MCMCSet object
        The set of MCMC chains
    fit_value : float
        The quality of fit at the given position.
    position : numpy.array
        Array of (log10-transformed) parameter values at the given position.
    fit_name : string
        A shorthand name for the fit position, e.g., "max_likelihood". Should
        conform to rules of Python variable naming (no spaces, doesn't
        start with a number, etc.).

    Returns
    -------
    A result object containing the fit value and the link to the accompanying
    HTML plot page, if any.
    """

    # If the MCMC object does not have a fit_plotting_function defined
    # (for example, if it is a base MCMC object), then don't create a
    # plot for visualization.
    if not hasattr(mcmc_set.chains[0], 'fit_plotting_function'):
        return Result(fit_value, None)

    # Prepare html for page showing plots at position
    html_str = "<html><head><title>Simulation of %s " \
               "with %s parameter values</title></head>\n" \
               % (mcmc_set.name, fit_name)
    html_str += "<body><p>Simulation of %s with %s " \
                "parameter values</p>\n" % (mcmc_set.name, fit_name)

    # Show the plot vs. the data at the position
    fig = mcmc_set.chains[0].fit_plotting_function(position=position)
    img_filename = '%s_%s_plot.png' % (mcmc_set.name, fit_name)
    fig.savefig(img_filename)
    html_str += '<p><img src="%s" /></p>' % img_filename

    # Show the plot of all observables at the position
    chain0 = mcmc_set.chains[0]
    tspan = chain0.options.tspan
    observables = chain0.options.model.observables
    x = chain0.simulate(position=position, observables=True)

    fig = Figure()
    ax = fig.gca()
    lines = []
    for o in observables:
        line = ax.plot(tspan, x[o.name])
        lines += line
    ax.set_title("Observables at %s" % fit_name)
    fig.legend(lines, [o.name for o in observables], 'lower right')
    canvas = FigureCanvasAgg(fig)
    fig.set_canvas(canvas)

    img_filename = '%s_%s_species.png' % (mcmc_set.name, fit_name)
    fig.savefig(img_filename)
    html_str += '<p><img src="%s" /></p>' % img_filename

    # Print the parameter values for the position as a dict that can be
    # used to override the initial values
    html_str += '<pre>%s_params = {\n' % fit_name
    for i, p in enumerate(chain0.options.estimate_params):
        html_str += "\t'%s': %.17g,\n" % \
                    (p.name, 10 ** position[i])
    html_str += '}</pre>'

    html_str += '</body></html>'

    # Create the html file
    html_filename = '%s_%s_plot.html' % (mcmc_set.name, fit_name)
    with open(html_filename, 'w') as f:
        f.write(html_str)

    return Result(fit_value, html_filename)

@reporter('Sample fits')
def sample_fits(mcmc_set):
    tspan = mcmc_set.chains[0].options.tspan
    fig = Figure()
    ax = fig.gca()
    plot_filename = '%s_sample_fits.png' % mcmc_set.name
    thumbnail_filename = '%s_sample_fits_th.png' % mcmc_set.name

    # Make sure we can call the method 'get_observable_timecourses'
    if not hasattr(mcmc_set.chains[0], 'get_observable_timecourses') or \
       not hasattr(mcmc_set.chains[0], 'plot_data'):
        return Result('None', None)

    # Plot the original data
    mcmc_set.chains[0].plot_data(ax)

    # Plot a sampling of trajectories from the original parameter set
    try:
        for i in range(num_samples):
            position = mcmc_set.get_sample_position()
            timecourses = mcmc_set.chains[0].get_observable_timecourses(
                                                                position=position)
            for obs_name, timecourse in timecourses.iteritems():
                ax.plot(tspan, timecourse, color='g', alpha=0.1, label=obs_name)
    except NoPositionsException as npe:
        pass

    canvas = FigureCanvasAgg(fig)
    fig.set_canvas(canvas)
    fig.savefig(plot_filename)
    fig.savefig(thumbnail_filename, dpi=10)

    return ThumbnailResult(thumbnail_filename, plot_filename)

@reporter('Marginals')
def marginals(mcmc_set):
    """Returns the vector of Gelman-Rubin convergence criterion values and a
    link to an HTML file containing plots of the traces of the walk for each
    parameter fitted."""

    # Prepare html for page showing plots of parameter traces
    html_str = "<html><head><title>Marginal distributions for " \
               "%s</title></head>\n" % mcmc_set.name
    html_str += "<body><p>Marginal distributions for for %s</p>\n" \
                % mcmc_set.name
    img_str_list = []

    # Make plots of parameter traces
    for i in range(mcmc_set.chains[0].num_estimate):
        param_name = mcmc_set.chains[0].options.estimate_params[i].name
        fig = Figure()
        ax = fig.gca()
        # Build a list of arrays containing positions for this parameter
        chains_for_param = []
        for chain in mcmc_set.chains:
            # Don't try to plot marginals for a chain with no accepted steps!
            if len(chain.positions) > 0:
                chains_for_param.append(chain.positions[:,i])
        # Plot the marginals
        ax.hist(chains_for_param, histtype='step')
        ax.set_title("Parameter: %s" % param_name)
        plot_filename = '%s_marginal_%s.png' % (mcmc_set.name, param_name)
        canvas = FigureCanvasAgg(fig)
        fig.set_canvas(canvas)
        fig.savefig(plot_filename)
        img_str_list.append(plot_filename)

    # Make the html file
    html_str += '\n'.join([
        '<a href="%s"><img src="%s" width=400 /></a>' %
        (i, i) for i in img_str_list])
    html_str += "</body></html>"
    html_filename = '%s_marginals.html' % mcmc_set.name
    with open(html_filename, 'w') as f:
        f.write(html_str)

    return Result(None, html_filename)

