import numpy as np
from matplotlib import pyplot as plt
from bayessb.report import reporter, Result, FloatListResult
from bayessb import convergence
from StringIO import StringIO

reporter_group_name = "Estimation"

@reporter('Number of chains')
def num_chains(mcmc_set):
    return Result(len(mcmc_set.chains), None)

"""
@reporter('Estimation parameters')
def estimation_parameters(mcmc_set):
    chain = mcmc_set.chains[0]
    opts = chain.options
    output = StringIO()
    output.write("use_hessian: %s, " % opts.use_hessian)
    output.write("hessian_period: %s, " % opts.hessian_period)
    output.write("hessian_scale: %s, " % opts.hessian_scale)
    output.write("norm_step_size: %s, " % opts.norm_step_size)
    output.write("sigma_adj_interval: %s, " % opts.sigma_adj_interval)
    output.write("anneal_length: %s, " % opts.anneal_length)
    output.write("T_init: %s, " % opts.T_init)
    output.write("accept_rate_target: %s, " % opts.accept_rate_target)
    output.write("sigma_max: %s, " % opts.sigma_max)
    output.write("sigma_min: %s, " % opts.sigma_min)
    output.write("sigma_step: %s, " % opts.sigma_step)
    output.write("seed: %s, " % opts.seed)
    #output.write("accept_window: %s, " % opts.accept_window)
    return Result(output.getvalue(), None)
"""

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

    # Make plots of parameter traces
    for i in range(mcmc_set.chains[0].num_estimate):
        param_name = mcmc_set.chains[0].options.estimate_params[i].name
        plt.figure()
        for chain in mcmc_set.chains:
            if chain.pruned:
                plt.plot(chain.thinned_accept_steps, chain.positions[:,i])
            else:
                plt.plot(chain.positions[:, i])
        plt.title("Parameter: %s" % param_name)
        plot_filename = '%s_trace_%s.png' % (mcmc_set.name, param_name)
        plt.savefig(plot_filename)
        img_str_list.append(plot_filename)

    # Make the html file
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
    (max_likelihood, max_likelihood_position) = mcmc_set.maximum_likelihood()

    # Plot the maximum likelihood fit
    if hasattr(mcmc_set.chains[0], 'fit_plotting_function'):
        mcmc_set.chains[0].fit_plotting_function(
                                        position=max_likelihood_position)
        filename = '%s_max_likelihood_plot.png' % mcmc_set.name
        plt.savefig(filename)
    else:
        filename = None

    return Result(max_likelihood, filename)

@reporter('Maximum posterior')
def maximum_posterior(mcmc_set):
    # Get the maximum posterior
    (max_posterior, max_posterior_position) = mcmc_set.maximum_posterior()

    # Plot the maximum posterior fit
    if hasattr(mcmc_set.chains[0], 'fit_plotting_function'):
        mcmc_set.chains[0].fit_plotting_function(
                                        position=max_posterior_position)
        filename = '%s_max_posterior_plot.png' % mcmc_set.name
        plt.savefig(filename)
    else:
        filename = None

    # Return the max posterior along with link to the plot
    return Result(max_posterior, filename)

