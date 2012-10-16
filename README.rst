BayesSB
=======

Bayesian model inference and parameter estimation for biological models

http://sorgerlab.github.com/bayessb

BayesSB is an algorithm and software suite for estimating parameter
distributions in ODE-based models of cellular biochemistry and for
discriminating between models having different numbers of unknown
parameters. The algorithm is described in detail in *Eydgahi et al. Bayesian
parameter estimation and model discrimination for complex biochemical
networks. Mol Syst Biol (in review)*.

The procedure returns joint probability distributions for model parameters and
makes it possible to compute uncertainty for model-based predictions based on
error in the data and the non-identifiability of model parameters. Bayesian
methods also make it possible to compute the odds ratio for competing models
having different numbers of parameters.

Installation
------------

BayesSB depends only on `numpy <http://numpy.scipy.org/>`_ for its core
functionality.  However `matplotlib <http://matplotlib.org/>`_ is also needed if
you wish to use the provided plotting routines in ``bayessb.plot``.  `PySB
<http://pysb.org/>`_ is our preferred tool for building models which integrate
well with BayesSB, but it is not required.

Documentation
-------------

The code has extensive inline documentation.  Once you have installed
the package, run ``pydoc bayessb`` and ``pydoc bayessb.plot`` to view
it.

For a fairly simple example of BayesSB usage, see the script
``examples/earm/fit_1_3_standalone.py``.

Future Directions
-----------------

* We should absolutely use the PyMC package instead of rolling our own MCMC
  code.
* Better support of non-PySB models, e.g. SBML import.
* Implementation of the Bayes factor calculations from the MSB paper.
