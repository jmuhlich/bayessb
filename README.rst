BioMC
=====

Bayesian model inference and parameter estimation for biological models

http://sorgerlab.github.com/biomc

BioMC is a Python software library for performing Bayesian model inference and
parameter estimation on biological models, using a Markov Chain Monte
Carlo (MCMC) technique described in the paper *Eydgahi et al. Bayesian parameter
estimation and model discrimination for complex biochemical networks. Mol Syst
Biol (in review)*.


Installation
------------

BioMC depends only on `numpy <http://numpy.scipy.org/>`_ for its core
functionality.  However `matplotlib <http://matplotlib.org/>`_ is also needed if
you wish to use the provided plotting routines in ``biomc.plot``.  `PySB
<http://pysb.org/>`_ is our preferred tool for building models which integrate
well with BioMC, but it is not required.

Documentation
-------------

The code has extensive inline documentation.  Once you have installed the
package, run ``pydoc biomc`` and ``pydoc biomc.plot`` to view it.

For a fairly simple example of BioMC usage, see the script
``examples/earm/fit_1_3_standalone.py``.

Future Directions
-----------------

* We should absolutely use the PyMC package instead of rolling our own MCMC
  code.
* Better support of non-PySB models, e.g. SBML import.
* Implementation of the Bayes factor calculations from the MSB paper.
