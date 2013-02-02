"""A file to make sure we are setting the mean and variance for PyMC
Lognormal variables correctly--since we are used to describing them with
the mean and variance in log base 10."""
from pymc import deterministic, stochastic, MvNormal, Normal, Lognormal, Uniform
from pymc import MCMC, Model
import numpy as np
from pylab import *

# The mu and tau are in log units; to get to log units,
# do the following
# (has mean around 1e2, with a variance of 9 logs in base 10)
mean_b10 = 2
var_b10 = 9

print "Setting mean (base 10) to %f, variance (base 10) to %f" % (mean_b10, var_b10)

# The lognormal variable
k = Lognormal('k', mu=np.log(10 ** mean_b10),
                   tau=1./(np.log(10) * np.log(10 ** var_b10)))

# Sample it
m = MCMC(Model([k]))
m.sample(iter=50000)

ion()

# Plot the distribution in base e
figure()
y = log(m.trace('k')[:])
y10 = log10(m.trace('k')[:])
hist(y, bins=100)
print
print "Mean, base e: %f; Variance, base e: %f" % (mean(y), var(y))

# Plot the distribution in base 10
figure()
hist(y10, bins=100)
print "Mean, base 10: %f; Variance, base 10: %f" % (mean(y10), var(y10))

