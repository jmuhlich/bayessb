import numpy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib

# some code to explore the proper math and code for hessian-guided sampling

# a slightly rotated hyperbolic paraboloid
hessian = numpy.array([[2,3],[3,-10]])

eig_val, eig_vec = numpy.linalg.eig(hessian)
adj_eig_val = numpy.maximum(abs(eig_val), 0.25)

# sample normally
r = numpy.random.randn(2,1000);
# transform samples
pr = (eig_vec / adj_eig_val ** 0.5).dot(r)
dir = eig_vec * eig_val

grid_1d = numpy.arange(-10, 10.5, 0.5)
gx, gy = numpy.meshgrid(grid_1d, grid_1d)
# flatten into N-by-2 list of vectors
g = numpy.dstack((gx, gy))
# compute quadric surface using quadratic form x'Ax (second multiplication
# "fakes it" with element-wise multiply and sum since we are actually operating
# on a list of vectors and not a single vector x)
psurf = (g.dot(hessian) * g).sum(2)

fig = plt.figure()
ax = fig.gca(projection='3d')
# quadric surface
ax.plot_surface(gx, gy, psurf, rstride=1, cstride=1, cmap=matplotlib.cm.jet,
                linewidth=0, alpha=0.2)
# hessian-biased samples
ax.scatter(pr[0,:], pr[1,:], c='r', s=20, marker='o')
# eigenvectors
ax.plot([0, dir[0,0]], [0, dir[1,0]], color='g')
ax.plot([0, dir[0,1]], [0, dir[1,1]], color='b')

plt.xlabel('X')
plt.ylabel('Y')
plt.axis([-10, 10, -10, 10])
plt.gca().set_aspect(1.0)
plt.show()
