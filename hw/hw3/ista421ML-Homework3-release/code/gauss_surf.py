## gauss_surf.py
## Port of gauss_surf.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Surface and contour plots of a Gaussian
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
plt.ion()
## The Multi-variate Gaussian pdf is given by:
# $p(\mathbf{x}|\mu,\Sigma) =
# \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\left\{-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right\}$
## Define the Gaussian
mu = np.array([1, 2])
sigma = np.mat([[1, 0.8], [0.8, 4]])
## Define the grid for visualisation
X,Y = np.meshgrid(np.arange(-5., 5.1, 0.1), np.arange(-5., 5.1, 0.1))
## Define the constant
const = (1/np.sqrt(2*np.pi))**2
const = const/np.sqrt(np.linalg.det(sigma))
temp = np.concatenate((np.asmatrix(X.flatten(1)).conj().transpose()
                       - mu[0], np.asmatrix(Y.flatten(1)).conj().transpose() - mu[1]), 1)
pdfv = const*np.exp(-0.5*np.diag(temp*np.linalg.inv(sigma)*temp.conj().transpose()))
pdfv = np.reshape(pdfv, X.shape).conj().transpose()
## Make the plots
plt.figure(1)
plt.contour(X, Y, pdfv)
fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, pdfv, rstride=1, cstride=1,
                       cmap=matplotlib.cm.jet, linewidth=0, antialiased=False)
fig.colorbar(surf)
plt.pause(.1) # required on some systems so that rendering can happen

raw_input('Press <ENTER> to continue...')

