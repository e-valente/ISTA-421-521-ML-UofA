## Extension of predictive_variance_example.py
# Port of predictive_variance_example.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Predictive variance example

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# change this to where you'd like the figures saved
figure_path = '../figs/'
# set to True in order to save figures
SAVE_FIGURES = False

def true_function(x):
    """$t = 1 + 0.1x + 0.5x^2 + 0.05x^3$"""
    return 1 + (0.1 * x) + (0.5 * np.power(x,2)) + (0.05 * np.power(x,3))

def sample_from_function(N=100, noise_var=1000, xmin=-5., xmax=5.):
    """ Sample data from the true function.
        N: Number of samples
        Returns a noisy sample t_sample from the function
        and the true function t. """
    #x_range = xmax - xmin
    #x_mid = x_range/2.
    #x = np.sort(x_range*np.random.rand(N) - x_mid)
    x = np.random.uniform(xmin, xmax, N)
    t = true_function(x)
    # add standard normal noise using np.random.randn
    # (standard normal is a Gaussian N(0, 1.0),
    #  so multiplying by np.sqrt(noise_var) make it N(0,noise_ver))
    t = t + np.random.randn(x.shape[0])*np.sqrt(noise_var)
    return x,t

xmin = -12.
xmax = 5.
noise_var = 6

## sample 100 points from function
x,t = sample_from_function(100, noise_var, xmin, xmax )

# Chop out some x data
# the following line expresses a boolean function over the values in x;
# this produces a list of the indices of list x for which the test
# was not met; these indices are then deleted from x and t.
xmin_remove = -2 # -0.5
xmax_remove = 2 # 2.5
pos = ((x>=xmin_remove) & (x<=xmax_remove)).nonzero()
x = np.delete(x, pos, 0)
t = np.delete(t, pos, 0)

# reshape the x and t to be column vectors in an np.matrix form
# so that we can perform matrix operations.
x = np.asmatrix(np.reshape(x,(x.shape[0],1)))
t = np.asmatrix(np.reshape(t,(t.shape[0],1)))

## Plot just the sampled data
plt.figure(0)
plt.scatter(np.asarray(x), np.asarray(t), color = 'k', edgecolor = 'k')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Sampled data from {0}, $x \in [{1},{2}]$'.format(true_function.__doc__,xmin,xmax))
plt.pause(.1) # required on some systems so that rendering can happen
if SAVE_FIGURES:
    plt.savefig(figure_path + 'data')

## Fit models of various orders
orders = [1,3,5,9]

## Make a set of 100 evenly-spaced x values between xmin and xmax
testx = np.asmatrix(np.linspace(xmin, xmax, 100)).conj().transpose()

## Generate plots of predicted variance (error bars) for various model orders
for i in orders:
    # create input representation for given model polynomial order
    X = np.asmatrix(np.zeros(shape = (x.shape[0], i + 1)))
    testX = np.asmatrix(np.zeros(shape = (testx.shape[0], i + 1)))
    for k in range(i + 1):
        X[:, k] = np.power(x, k)
        testX[:, k] = np.power(testx, k)
    N = X.shape[0]

    # fit model parameters
    w = np.linalg.inv(X.T*X)*X.T*t
    ss = (1./N)*(t.T*t - t.T*X*w)

    # calculate predictions
    testmean = testX*w
    testvar = ss * np.diag(testX*np.linalg.inv(X.T*X)*testX.T)

    # Plot the data and predictions
    plt.figure()
    plt.scatter(np.asarray(x), np.asarray(t), color = 'k', edgecolor = 'k')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.errorbar(np.asarray(testx.T)[0], np.asarray(testmean.T)[0], np.asarray(testvar)[0])
    
    # find ylim plot bounds automagically...
    testmean_flat = np.asarray(testmean).flatten()
    testvar_flat = np.asarray(testvar).flatten()
    min_model = min(testmean_flat - testvar_flat)
    max_model = max(testmean_flat + testvar_flat)
    min_testvar = min(min(np.asarray(t).flatten()), min_model)
    max_testvar = max(max(np.asarray(t).flatten()), max_model)
    plt.ylim(min_testvar,max_testvar)
    
    ti = 'Plot of predicted variance for model with polynomial order {:g}'.format(i)
    plt.title(ti)
    plt.pause(.1) # required on some systems so that rendering can happen

    if SAVE_FIGURES:
        filename = 'error-{0}'.format(i)
        plt.savefig(figure_path + filename)

## Generate plots of functions whose parameters are sampled based on cov(\hat{w})
num_function_samples = 20
for i in orders:
    # create input representation for given model polynomial order
    X = np.asmatrix(np.zeros(shape = (x.shape[0], i + 1)))
    testX = np.asmatrix(np.zeros(shape = (testx.shape[0], i + 1)))
    for k in range(i + 1):
        X[:, k] = np.power(x, k)
        testX[:, k] = np.power(testx, k)

    # fit model parameters
    w = np.linalg.inv(X.T*X)*X.T*t
    ss = (1./N)*(t.T*t - t.T*X*w)
    
    # Sample functions with parameters w sampled from a Gaussian with
    # $\mu = \hat{\mathbf{w}}$
    # $\Sigma = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$
    # determine cov(w)
    covw = np.asarray(ss)[0][0]*np.linalg.inv(X.T*X)
    # The following samples num_function_samples of w from Gaussian based on covw
    wsamp = np.random.multivariate_normal(np.asarray(w.T)[0], covw, num_function_samples)
    # Calculate means for each function
    testmean = testX*wsamp.T
    
    # Plot the data and functions
    plt.figure()
    plt.scatter(np.asarray(x), np.asarray(t), color = 'k', edgecolor = 'k')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.plot(np.asarray(testx), np.asarray(testmean), color = 'b')
    
    # find reasonable ylim bounds
    plt.xlim(xmin_remove-2, xmax_remove+2) # (-2,4) # (-3, 3)
    min_model = min(np.asarray(testmean).flatten())
    max_model = max(np.asarray(testmean).flatten())
    min_testvar = min(min(np.asarray(t).flatten()), min_model)
    max_testvar = max(max(np.asarray(t).flatten()), max_model)
    plt.ylim(min_testvar, max_testvar) # (-400,400)
    
    ti = 'Plot of {0} functions where parameters '\
         .format(num_function_samples, i) + \
         r'$\widehat{\bf w}$ were sampled from' + '\n' + r'cov($\bf w$)' + \
         ' of model with polynomial order {1}' \
         .format(num_function_samples, i)
    plt.title(ti)
    plt.pause(.1) # required on some systems so that rendering can happen
    
    if SAVE_FIGURES:
        filename = 'sampled-fns-{0}'.format(i)
        plt.savefig(figure_path + filename)
    
    #raw_input('Press <ENTER> to see next plot...')

raw_input('Press <ENTER> to continue...')

