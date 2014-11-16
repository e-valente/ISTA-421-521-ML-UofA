## w_variation_demo.py
# Port of w_variation_demo.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# The bias in the estimate of the variance
# Generate lots of datasets and look at how the average fitted variance
# agrees with the theoretical value
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.ion()
## Generate the datasets and fit the parameters
true_w = np.mat([[-2], [3]])
Nsizes = np.arange(20, 1020, 20)
N_data = 10000 # Number of datasets
all_ss = np.zeros((Nsizes.shape[0], N_data))

total = Nsizes.shape[0]

print 'NOTE: this will take a long time to run... genereating', total, 'datasets'
for j in range(total):

    print 'processing dataset',j+1,' of',total
    
    N = Nsizes[j] # Number of objects
    x = np.asmatrix(np.random.rand(N)).conj().transpose()
    X = np.asmatrix(np.zeros((x.shape[0], 2)))
    X[:, 0] = np.power(x, 0)
    X[:, 1] = np.power(x, 1)
    noisevar = 0.5**2
    for i in range(N_data):
        t = X*true_w + np.asmatrix(np.random.randn(N)).conj().transpose()*np.sqrt(noisevar)
        w = np.linalg.inv(X.conj().transpose()*X)*X.conj().transpose()*t
        ss = (1./N)*(t.conj().transpose()*t - t.conj().transpose()*X*w)
        all_ss[j, i] = ss


## The expected value of the fitted variance is equal to:
# $\sigma^2\left(1-\frac{D}{N}\right)$
# where $D$ is the number of dimensions (2) and $\sigma^2$ is the true
# variance.
# Plot the average empirical value of the variance against the 
# theoretical expected value as the size of the datasets increases
plt.figure()
plt.scatter(Nsizes, np.mean(all_ss, 1), color = 'white', s = 40,
            edgecolor = 'black', label = 'Empirical')
plt.plot(Nsizes, noisevar*(1-2./Nsizes), color = 'r', linewidth = 2, label = 'Theoretical')
plt.plot(Nsizes, [0.25 for i in range(Nsizes.shape[0])],
         color = 'b', linestyle = 'dashed', label = 'Actual')
plt.xlabel('Dataset size')
plt.ylabel('Variance')
plt.legend(loc=4)
plt.pause(.1) # required on some systems so that rendering can happen

raw_input('Press <ENTER> to continue...')

