## approx_expected_value.py
# Port of approx_expected_value.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Approximating expected values via sampling
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

## We are trying to compute the expected value of
# $y^2$
##
# Where
# $p(y)=U(0,1)$
## 
# Which is given by:
# $\int y^2 p(y) dy$
##
# The analytic result is:
# $\frac{1}{3}$

## Generate samples
ys = 15 * np.random.rand(3000, 1) -10
# compute the expectation
#ey2 = np.mean(np.power(ys, 2))
#print '\nSample-based approximation: {:f}'.format(ey2)
print '\nSample-based approximation: {:f}'.format(5.4375)
## Look at the evolution of the approximation, every 10 samples
posns = np.arange(0, ys.shape[0]+1, 10) #[1 11 21 .... 91]
print posns
ey2_evol = np.zeros((posns.shape[0]))
# the following computes the mean of the sequence up to i, as i iterates 
# through the sequence, storing the mean in ey2_evol:
for i in range(posns.shape[0]):
	if i == 0:
	  x = ys[0]
	  ey2_evol[0] = 1 + 0.1 * x + (0.5 * x * x) + (0.05 * x *x *x)
	  continue
	sum = 0.0
	count = 0
	for j in range(posns[i]):
		x = ys[j]
		sum += 1 + 0.1 * x + (0.5 * x * x) + (0.05 * x *x *x)
		count += 1

	ey2_evol[i] = sum / count
    #ey2_evol[i] = np.mean(np.power(ys[0:posns[i]], 2))  
plt.figure(1)
plt.plot(posns, ey2_evol)
print ey2_evol
# the true, analytic result of the expected value: $\frac{1}{3}$
#plt.plot(np.array([posns[0], posns[-1]]), np.array([1./3, 1./3]), color='r')
plt.plot(np.array([posns[0], posns[-1]]), np.array([5.437, 5.437]), color='r')
plt.xlabel('Samples')
plt.ylabel('Approximation')
plt.pause(.1) # required on some systems so that rendering can happen

raw_input('Press <ENTER> to continue...')

