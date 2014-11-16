from scipy.stats import norm
from scipy.stats import beta
from scipy import misc
import numpy as np
import math
import matplotlib.pyplot as plt



#1
a = 5
b = 5
N = 20
y = 10

#2
'''
a = 3
b = 15
N = 10
y = 3
'''

#3
'''
a = 1
b = 30
N = 10
y = 3
'''
mean = (y + a -1.0)/(a + N + b -2.0)
mean2 = mean * mean 

variance = -(mean2 * (mean -1.0) * (mean -1.0)) / \
	( ((mean -1.0) * (mean -1.0)) * (-y - a +1.0) \
	+ (mean2 * (-N + y - b +1.0)) )

print "mean: ", mean 
print "variance", variance
x = np.arange(0, 1, 0.01)
plt.plot(x, beta.pdf(x, a + y, b + N -y), 'b-',lw=4, alpha=6, label='true beta')
plt.plot(x, norm.pdf(x, loc=mean, scale=math.sqrt(variance)), 'r-',lw=4, alpha=6, label='Laplace est.')
plt.legend()
plt.show()
