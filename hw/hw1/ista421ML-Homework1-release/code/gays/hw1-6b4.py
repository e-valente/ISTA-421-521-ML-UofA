## plotsin.py

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

## Define points for the x-axis
#from 0.0 to 10.0 (steps: 0.01)
#x = np.arange(0, 10, 0.01);
seed = 1
np.random.seed(seed)
a = np.random.rand(3)
b = np.random.rand(3)

seed = 2
np.random.seed(seed)
X = np.random.rand(3, 3)


#print "a = " + str(a)
#print "b = " + str(b)
#print "X = \n" + str(X)

print "aT.X = \n" + str(np.dot(a.T,X))


