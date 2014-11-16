## Solution for ISTA 421/521 Fall 2014, HW 2, Problem 1
## Author: Clayton T. Morrison, 14 September 2014

import numpy as np
import matplotlib.pyplot as plt

### -------------------------------------------------------------------------

def read_data(filepath, d = ','):
    """ returns an np.matrix of the data """
    return np.asmatrix(np.genfromtxt(filepath, delimiter = d, dtype = None))

def plot_data(x, t):
    plt.figure()
    plt.scatter(np.asarray(x), np.asarray(t),
                edgecolor = 'b', color = 'w', marker = 'o')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Data')
    plt.pause(.1) # required on some systems so that rendering can happen
    
def plot_model(x, w):
    flatx = np.asarray(x).flatten()
    flatx = np.asarray(sorted(flatx))
    plotx = np.asmatrix(np.linspace(flatx[0]-1, flatx[-1]+1,
                                    (flatx[-1]-flatx[0]+4)/0.01 + 1)).conj().transpose()
    plotX = np.asmatrix(np.zeros((plotx.shape[0], w.size)))
    for k in range(w.size):
        plotX[:,k] = np.power(plotx, k)
    plt.plot(plotx, plotX*w, color = 'r', linewidth = 2)
    plt.pause(.1) # required on some systems so that rendering can happen
    return plotx, plotX*w

### -------------------------------------------------------------------------

def fitpoly(x, t, model_order):
    
    #### YOUR CODE HERE ####
    w = None  ### Calculate w column vector (as an np.matrix)
    
    return w

### -------------------------------------------------------------------------
### Script to run on particular data set

filepath1 = '../data/womens100.csv'       ## Problem 2
#filepath1 = '../data/synthdata2014.csv'   ## Problem 4

## The following data is provided just for fun, not used in HW 2.
## This is the data for the men's 100, which has been the recurring 
## example in the class
#filepath1 = '../data/mens100.csv'

model_order = 1 # for problem 2
#model_order = 3 # for problem 4

Data = read_data(filepath1, ',')

x = Data[:, 0] # extract x (slice first column)
t = Data[:, 1] # extract t (slice second column)

plt.ion()

plot_data(x, t)
w = fitpoly(x, t, model_order)

print 'Identified model parameters:'
print w

plot_model(x, w)

# This last line may or may not be necessary to keep your matplotlib window open
# raw_input('Press <ENTER> to quit...')
