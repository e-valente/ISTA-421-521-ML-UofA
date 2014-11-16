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

    myarray = np.ones((x.shape[0], 1))
    print(x.shape)
    print(myarray.shape)
    #myarray = np.concatenate((myarray, x))
    x = np.column_stack((myarray, x))

    #np.insert(x,myarray, axis=0)
    #x[:,:-1] = myarray

    #print(x)

    #m1 = ((X^t)X)
    m1 = x.T.dot(x)

    #print m1

    #m2 = m1^-1 = ((X^t)X)^1

    m2 = np.linalg.inv(m1)

    #print m2

    #m3 = m2.X^t = (((X^t)X)^1) X^t

    m3 = m2.dot(x.T)

    w = m3.dot(t)

    
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

#Data is a Matrix with all your data 
Data = read_data(filepath1, ',')

x = Data[:, 0] # extract x (slice first column)
t = Data[:, 1] # extract t (slice second column)

#print(Data[0,1])
#print(t)
plt.ion()

plot_data(x, t)
#print(Data.shape)
#print(t)
w = fitpoly(x, t, model_order)
#print(w.shape)
#print(w)

print 'Identified model parameters:'
print w

plot_model(x, w)

# This last line may or may not be necessary to keep your matplotlib window open
raw_input('Press <ENTER> to quit...')
