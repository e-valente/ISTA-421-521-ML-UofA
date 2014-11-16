## cv_demo.py
# Port of cv_demo.m
# From A First Course in Machine Learning, Chapter 1.
# Simon Rogers, 31/10/11 [simon.rogers@glasgow.ac.uk]
# Demonstration of cross-validation for model selection
# Translated to python by Ernesto Brau, 7 Sept 2012

# NOTE: In its released form, this script will NOT run
#       You will get a syntax error on line 80 because w has not been defined

import numpy as np
import matplotlib.pyplot as plt

def read_data(filepath, d = ','):
    """ returns an np.matrix of the data """
    return np.asmatrix(np.genfromtxt(filepath, delimiter = d, dtype = None))


def fitpoly(x, t, model_order):
    
    #### YOUR CODE HERE ####
    w = None  ### Calculate w column vector (as an np.matrix)

    myarray = np.ones((x.shape[0], 1))
    #print(x.shape)
    #print(myarray.shape)
    #myarray = np.concatenate((myarray, x))
    x = np.column_stack((myarray, x))

    #clear our array
    myarray = np.zeros((x.shape[0], 1))

    mypow = 2;
    for col in range(model_order -1):
        myarray = np.power(x.T[1], mypow)
        #print(myarray)
        #rint(x.T[1])
        print("vai\n\n")
        mypow += 1
        x = np.column_stack((x,myarray.T))
        
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

def makeTestMatrix_X(indexArray, X):
    testX = np.zeros(N-K)

    for i in range(N):
        if ((i in indexArray) == False):
            testX()


# make plots interactive
plt.ion()

## Generate some data
# Generate x between -5 and 5
#N = 100
#x = 10*np.random.rand(N, 1) - 5
#t = 5*x**3 - x**2 + x + 150*np.random.randn(x.shape[0], x.shape[1])

N = 50


filepath1 = '../data/synthdata2014.csv'   ## Problem 4


## Run a cross-validation over model orders
maxorder = 7
Data = read_data(filepath1, ',')

X = Data[:, 0] # extract x (slice first column)
t = Data[:, 1] # extract t (slice second column)
#X = np.asmatrix(np.zeros(shape = (x.shape[0], maxorder + 1)))
#testX = np.asmatrix(np.zeros(shape = (testx.shape[0], maxorder + 1)))

K = 10 # K-fold CV
N = Data.shape[0]

indexArray = X[:]
np.random.seed(1)
np.random.shuffle(indexArray)

print indexArray
exit()

for i in range(K):
    print "k= ", i
    for j in range(N/K):
        #print ((i*(N/K)) + j)
        print indexArray[(i*(N/K)) + j]
        test_X = makeTestMatrix_X(indexArray, X)

print indexArray


raw_input('Press <ENTER> to quit...')