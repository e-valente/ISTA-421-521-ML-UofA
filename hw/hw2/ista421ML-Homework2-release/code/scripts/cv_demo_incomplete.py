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

# make plots interactive
plt.ion()

## Generate some data
# Generate x between -5 and 5
N = 100
x = 10*np.random.rand(N, 1) - 5
t = 5*x**3 - x**2 + x + 150*np.random.randn(x.shape[0], x.shape[1])

xgen = np.linspace(-5, 5, 100)
tgen = 5*xgen**3  - xgen**2 + xgen

testx = np.linspace(-5, 5, 1001) # Large, independent test set
testt = 5*testx**3 - testx**2 + testx + 150*np.random.randn(testx.shape[0], 1)

# plot the synthetic data
plt.close()
plt.figure(0)
plt.scatter(np.asarray(x), np.asarray(t), edgecolor = 'b', color = 'w', marker = 'o')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Plot of synthetic data; green curve is original generating function');
plt.plot(xgen , tgen, color='g', linewidth=2)
plt.pause(.1) # required on some systems so that rendering can happen

# convert everything to matrices
x = np.asmatrix(x)
t = np.asmatrix(t)
testx = np.asmatrix(testx).conj().transpose()
testt = np.asmatrix(testt).conj().transpose()

## Run a cross-validation over model orders
maxorder = 7
X = np.asmatrix(np.zeros(shape = (x.shape[0], maxorder + 1)))
testX = np.asmatrix(np.zeros(shape = (testx.shape[0], maxorder + 1)))

K = 10 # K-fold CV
sizes = np.floor(N/K)*np.ones(K)
csizes = np.zeros(sizes.shape[0] + 1)
csizes[1:csizes.shape[0]] = sizes.cumsum()

cv_loss = np.zeros((K, maxorder + 1))
ind_loss = np.zeros((K, maxorder + 1))
train_loss = np.zeros((K, maxorder + 1))

for k in range(maxorder + 1):
    X[:, k] = np.power(x, k)
    testX[:, k] = np.power(testx, k)

    for fold in range(K):
        # Partition the data
        # foldX contains the data for just one fold
        # trainX contains all other data
        
        foldX = X[csizes[fold]+1:csizes[fold+1]+1,0:k+1]
        foldt = t[csizes[fold]+1:csizes[fold+1]+1,:]
        
        trainX = np.copy(X[:, 0:k+1])
        # remove the fold x from the training set:
        trainX = np.asmatrix(np.delete(trainX, np.arange(csizes[fold]+1, csizes[fold+1]+1), 0))
        
        traint = np.copy(t)
        # remove the fold t from the training set:
        traint = np.asmatrix(np.delete(traint, np.arange(csizes[fold]+1, csizes[fold+1]+1), 0))

        # find the least-squares fit!
        # NOTE: YOU NEED TO FILL THIS IN
        w = #### YOUR CODE HERE ####

        # calculate losses
        fold_pred = foldX*w
        if foldt.shape[0] == 1:
            cv_loss[fold,k] = np.power(fold_pred - foldt, 2)
        else:
            cv_loss[fold,k] = np.mean(np.power(fold_pred - foldt, 2))
        ind_pred = testX[:,0:k+1]*w
        ind_loss[fold,k] = np.mean(np.power(ind_pred - testt, 2))
        train_pred = trainX*w
        train_loss[fold,k] = np.mean(np.power(train_pred - traint, 2))

# The results look a little more dramatic if you display the loss on
# log scale, so the following scales the loss scores
log_cv_loss = np.log(cv_loss)
log_train_loss = np.log(train_loss)
log_ind_loss = np.log(ind_loss)

## Plot log scale loss results
plt.figure(1);
plt.title('Log-scale Loss')

plt.subplot(131)
plt.plot(np.arange(0, maxorder + 1), np.mean(log_cv_loss, 0), linewidth = 2)
plt.xlabel('Model Order')
plt.ylabel('Log Loss')
plt.title('CV Loss')
plt.pause(.1) # required on some systems so that rendering can happen

plt.subplot(132)
plt.plot(np.arange(0, maxorder + 1), np.mean(log_train_loss, 0), linewidth = 2)
plt.xlabel('Model Order')
plt.ylabel('Log Loss')
plt.title('Train Loss')
plt.pause(.1) # required on some systems so that rendering can happen

plt.subplot(133)
plt.plot(np.arange(0, maxorder + 1), np.mean(log_ind_loss, 0), linewidth = 2)
plt.xlabel('Model Order');
plt.ylabel('Log Loss');
plt.title('Independent Test Loss')
plt.pause(.1) # required on some systems so that rendering can happen

plt.subplots_adjust(right=0.95, wspace=0.5)

# remove the quotes to the following to see them ploted without log-scaling
'''
## Plot the linear-scale loss results
plt.figure(2);
plt.subplot(131)
plt.plot(np.arange(0, maxorder + 1), np.mean(cv_loss, 0), linewidth = 2)
plt.xlabel('Model Order');
plt.ylabel('LLoss');
plt.title('CV Loss');

plt.subplot(132)
plt.plot(np.arange(0, maxorder + 1), np.mean(train_loss, 0), linewidth = 2)
plt.xlabel('Model Order');
plt.ylabel('Loss');
plt.title('Train Loss');

plt.subplot(133)
plt.plot(np.arange(0, maxorder + 1), np.mean(ind_loss, 0), linewidth = 2)
plt.xlabel('Model Order');
plt.ylabel('Loss');
plt.title('Independent Test Loss')

plt.subplots_adjust(right=0.95, wspace=0.5)
'''

# This last line may or may not be necessary to keep your matplotlib window open
raw_input('Press <ENTER> to quit...')