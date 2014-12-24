import numpy as np
import scipy.optimize
import utils_hw
import gradient
import display_network
import load_MNIST
import sys
import time

##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to
#  change the parameters below.

# number of input units
visible_size = 28 * 28
# number of input units
hidden_size = 25

# desired average activation of the hidden units.
# (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
#  in the lecture notes).

# weight decay parameter
lambda_ = 0.0001

# debug (set to True in Ex 3)
debug = False


##======================================================================
## Ex 1: Load MNIST
## In this example, you will load the mnist dataset
## First download the dataset from the following website: 
##Training Images: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
##Training Labels: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

# Loading Sample Images
 #Loading 10K images from MNIST database
images = load_MNIST.load_MNIST_images('../../data/mnist/train-images-idx3-ubyte')
patches = images[:, 0:10000]
patches = patches[:,1:200]
display_network.display_network(patches[:,1:100])