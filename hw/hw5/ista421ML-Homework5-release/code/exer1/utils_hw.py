#This is a set of utilities to run the NN excersis in ISTA 421, Introduction to ML
#By Leon F. Palafox, December, 2014
import numpy as np
import math, sys
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
  
def sigmoid_derivative(x):
    my_exp = np.exp(x)
    return (my_exp)/((1 + my_exp ) * (1 + my_exp))
  
def initialize(hidden_size, visible_size):
    # we'll choose weights uniformly from the interval [-r, r] following what we saw in class
    
    #as we saw in the class: n_in + n_out = hidden_size + visible_size
    my_guess_range = math.sqrt(6) / math.sqrt(hidden_size + visible_size + 1)
    
    #In our example, hidden_size = 3 + visible_size = 2 => it will generates
    #number between -1 and 1 since myguess_range = 1 (in this case)
    my_w1 = 2 * my_guess_range * np.random.rand((hidden_size * visible_size)) - my_guess_range
    #we will reshape, so it will be 1x(hidden_size + visible_size)
    my_w1 = my_w1.reshape(hidden_size * visible_size)
    
    #the same thing for w2
    my_w2 = 2 * my_guess_range * np.random.rand((hidden_size * visible_size)) - my_guess_range
    #we will reshape, so it will be 1x(hidden_size + visible_size)
    my_w2 = my_w2.reshape(hidden_size * visible_size)
    
    #b_1 is our first bias unit ( so it is the input from the visible layer to the hidden layer)
    #it will have "hidden_size" arrows
    my_b1 = np.ones(hidden_size)
    #reshapping -> 1xlength array
    my_b1 = my_b1.reshape(hidden_size)
    
    #on the other hand b_2 will have visible_size arrows since it is the 
    #input from the hidden layer to visible layer
    my_b2 = np.ones(visible_size)
    #reshapping -> 1xlength array
    my_b2 = my_b2.reshape(visible_size)
    
    #concatenating in order to we have a 1X(length) (ONE LINE) array 
    #we will have [w1, w2, b1, b2]
    theta = np.concatenate((my_w1, my_w2, my_b1, my_b2))
    #print(len(theta))

    #print("w1 shape ", my_w1.shape(0))

    return theta


def sparse_autoencoder_cost(theta, visible_size, hidden_size,
                            lambda_, data):
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.
        # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.
    
    weights_length = hidden_size * visible_size
    b1_lenth = hidden_size
    b2_length = visible_size
    
    #unwrapping our arrays
    my_W1 = theta[0:weights_length].reshape(hidden_size, visible_size)
    
    my_W2 = theta[weights_length:2*weights_length].reshape(visible_size, hidden_size)
    
    my_b1 = theta[2*weights_length:2*weights_length + b1_lenth]
    
    my_b2 = theta[2*weights_length + b1_lenth:] # = b2_length
    
    # total of columns (data to be trained)
    m = data.shape[1]
    
  
    #FP
    my_z2 = my_W1.dot(data) + np.tile(my_b1, (m, 1)).transpose() #np.tile(my_b1, (m, 1)).transpose() = 25x200
    my_a2 = sigmoid(my_z2)
    my_z3 = my_W2.dot(my_a2) + np.tile(my_b2, (m, 1)).transpose()
    my_h = sigmoid(my_z3)
    
  
  
    #Our cost func.
    cost = (0.5 * np.sum(np.power((my_h - data), 2)) / m) + ((lambda_ / 2.0) * \
	( np.sum(np.power(my_W1, 2)) + np.sum(np.power(my_W2, 2)) ) )
    
    #deltas
    delta_output_layer = -(data - my_h) * sigmoid_derivative(my_z3)
    delta_output_layer = -(data - my_h) * (my_h *(1 - my_h))
    delta_middle_layer = my_W2.transpose().dot(delta_output_layer) * sigmoid_derivative(my_z2)
    delta_middle_layer = (my_W2.transpose().dot(delta_output_layer)) * (my_a2*(1 - my_a2))
    
   
   #grad arrays
    my_W2_grad = delta_output_layer.dot(my_a2.transpose())
    my_W1_grad = delta_middle_layer.dot(data.transpose())
    my_b1_grad =  np.sum(delta_middle_layer, axis=1) / m
    my_b2_grad =  np.sum(delta_output_layer, axis=1) / m
    
    
    #shapping grad arrays
    my_W1_grad = my_W1_grad.reshape(hidden_size * visible_size)
    my_W2_grad = my_W2_grad.reshape(hidden_size * visible_size)
    my_b1_grad = my_b1_grad.reshape(hidden_size)
    my_b2_gra = my_b2_grad.reshape(visible_size)
    
    
    #concatenating gra arrays
    grad = np.concatenate((my_W1_grad, my_W2_grad, \
		    my_b1_grad, my_b2_grad))
    
    
    
    return cost, grad

# visible_size: the number of input units (probably 64)
# hidden_size: the number of hidden units (probably 25)
# lambda_: weight decay parameter
# sparsity_param: The desired average activation for the hidden units (denoted in the lecture
#                            notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
#
# The input theta is a vector (because minFunc expects the parameters to be a vector).
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
# follows the notation convention of the lecture notes.
# Returns: (cost,gradient) tuple

def sparse_autoencoder(theta, hidden_size, visible_size, data):
    """
    :param theta: trained weights from the autoencoder
    :param hidden_size: the number of hidden units (probably 25)
    :param visible_size: the number of input units (probably 64)
    :param data: Our matrix containing the training data as columns.  So, data[:,i] is the i-th training example.
    """

    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.
    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    return a2
