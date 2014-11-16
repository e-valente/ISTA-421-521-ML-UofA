#!/usr/bin/python
import math, sys

def poisson_probability(y, lamb):
	sum = 0.0;
	for i in range(y + 1):
		sum += (math.pow(lamb, i) * math.exp(-lamb)) / math.factorial(i)
	return sum
	
y = 6
lamb = 8
prob = poisson_probability(y, lamb)
print "The Poisson probability is ", prob


