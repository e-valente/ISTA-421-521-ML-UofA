## plotlinear.py

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

## Define two points for the x-axis
x = np.array([-50, 50])

#line 1

w0 = np.array([11, -5, 2])
w1 = np.array([0.2, 4, -1.5])


plt.figure(0)

plt.xlabel("X")
plt.ylabel("Y")

for i in range(w0.shape[0]):  #or w1.shape[1]
	plt.plot(x, w0[i] + w1[i]*x)
		#" + " + str(w1[i]) + " x \nintercept: " + str(w0[i]) + \
		#"\ngradient: " + str(w1[i]))
	#plt.legend(loc=4)
	#print "\ny = " + str(w0[i]) + " + " + str(w1[i]) + " x";


raw_input("\nPress <ENTER> to exit...");

