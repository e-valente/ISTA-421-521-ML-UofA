## plotlinear.py

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

## Define two points for the x-axis
x = np.array([-50, 50])

#line 1

w0 = np.array([20, -13, 1])
w1 = np.array([-1, 2, 0.3])


plt.figure(0)

plt.xlabel("X")
plt.ylabel("Y")
plt.grid()

for i in range(w0.shape[0]):  #or w1.shape[1]
	plt.plot(x, w0[i] + w1[i]*x, label="y = " + str(w0[i]) + \
		" + " + str(w1[i]) + " x \nintercept: " + str(w0[i]) + \
		"\ngradient: " + str(w1[i]))
	plt.legend(loc=4)
	print "\ny = " + str(w0[i]) + " + " + str(w1[i]) + " x";


raw_input("\nPress <ENTER> to exit...");

