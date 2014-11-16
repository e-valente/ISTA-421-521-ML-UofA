## plotsin.py

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

## Define points for the x-axis
#from 0.0 to 10.0 (steps: 0.01)
#x = np.arange(0, 1000, 0.01);
x = np.arange(0, 1000);

#plt.figure(0)

#Define x and y labels as well as title
plt.xlabel("t values")
plt.ylabel("N")
plt.title("Graphic 2")
#our grid (just for a better reading)
plt.grid()

#Ploting our function and its label
plt.plot(x, 30000 *np.exp(-0.3*np.exp(-0.05*x) -0.001*x), label="N = 30000 * exp(-0.3*exp(-0.05*t) -0.001*t)")
#plt.plot(x, np.power((x), 2), label="y = sin(x)")
#loc = 4 bottom right
plt.legend(loc=4)

#holds our graph on screen
raw_input("\nPress <ENTER> to exit...");

