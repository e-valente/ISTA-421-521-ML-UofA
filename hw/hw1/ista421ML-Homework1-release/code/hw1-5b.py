## plotsin.py

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

## Define points for the x-axis
#from 0.0 to 10.0 (steps: 0.01)
#x = np.arange(0, 1000, 0.01);
x = np.arange(0, 76367.5);

#plt.figure(0)

#Define x and y labels as well as title
plt.xlabel("t values")
plt.ylabel("H")
plt.title("Graphic 1")
#our grid (just for a better reading)
plt.grid()

#Ploting our function and its label
plt.plot(x, np.power(3.1623 - (x*4.14 * 0.00001), 2), label="H = 3.1623 - (4.14*(10^-5)t ")
#plt.plot(x, np.power((x), 2), label="y = sin(x)")
#loc = 4 bottom right
plt.legend(loc=3)

#holds our graph on screen
raw_input("\nPress <ENTER> to exit...");

