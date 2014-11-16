

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

## Define points for the x-axis
#from 0.0 to 10.0 (steps: 0.01)
x = np.arange(0, 10, 0.01);


#Define x and y labels as well as title
plt.xlabel("x values")
plt.ylabel("sin(x)")
plt.title("Sine Function for x from 0.0 to 10.0 ")
#our grid (just for a better reading)
plt.grid()

#Ploting our function and its label
plt.plot(x, np.sin(x), label="y = sin(x)")


#holds our graph on screen
raw_input("\nPress <ENTER> to exit...");

