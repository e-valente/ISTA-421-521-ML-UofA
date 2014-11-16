## plotlinear.py

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

## Define two points for the x-axis
x = np.array([-5, 5])

## Define the different intercepts and gradients to plot
w0 = np.arange(0, 20);
w1 = np.arange(0, 8, 0.4);

## Plot all of the lines
plt.figure(0)
plt.plot()

for i in range(w0.shape[0]):
    plt.plot(x, w0[i] + w1[i]*x)
    print "\ny = " + str(w0[i]) + " + " + str(w1[i]) + " x";

raw_input("\nPress <ENTER> to start creating your own lines...");

## Request user input
plt.close();
plt.figure(1)
plt.plot()
print "\nKeeps plotting lines on the current plot until you quit (Ctrl-D)"
print "   or enter a non-number (will throw error, but that's OK).\n"
while 1:
    intercept = float(raw_input("Enter intercept: "))
    gradient = float(raw_input("Enter gradient (slope): "))
    plt.plot(x, intercept + gradient*x)
    print "\ny = " + str(intercept) + " + " + str(gradient) + " x\n";

