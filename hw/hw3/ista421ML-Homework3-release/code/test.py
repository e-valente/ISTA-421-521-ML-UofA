import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
#t = np.arange(0., 5., 0.2)
x = np.arange(-10, 10, 0.2)

# red dashes, blue squares and green triangles
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.plot(x, 1 + (0.1 * x) + (0.5 * np.power(x,2)) + (0.05 * np.power(x,3)), 'r--')
plt.show()