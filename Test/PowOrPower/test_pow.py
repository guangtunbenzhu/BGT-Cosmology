import numpy as np

a = {}
for i in np.arange(100000):
    a[i] = pow(i/100., 10)
