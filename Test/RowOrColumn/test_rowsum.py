import numpy as np

x = np.ones((10000, 10000))
for i in np.arange((x.shape)[1]):
    x[:,i] *= 3.

y = np.sum(x, axis=1)
