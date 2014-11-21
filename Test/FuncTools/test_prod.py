import numpy as np

a = np.ones(10000)
for i in np.arange(1000):
    b = np.prod(a)

print(b)
