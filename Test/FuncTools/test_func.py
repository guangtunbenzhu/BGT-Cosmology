import numpy as np
import functools
import operator

a = np.ones(10000)
for i in np.arange(1000):
    b = functools.reduce(operator.mul, a)

print(b)
