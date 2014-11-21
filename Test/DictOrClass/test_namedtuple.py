import collections

Obj = collections.namedtuple('Obj', 'i l')

all = {}
for i in range(1000000):
  all[i] = Obj(i, [])
