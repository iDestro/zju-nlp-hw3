import inspect
from collections import OrderedDict


def printab(a, b, c, d):
    print(a, b, c, d)


params = inspect.signature(printab).parameters
params = OrderedDict(params)
out = {}
out[printab.__name__] = params

for key, param in out[printab.__name__].items():
    print(key, param)

