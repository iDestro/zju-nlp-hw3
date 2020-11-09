import torch


x = torch.randn((1, 10556, 1))
y = torch.randn((10556, 3, 32))

print(x.expand_as(y).shape)