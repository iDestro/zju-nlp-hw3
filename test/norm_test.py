import torch

a = torch.Tensor([1, 2, 3, 4])

lin = lambda x: x / torch.norm(x)

print(lin(a))