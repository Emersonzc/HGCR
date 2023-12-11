import torch

a = torch.Tensor([[0,1],[2,3],[4,5]])
b = torch.Tensor([[0,1],[2,3],[3,4],[4,5]])

superset = torch.cat([a, b])
uniset, count = superset.unique(return_counts=True)

mask = (count == 1)
result = uniset.masked_select(mask)

print(result)

