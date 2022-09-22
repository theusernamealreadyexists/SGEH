import torch
pro = torch.Tensor([[1,1],[1,1]])
size = 1
a  = torch.arange(5)
print(a)
#
# pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(
#     -1)] = 0.
# pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(
#     -1)] = 0.