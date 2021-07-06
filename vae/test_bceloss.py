import torch
import torch.nn as nn

x = torch.randint(low=0, high=1, size=(10,1,28,28), dtype=torch.float)
xhat = torch.rand((10, 1, 28, 28),)
xhat = x
bce_loss = nn.BCELoss(reduction='none')
bce = bce_loss(xhat, x)
print(bce.shape)
sum_bce = torch.sum(bce)
print(sum_bce)

mybce_loss = x * torch.log(xhat) + (1 - x) * torch.log(1 - xhat)
print(mybce_loss.shape)
print(torch.sum(mybce_loss))





