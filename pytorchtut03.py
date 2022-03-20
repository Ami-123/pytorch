import torch
# x = torch.randn(3, requires_grad=True)
# print(x) #tensor([-0.5015,  0.1533, -0.5941], requires_grad=True)
# y=x+2
# print(y) #tensor([1.4985, 2.1533, 1.4059], grad_fn=<AddBackward0>)

# z=y*y*2
# print(z) #tensor([15.7885, 20.8727,  1.3202], grad_fn=<MulBackward0>)
# # z=z.mean()
# # print(z) #tensor(4.0901, grad_fn=<MeanBackward0>)

# ''' Jaccobian vector product J.v '''
# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# print(z.backward(v)) # dz/dx - if not scalar the pass a vector value.
# print(x.grad)


''' how we can prevent pytorch from tracking history and calculating grad_fn attribute'''
#1. x.requires_grad_(False)
#2. x.detach()
#3. with torch.no_grad()

# 1st method
# x.requires_grad_(False)
# print(x)

# 2nd method
# y = x.detach()
# print(y)

# 3rd method
# with torch.no_grad():
#   y=x+2
#   print(y)



''' when we call backward fn then gradient fn this tensor will be accumulated into the .grad attribute '''

#weights = torch.ones(4, requires_grad=True)
# for epoch in range(3):
#   model_output = (weights*3).sum()

#   model_output.backward()

#   print(weights.grad)
#   weights.grad.zero_() # important step during training steps - before moving to next iteration we should make grad 0

# weights = torch.ones(4, requires_grad=True)
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()
