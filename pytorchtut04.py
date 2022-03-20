''' BACKPROPOGATION
    steps - 1. Forward pass : Compute loss
            2. Compute local gradients
            3. Backward pass: compute dLoss/dWeights usng the chain rule'''
import torch
x =torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad = True)
# forward pass and compute the loss
y_hat = w*x
loss = (y_hat-y)**2
print(loss)
# backward pass
loss.backward()
print(w.grad)

# update weights
# next forward and backward
