''' Linear Regression Logistic Regression
    1. GRADIENT DESCENT IMPLEMENT --
        Prediction : Manually
        gradients computation : manually
        loss computayion : manually
        parameter updates : manually
    2. GRADIENT DESCENT IMPLEMENT --
        Prediction : Manually
        gradients computation : Autograd
        loss computayion : manually
        parameter updates : manually
    3. GRADIENT DESCENT IMPLEMENT --
        Prediction : Manually
        gradients computation : Autograd
        loss computayion : PyTorch Loss
        parameter updates : PyTorch Optimizer
    4. GRADIENT DESCENT IMPLEMENT --
        Prediction : PyTorch Model
        gradients computation : Autograd
        loss computayion : PyTorch Loss
        parameter updates : PyTorch Optimizer'''
# steps 1 and 2 are covered here

''' Linear Regression '''
''' Step 1 '''
# import numpy as np
# # f = w*x
# # f = 2*x
# X=np.array([1,2,3,4], dtype=np.float32)
# Y=np.array([2,4,6,8], dtype=np.float32)
# w=0
# # model prediction
# def forward(x):
#   return w*x

# # loss = MSE - mean squaared error
# def loss(y, y_predicted):
#   return ((y_predicted-y)**2).mean()
# # gradient
# # MSE = 1/N*(w*x-y)**2
# # dJ/dw = 1/N 2x (w*x -y)
# def gradient(x,y,y_predicted):
#   return np.dot(2*x, y_predicted-y).mean()

# print(f'Prediction before training: f(5) = {forward(5):.3f}')

# # Training
# learning_rate = 0.01
# n_iters = 20

# for epoch in range(n_iters):
#   # prediction  forward pass
#   y_pred = forward(X)
#   # loss
#   l=loss(Y,y_pred)
#   # gradient
#   dw = gradient(X,Y,y_pred)

#   # update weight
#   w-=learning_rate*dw
#   if epoch % 1 == 0:
#     print(f'epoch {epoch+1}: w={w:.3f}, loss = {l:.8f}')

# print(f'Prediction after training: f(5) = {forward(5):.3f}')





''' Step 2 '''

import torch
# f = w*x
# f = 2*x
X=torch.tensor([1,2,3,4], dtype=torch.float32)
Y=torch.tensor([2,4,6,8], dtype=torch.float32)
w=torch.tensor(5.0, dtype=torch.float32, requires_grad=True)
# model prediction
def forward(x):
  return w*x

# loss = MSE - mean squaared error
def loss(y, y_predicted):
  return ((y_predicted-y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
  # prediction  forward pass
  y_pred = forward(X)
  # loss
  l=loss(Y,y_pred)
  # gradient = backward pass
  l.backward() # dl/dw

  # update weight
  with torch.no_grad():
    w-=learning_rate*w.grad
  # zero gradient
  w.grad.zero_()
  if epoch % 1 == 0:
    print(f'epoch {epoch+1}: w={w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

