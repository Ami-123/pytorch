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
# steps 3 and 4 are covered here

''' Pipeline
    1) Design model(input, otput size, forward pass)
    2) construct loss and optimizer
    3) Training Loop
        - forward pass : compute prediction
        - backward pass : gradients
        - update weights'''
# # step 3
# import torch
# import torch.nn as nn
# # f = w*x
# # f = 2*x
# X=torch.tensor([1,2,3,4], dtype=torch.float32)
# Y=torch.tensor([2,4,6,8], dtype=torch.float32)
# w=torch.tensor(5.0, dtype=torch.float32, requires_grad=True)
# # model prediction
# def forward(x):
#   return w*x

# # loss = MSE - mean squaared error
# print(f'Prediction before training: f(5) = {forward(5):.3f}')

# # Training
# learning_rate = 0.01
# n_iters = 100
# loss = nn.MSELoss()
# optimizer = torch.optim.SGD([w], lr=learning_rate)
# for epoch in range(n_iters):
#   # prediction  forward pass
#   y_pred = forward(X)
#   # loss
#   l=loss(Y,y_pred)
#   # gradient = backward pass
#   l.backward() # dl/dw

#   # update weight
#   optimizer.step()
#   # zero gradient
#   optimizer.zero_grad()
#   if epoch % 1 == 0:
#     print(f'epoch {epoch+1}: w={w:.3f}, loss = {l:.8f}')

# print(f'Prediction after training: f(5) = {forward(5):.3f}')









# step 4
import torch
import torch.nn as nn
# f = w*x
# f = 2*x
X=torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)
# prediction PyTorch model
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

''' Below class will do exactly same work as model above '''
# class LinearRegression(nn.Module):
#   def __init__(self, input_dim, output_dim):
#     super(LinearRegression, self).__init__()
#     #define layers
#     self.lin = nn.Linear(input_dim, output_dim)

#   def forward(self, x):
#     return self.lin(x)

# model = LinearRegression(input_size, output_size)


# loss = MSE - mean squaared error
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(n_iters):
  # prediction  forward pass
  y_pred = model(X)
  # loss
  l=loss(Y,y_pred)
  # gradient = backward pass
  l.backward() # dl/dw

  # update weight
  optimizer.step()
  # zero gradient
  optimizer.zero_grad()
  if epoch % 1 == 0:
    [w,b] = model.parameters()
    print(f'epoch {epoch+1}: w={w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')



