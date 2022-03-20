# For python code, the "comment block" command Alt + Shift + A actually wraps the selected text in a multiline string, whereas Ctrl + / is the way to toggle any type of comment (including a "block" comment as asked here). ctrl + shift + A worked for me for adding multi-line comment.
import torch
import numpy as np
# print(torch.__version__)
# x=torch.rand(3)
# print(x)
# print(torch.cuda.is_available())


#x = torch.empty(2,2) #torch.zeros or torch.ones(2,3 , dtype=torch.int)
#x = torch.tensor([2.3.0.8]) this way also we cn create a tensor
#print(x.dtype)
#print(x.size())

# x=torch.rand(2,2)
# y=torch.rand(2,2)
# print(x)
# print(y)
# z=x+y # z= torch.add(x,y)
# print(z)

# # inplace addition
# y.add_(x) # _ after add denotes inplace opertion
# print(y)

# z = x-y # z=torch.sub(x,y)
# print(z)

# z = x*y # z=torch.mul(x,y)
# print(z)
# y.mul_(x) #inplace multplication
# print(y)

# z = x/y # z=torch.div(x,y)
# print(z)

''' slicing operation  '''
# x=torch.rand(5,3)
# print(x)
# print(x[:,0])
# print(x[1,:])
# print(x[1,1])
# print(x[1,1].item()) #we use item method only whrn we have one element to print acual value

''' reshaping tensor '''
# x=torch.rand(4,4)
# print(x)
# y=x.view(16)
# print(y)
# y=x.view(-1,8)
# print(y.size())

''' converting from numpy to tensor and vice versa '''
''' torch to numpy '''
# a=torch.ones(5)
# print(a)
# b=a.numpy()
# print(b)

# # if we have same memory like cpu not cpu and gpu then changes made in one will be reflected in the other as below
# a.add_(1)
# print(a)
# print(b)
''' numpy to torch '''
# a= np.ones(5)
# print(a)
# b=torch.from_numpy(a)
# print(b)
# a+=1
# print(a)
# print(b)

''' cuda gpu '''
# if torch.cuda.is_available():
#   device = torch.device("cuda")
#   x=torch.ones(5,device=device)
#   y=torch.ones(5)
#   y=y.to(device)
#   z=x+y
#   z=z.to("cpu")

x= torch.ones(5, requires_grad=True) # requires_grad true means later you want to perform gradent descent wrt x
print(x)




