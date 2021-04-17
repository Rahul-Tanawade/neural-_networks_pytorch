#every thing in pytorch is tensor

import torch

a=torch.empty(1)

print(a)

b=torch.empty(2)

print(b)

c=torch.zeros(2,2)

print(c)


d=torch.ones(5)

print(d)

print(d.size())

print(d.dtype)

e=torch.ones(5,dtype=int)

print(e.dtype)


# requires_grad argument
# This will tell pytorch that it will need to calculate the gradients for this tensor
# later in your optimization steps
# i.e. this is a variable in your model that you want to optimize
k = torch.tensor([5.5, 3], requires_grad=True)
print(k)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

x=torch.rand(2,2)
y=torch.rand(2,2)

z=x+y
print(z)

y=x-y
print(y)

#inplace operations , all with '_' are inplace functions 
print(x)
x.add_(y)
print(x)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#from numpy to torch
import numpy as np
a=np.ones(5)
print(a)
print(type(a))

import torch
b=torch.from_numpy(a)
print(b)
print(type(b))


#rom torch to  numpy
x=torch.ones(5)
print(x)
print(type(x))
y=x.numpy()
print(y)
print(type(y))
