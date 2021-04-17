#torch. autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.
#It requires minimal changes to the existing code -
#you only need to declare Tensor s for which gradients should be computed with the requires_grad=True keyword.

#Each Tensor has a something an attribute called grad_fn , 
#which refers to the mathematical operator that create the variable. If requires_grad is set to False, grad_fn would be None.
#if tensor is created from existing tensor with requires_grad=True, its included in Dynamic computation graph

import torch

x=torch.rand(3,requires_grad=True)
print(x) #tensor([0.5514, 0.6389, 0.1304], requires_grad=True)

y=x+2
print(y)#tensor([2.5514, 2.6389, 2.1304], grad_fn=<AddBackward0>)
print(y.grad_fn)#<AddBackward0 object at 0x7fc4a6c4ae48>

z=y*y*2
z=z.mean()#scalar ouptut will be produces and hence backward() can be directly applied
print(z)#tensor(12.0078, grad_fn=<MeanBackward0>)

# When we finish our computation we can call .backward() and have all the gradients computed automatically.
# The gradient for this tensor will be accumulated into .grad attribute.
# It is the partial derivate of the function w.r.t. the tensor
z.backward()
print(x.grad) #dz/dx , tensor([3.4018, 3.5185, 2.8405])

----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Model with non-scalar output:
# If a Tensor is non-scalar (more than 1 elements), we need to specify arguments for backward() 
# specify a gradient argument that is a tensor of matching shape.
import torch

x=torch.rand(3,requires_grad=True) 
print(x)#tensor([0.0394, 0.9776, 0.2795], requires_grad=True)

y=x+2
print(y)#tensor([2.0394, 2.9776, 2.2795], grad_fn=<AddBackward0>)
print(y.grad_fn)#<AddBackward0 object at 0x7fc4a74c5da0>

z=y*y*2
#scalar ouptut will be produces and hence backward() can be directly applied
print(z)#tensor([ 8.3185, 17.7318, 10.3920], grad_fn=<MulBackward0>)

#define v to avoide runtime error: RuntimeError: grad can be implicitly created only for scalar outputs
v=torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
z.backward(v)
print(x.grad) #dz/dx , tensor([1.1141e+00, 1.0773e+01, 9.9138e-03])

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#requires_grad indicates whether a variable is trainable. By default, requires_grad is False in creating a Variable.
#If one of the input to an operation requires gradient, its output and its subgraphs will also require gradient
  
#requires_grad = True they start forming a backward graph that tracks every operation applied on them to calculate the gradients 
#using something called a dynamic computation graph (DCG)


#Each Tensor has a something an attribute called grad_fn ,
#which refers to the mathematical operator that create the variable. If requires_grad is set to False, grad_fn would be None

# -------------
# Stop a tensor from tracking history:
# For example during our training loop when we want to update our weights
# then this update operation should not be part of the gradient computation

#3 ways to do it:
# - x.requires_grad_(False)
# - x.detach()
# - wrap in 'with torch.no_grad():'

####using detach_
#you can only change requires_grad flags of leaf variables.
#If you want to use a computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().

print("\nusing detach_")
import torch
x=torch.ones(5,requires_grad=True)
print(x) #tensor([1., 1., 1., 1., 1.], requires_grad=True)

y=x+2
print(y.grad_fn)#<AddBackward0 object at 0x7f20461dd8d0>
print(y)#tensor([3., 3., 3., 3., 3.], grad_fn=<AddBackward0>)
#y.requires_grad_(False) this produces error
y.detach_()

print(y.grad_fn)#none
print(y)#tensor([3., 3., 3., 3., 3.])

####using requires_grad_(false)
print("\nusing requires_grad_")

a=torch.ones(5, requires_grad=True)#tensor([1., 1., 1., 1., 1.], requires_grad=True)
print(a)

b=a+2
print(b)#tensor([3., 3., 3., 3., 3.], grad_fn=<AddBackward0>)

a.requires_grad_(False)
print(a)#tensor([1., 1., 1., 1., 1.])
c=a+10
print(c)#tensor([11., 11., 11., 11., 11.])
print(c.requires_grad)#false
c.requires_grad_(True)
print(c)#tensor([11., 11., 11., 11., 11.], requires_grad=True)
print(c.requires_grad)#true

#########using torch.no_grad()
print("\n using torch.no_grad()")
j=torch.ones(5,requires_grad=True)
print(j.requires_grad)#true
with torch.no_grad():
  k=j+5
print(k) #tensor([6., 6., 6., 6., 6.])
print(k.grad_fn)#None
print(k.requires_grad)#false

-------------------------------------------------------------------------------------------------------------------------------------------------------

# -------------
# backward() accumulates the gradient for this tensor into .grad attribute.
# 
# Use .zero_() to empty the gradients before a new optimization step.
#if this is not used then, weight.grad below will calculated and added to weight.grad while calculating same.

#no emptying the gradeint in below
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    # just a dummy example
    model_output = (weights*3).sum()
    print("weight.grad calculated before backward (no .zero-())-->",weights.grad)
   
    model_output.backward()# print("calculating gradient")
    
    print("weight.grad calculated after backward-->",weights.grad)#over here old weight.grad is getting added
    print("\n")
    
"""output:
weight.grad calculated before backward (no .zero-())--> None
weight.grad calculated after backward--> tensor([3., 3., 3., 3.])


weight.grad calculated before backward (no .zero-())--> tensor([3., 3., 3., 3.])
weight.grad calculated after backward--> tensor([6., 6., 6., 6.])


weight.grad calculated before backward (no .zero-())--> tensor([6., 6., 6., 6.])
weight.grad calculated after backward--> tensor([9., 9., 9., 9.])
"""

--------------------------------------------------------------------------------------------------------------------------------------------------------------

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    # just a dummy example
    model_output = (weights*3).sum()
    print("weight.grad calculated before backward (with .zero-())-->",weights.grad)
    model_output.backward()
    
    print("weight.grad calculated- after backward->",weights.grad)

    # weights.grad -> make it zeroIt affects the final weights & output
    weights.grad.zero_()##over here old weight.grad will not be  getting added
    print("weight.grad after zero()-->",weights.grad)
    print("\n")

"""output
weight.grad calculated before backward (with .zero-())--> None
weight.grad calculated- after backward-> tensor([3., 3., 3., 3.])
weight.grad after zero()--> tensor([0., 0., 0., 0.])


weight.grad calculated before backward (with .zero-())--> tensor([0., 0., 0., 0.])
weight.grad calculated- after backward-> tensor([3., 3., 3., 3.])
weight.grad after zero()--> tensor([0., 0., 0., 0.])


weight.grad calculated before backward (with .zero-())--> tensor([0., 0., 0., 0.])
weight.grad calculated- after backward-> tensor([3., 3., 3., 3.])
weight.grad after zero()--> tensor([0., 0., 0., 0.])
"""
