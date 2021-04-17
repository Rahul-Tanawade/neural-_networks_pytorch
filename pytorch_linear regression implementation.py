import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#preparing a dummy dataset with 100 features to implement linear regression

x_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

X=torch.from_numpy(x_numpy.astype(np.float32))# convert numpy to tensor
y=torch.from_numpy(y_numpy.astype(np.float32))# convert numpy to tensor

print(X.shape) #torch.Size([100, 1])
print(y.shape) #torch.Size([100])
y=y.view(y.shape[0],1)#transform it into one column
print(y.shape)#torch.Size([100, 1])


#1 defining a linear model of one layer only
n_samples , n_features = X.shape
input_size=n_features
output_size=1

model=nn.Linear(input_size,output_size)

#2 define loss and optimizer
criterion=nn.MSELoss()

learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#3 training loop

"""We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch."""
num_epochs=200

for epoch in range(num_epochs):
  #forward pass
  y_predicted=model(X)
  loss=criterion(y_predicted,y)

  #backward pass
  loss.backward()#perform backward pass and calculate gradient

  #update parameters
  optimizer.step()

  #zero all gradients
  optimizer.zero_grad()

  if (epoch % 10 ==0):
    print( "epoch -> {}  loss -> {:.4f} ".format(epoch,loss))


#all parameters of linear model are finalised now

y_prdicted=model(X).detach().numpy()# DETACH()is used to stop including y_predicted in dynamic graph

#plotting
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,y_prdicted,'b')
plt.show()



