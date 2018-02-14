"""
使用nnPackage搭建一个简单的2层神经网络，同时我也将它改造为GPU运行的版本
"""
import torch
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

N, D_in, H, D_out = 61, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype))
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H).type(dtype),
    torch.nn.ReLU().type(dtype),
    torch.nn.Linear(H, D_out).type(dtype),
)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    model.zero_grad()

    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

print(model)
